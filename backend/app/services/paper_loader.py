import arxiv
from pathlib import Path
import multiprocessing
import re
from grobid_client.grobid_client import GrobidClient
from langchain_community.document_loaders.parsers import GrobidParser
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.base import BaseBlobParser
from app.db.models import Paper
from app.core.config import settings
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy import create_engine
from pydantic import BaseModel
from sqlalchemy.orm import Session
import yaml
from app.core.config import settings, PaperLoaderConfig
from app.services.qdrant import QdrantService
from .elasticsearch import ElasticsearchService
import json
from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime
from app.db.schema import ArxivPaper
from typing import Dict, Any
import time
import tempfile
import shutil
from tqdm import tqdm

# Global worker state (initialized per process)
_worker_embedding_model = None
_worker_es_client = None
_worker_qdrant_client = None
_worker_vector_store = None

def init_metadata_worker():
    """Initialize worker process with embedding model and clients."""
    global _worker_embedding_model, _worker_es_client, _worker_qdrant_client, _worker_vector_store
    from sentence_transformers import SentenceTransformer
    from elasticsearch import Elasticsearch
    from qdrant_client import QdrantClient
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_qdrant import QdrantVectorStore
    
    print(f"Initializing worker process {multiprocessing.current_process().name}...")
    
    # Load embedding model once per worker
    _worker_embedding_model = SentenceTransformer('allenai-specter')
    
    # Initialize Elasticsearch client
    auth = (settings.elasticsearch_config.username, settings.elasticsearch_config.password) \
           if settings.elasticsearch_config.username and settings.elasticsearch_config.password else None
    _worker_es_client = Elasticsearch(
        [settings.elasticsearch_config.url],
        basic_auth=auth,
        verify_certs=False,
        ssl_show_warn=False,
    )
    
    # Initialize Qdrant client with optimized settings
    _worker_qdrant_client = QdrantClient(
        url=settings.qdrant_config.url,
        timeout=300,  # 5 minute timeout for large batches
        prefer_grpc=False  # Use HTTP for better compatibility
    )
    
    # Initialize Qdrant vector store once per worker (reuses embedding model)
    embeddings = HuggingFaceEmbeddings(
        model_name='allenai-specter',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    _worker_vector_store = QdrantVectorStore(
        client=_worker_qdrant_client,
        collection_name=settings.qdrant_config.collection,
        embedding=embeddings
    )
    
    print(f"Worker {multiprocessing.current_process().name} initialized successfully")

def _check_papers_exist(worker_id: int, papers: List[ArxivPaper]) -> List[ArxivPaper]:
    """
    Check which papers already exist in Elasticsearch and filter them out.
    
    Returns:
        List of papers that don't exist yet
    """
    new_papers = []
    for paper in papers:
        try:
            # Check if document exists in Elasticsearch
            exists = _worker_es_client.exists(
                index=settings.elasticsearch_config.index,
                id=paper.id
            )
            if not exists:
                new_papers.append(paper)
            else:
                print(f"Worker {worker_id}: Paper {paper.id} already exists, skipping")
        except Exception as e:
            # If check fails, include the paper to be safe
            print(f"Worker {worker_id}: Error checking paper {paper.id}: {e}")
            new_papers.append(paper)
    
    return new_papers

def _index_papers_batch(worker_id: int, papers: List[ArxivPaper], skip_existing: bool = True, 
                        process_pdfs: bool = False) -> int:
    """
    Index a batch of papers to both Elasticsearch and Qdrant.
    Uses global worker resources.
    
    Args:
        worker_id: Worker identifier
        papers: List of papers to index
        skip_existing: If True, skip papers that already exist in Elasticsearch
        process_pdfs: If True, download PDFs and parse with GROBID (slow). If False, only index metadata.
    
    Returns:
        Number of successfully indexed papers
    """
    from elasticsearch.helpers import bulk as es_bulk
    from langchain_community.document_loaders.generic import GenericLoader
    from langchain_community.document_loaders.parsers import GrobidParser
    from langchain_qdrant import QdrantVectorStore
    from langchain_huggingface import HuggingFaceEmbeddings
    
    if not papers:
        return 0
    
    # Filter out existing papers if requested
    if skip_existing:
        papers = _check_papers_exist(worker_id, papers)
        if not papers:
            print(f"Worker {worker_id}: All papers already exist, skipping batch")
            return 0
    
    try:
        # 1. Index to Elasticsearch (metadata + embeddings)
        def generate_es_docs():
            for paper in papers:
                doc = paper.to_elasticsearch_doc()
                if paper.title:
                    doc['title_vector'] = _worker_embedding_model.encode(paper.title).tolist()
                if paper.abstract:
                    doc['abstract_vector'] = _worker_embedding_model.encode(paper.abstract).tolist()
                if paper.categories:
                    doc['categories_array'] = paper.categories.split()
                
                yield {
                    "_index": settings.elasticsearch_config.index,
                    "_id": paper.id,
                    "_source": doc
                }
        
        es_success, es_errors = es_bulk(
            _worker_es_client,
            generate_es_docs(),
            chunk_size=50,
            request_timeout=60,
            raise_on_error=False
        )
        
        print(f"Worker {worker_id}: Indexed {es_success} papers to Elasticsearch")
        
        # 2. Download PDFs and parse with GROBID (optional, only if process_pdfs=True)
        if not process_pdfs:
            print(f"Worker {worker_id}: Skipping PDF processing (process_pdfs=False)")
            return es_success
        
        temp_dir = tempfile.mkdtemp(prefix=f'worker_{worker_id}_')
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import time
            
            def download_single_pdf(paper_id, temp_dir_path):
                """Download a single PDF from ArXiv."""
                try:
                    arxiv_client = arxiv.Client()
                    search = arxiv.Search(id_list=[paper_id])
                    results = list(arxiv_client.results(search))
                    if results:
                        result = results[0]
                        result.download_pdf(dirpath=temp_dir_path, filename=f"{paper_id}.pdf")
                        return paper_id, True, None
                except Exception as e:
                    return paper_id, False, str(e)
            
            # Download PDFs in parallel (up to 5 concurrent downloads per worker)
            download_start = time.time()
            downloaded_papers = set()
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(download_single_pdf, paper.id, temp_dir): paper for paper in papers}
                for future in as_completed(futures):
                    paper_id, success, error = future.result()
                    if success:
                        downloaded_papers.add(paper_id)
                    else:
                        print(f"Worker {worker_id}: Failed to download {paper_id}: {error}")
            
            download_time = time.time() - download_start
            print(f"Worker {worker_id}: Downloaded {len(downloaded_papers)}/{len(papers)} PDFs in {download_time:.1f}s ({len(downloaded_papers)/download_time:.1f} PDFs/sec)")
            
            # Parse with GROBID (processes all PDFs in the temp directory)
            if not downloaded_papers:
                print(f"Worker {worker_id}: No PDFs downloaded successfully, skipping GROBID parsing")
                return es_success
                
            loader = GenericLoader.from_filesystem(
                temp_dir,
                glob="*.pdf",
                suffixes=[".pdf"],
                parser=GrobidParser(segment_sentences=False)
            )
            
            grobid_start = time.time()
            print(f"Worker {worker_id}: Starting GROBID parsing for {len(downloaded_papers)} PDFs...")
            docs = loader.load()
            grobid_time = time.time() - grobid_start
            print(f"Worker {worker_id}: GROBID parsed {len(docs)} document chunks in {grobid_time:.1f}s ({len(docs)/grobid_time:.1f} chunks/sec)")
            
            # Match docs to papers and update metadata
            if docs:
                # Map paper metadata to parsed docs
                for doc in docs:
                    # Extract paper ID from filename
                    source_path = doc.metadata.get('source', '')
                    paper_id = Path(source_path).stem if source_path else None
                    matching_paper = next((p for p in papers if p.id == paper_id), None)
                    if matching_paper:
                        doc.metadata.update(matching_paper.model_dump())
                
                # Index to Qdrant using pre-initialized worker vector store
                # This reuses the embedding model loaded during worker initialization
                import time
                start_time = time.time()
                
                # Split into smaller chunks for better progress feedback
                chunk_size = 50
                total_indexed = 0
                for i in range(0, len(docs), chunk_size):
                    chunk = docs[i:i+chunk_size]
                    _worker_vector_store.add_documents(chunk)
                    total_indexed += len(chunk)
                    
                elapsed = time.time() - start_time
                print(f"Worker {worker_id}: Indexed {total_indexed} documents to Qdrant in {elapsed:.1f}s ({total_indexed/elapsed:.1f} docs/sec)")
            else:
                print(f"Worker {worker_id}: No documents parsed from PDFs")
        
        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return es_success
        
    except Exception as e:
        print(f"Worker {worker_id}: Batch indexing error: {e}")
        import traceback
        traceback.print_exc()
        return 0

def process_metadata_chunk(worker_id: int, start_line: int, end_line: int, 
                          json_path: str, categories_filter: List[str],
                          batch_size: int = 50, skip_existing: bool = True, 
                          progress_dict=None, process_pdfs: bool = False):
    """
    Worker function to process a chunk of the metadata JSON file.
    
    Args:
        worker_id: Worker identifier for logging
        start_line: Starting line number (0-indexed)
        end_line: Ending line number (exclusive)
        json_path: Path to ArXiv metadata JSON
        categories_filter: List of category filters
        batch_size: Batch size for indexing
        skip_existing: If True, skip papers that already exist
        progress_dict: Shared dict for progress tracking (optional)
        process_pdfs: If True, download and parse PDFs (slow). If False, only metadata.
    """
    from itertools import islice
    
    papers_batch = []
    processed = 0
    errors = 0
    lines_read = 0
    
    total_lines = end_line - start_line
    
    with open(json_path, 'r', encoding='utf-8') as f:
        # Efficiently skip to start line using islice
        for _ in islice(f, start_line):
            pass
        
        # Process lines in range
        for line_num in range(start_line, end_line):
            line = f.readline()
            lines_read += 1
            
            # Update shared progress every 100 lines
            if progress_dict is not None and lines_read % 100 == 0:
                progress_dict['lines'] = progress_dict.get('lines', 0) + 100
            
            if not line:
                break
                
            line = line.strip()
            if not line:
                continue
            
            try:
                obj = json.loads(line)
                cats = obj['categories'].split(' ')
                
                # Apply category filter
                if categories_filter and not any(cat in cats for cat in categories_filter):
                    continue
                
                # Create ArxivPaper
                paper = ArxivPaper(
                    id=obj['id'],
                    submitter=obj.get('submitter', ''),
                    authors=obj.get('authors', ''),
                    title=obj.get('title', ''),
                    abstract=obj.get('abstract', ''),
                    comments=obj.get('comments'),
                    journal_ref=obj.get('journal-ref'),
                    doi=obj.get('doi'),
                    report_no=obj.get('report-no'),
                    categories=obj.get('categories', ''),
                    license=obj.get('license')
                )
                
                papers_batch.append(paper)
                
                # Process batch when size reached
                if len(papers_batch) >= batch_size:
                    success = _index_papers_batch(worker_id, papers_batch, skip_existing, process_pdfs)
                    processed += success
                    errors += (len(papers_batch) - success)
                    papers_batch = []
                    
                    # Update shared progress
                    if progress_dict is not None:
                        progress_dict['papers'] = progress_dict.get('papers', 0) + success
                        progress_dict['errors'] = progress_dict.get('errors', 0) + errors
                    
            except json.JSONDecodeError as e:
                errors += 1
            except Exception as e:
                errors += 1
        
        # Process remaining papers
        if papers_batch:
            success = _index_papers_batch(worker_id, papers_batch, skip_existing, process_pdfs)
            processed += success
            errors += (len(papers_batch) - success)
            
            # Final progress update
            if progress_dict is not None:
                progress_dict['papers'] = progress_dict.get('papers', 0) + success
                progress_dict['errors'] = progress_dict.get('errors', 0) + errors
        
        # Update final line count
        if progress_dict is not None:
            remaining_lines = lines_read % 100
            if remaining_lines > 0:
                progress_dict['lines'] = progress_dict.get('lines', 0) + remaining_lines
    
    print(f"Worker {worker_id} completed: {processed} indexed, {errors} errors")
    return {'processed': processed, 'errors': errors}

def filter_json_stream(file_path: str, filter_values: List[str]):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            try:
                obj = json.loads(line)
                cats = obj['categories'].split(' ')
                if any(filter_value in cats for filter_value in filter_values):
                    yield obj
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue

def create_db_session():
    engine = create_engine(settings.database_config.url)
    Session = scoped_session(sessionmaker(bind=engine))
    return Session()

def init_worker():
    global session
    session = create_db_session()


class PaperLoader:
    def __init__(self, config: PaperLoaderConfig):
        self.config = config
        # lazy init elasticsearch service
        if not self.config.use_postgres:
            self.elasticsearch_service = ElasticsearchService(settings.elasticsearch_config)
        self.qdrant_service = QdrantService(settings.qdrant_config)
    
    def load_by_search(self, query: str):
        params = [None] * self.config.workers
        offset = 0
        for i in range(self.config.workers):
            params[i] = (i, query, offset)
            offset += self.config.batch_size
        if self.config.use_postgres:
            with multiprocessing.Pool(processes=self.config.workers, initializer=init_worker) as pool:
                pool.starmap(self.search_workers, params)
        else:
            with multiprocessing.Pool(processes=self.config.workers) as pool:
                pool.starmap(self.search_workers, params)

    def search_workers(self, i: int, query: str, offset: int):
        # Construct the default API client.
        client = arxiv.Client()
        # search for the most recent 100 papers in the cs.LG category
        search = arxiv.Search(query=query, max_results=offset + self.config.batch_size, sort_by=arxiv.SortCriterion.SubmittedDate)
        results = client.results(search, offset=offset)

        # create folder if not exists
        Path(self.config.output_dir+f"/{i}").mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading {offset} to {offset + self.config.batch_size - 1}")
        # papers = []
        for result in results:
            # Print all properties of the result
            # Clean the title to create a valid filename
            file_name = re.sub(r'[<>:"/\\|?*]', '_', result.title.replace(" ", "_"))
            result.download_pdf(dirpath=self.config.output_dir+f"/{i}", filename=file_name+".pdf")
            
            full_text = self.parse_pdf(file_name, self.config.output_dir+f"/{i}")

    def parse_pdf(self, file_name: str, folder_path: str):
        loader = GenericLoader.from_filesystem(
            folder_path,
            glob=file_name+".pdf",
            suffixes=[".pdf"],
            parser= GrobidParser(segment_sentences=False)
        )
        docs = loader.load()
        full_text = '\n'.join(doc.page_content for doc in docs)
        return full_text

    def build_line_index(self, file_path: str, batch_size: int, workers: int):
        line_index = [0]
        with open(file_path, 'r', encoding='utf-8') as file:
            for _ in range(workers - 1):
                skip = batch_size
                while skip > 0:
                    line = file.readline()
                    if not line:
                        break
                    skip -= 1
                line_index.append(file.tell())
            return line_index
    
    def load_by_metadata(self, categories_filter: List[str] = None, limit: int = None):
        """
        Load papers from JSON file and index them into Elasticsearch and Qdrant.
        
        Args:
            json_path: Path to the ArXiv JSON file
            categories_filter: Optional list of categories to filter (e.g., ['cs.CL', 'cs.AI'])
            limit: Optional limit on number of papers to process
        """
        json_path = self.config.arxiv_metadata_path
        if not self.elasticsearch_service:
            raise RuntimeError("Elasticsearch service not initialized")
        
        papers_batch = []
        processed = 0
        
        # Process papers from JSON stream
        for paper_data in filter_json_stream(json_path, categories_filter or []):
            try:
                # Create ArxivPaper model from JSON data
                paper = ArxivPaper(
                    id=paper_data['id'],
                    submitter=paper_data.get('submitter', ''),
                    authors=paper_data.get('authors', ''),
                    title=paper_data.get('title', ''),
                    abstract=paper_data.get('abstract', ''),
                    comments=paper_data.get('comments'),
                    journal_ref=paper_data.get('journal-ref'),
                    doi=paper_data.get('doi'),
                    report_no=paper_data.get('report-no'),
                    categories=paper_data.get('categories', ''),
                    license=paper_data.get('license')
                )
                
                papers_batch.append(paper)
                processed += 1
                
                # Index in batches of 50
                if len(papers_batch) >= 50:
                    result = self.elasticsearch_service.add_papers_batch(papers_batch)
                    print(f"Added to Elasticsearch: {result}")
                    result = self.qdrant_service.add_papers_batch(papers_batch)
                    print(f"Added to Qdrant")
                    papers_batch = []
                
                if limit and processed >= limit:
                    break
                    
            except Exception as e:
                print(f"Error processing paper {paper_data.get('id', 'unknown')}: {e}")
                continue
        
        # Index remaining papers
        if papers_batch:
            result = self.elasticsearch_service.add_papers_batch(papers_batch)
            print(f"Added to Elasticsearch: {result}")
            result = self.qdrant_service.add_papers_batch(papers_batch)
            print(f"Added to Qdrant")
        
        print(f"Total papers processed: {processed}")
    
    def load_by_metadata_parallel(self, categories_filter: List[str] = None, 
                                  limit: int = None, skip_existing: bool = True,
                                  process_pdfs: bool = False) -> Dict[str, Any]:
        """
        Parallel version of load_by_metadata using multiprocessing.
        
        Args:
            categories_filter: List of categories to filter
            limit: Optional limit on total papers (approximate)
            skip_existing: If True, skip papers that already exist in Elasticsearch (default: True)
            process_pdfs: If True, download and parse PDFs with GROBID (SLOW - 20x slower!).
                         If False (default), only index metadata with title/abstract embeddings (FAST).
            
        Returns:
            Dict with statistics: total_processed, total_errors, duration
        """
        from multiprocessing import Pool, Manager
        import threading
        
        json_path = self.config.arxiv_metadata_path
        workers = self.config.workers
        batch_size = self.config.batch_size
        
        print(f"📊 Counting total lines in {json_path}...")
        # Count total lines for chunking with progress bar
        with open(json_path, 'r') as f:
            total_lines = sum(1 for _ in tqdm(f, desc="Counting lines", unit="lines"))
        
        # Calculate lines per worker
        if limit:
            total_lines = min(total_lines, limit)
        
        lines_per_worker = total_lines // workers
        
        # Create shared progress tracking using Manager
        manager = Manager()
        progress_dict = manager.dict()
        progress_dict['lines'] = 0
        progress_dict['papers'] = 0
        progress_dict['errors'] = 0
        
        # Build work chunks with progress_dict
        work_chunks = []
        for i in range(workers):
            start_line = i * lines_per_worker
            end_line = start_line + lines_per_worker if i < workers - 1 else total_lines
            work_chunks.append((i, start_line, end_line, json_path, 
                              categories_filter or [], batch_size, skip_existing, progress_dict, process_pdfs))
        
        print(f"\n🚀 Starting parallel processing with {workers} workers...")
        print(f"📄 Total lines to process: {total_lines:,}")
        print(f"📦 Lines per worker: ~{lines_per_worker:,}")
        print(f"📦 Batch size: {batch_size}")
        print(f"🎯 Category filters: {categories_filter or 'None'}")
        print(f"🔄 Skip existing: {'Yes' if skip_existing else 'No'}")
        print(f"📑 Process PDFs: {'Yes (SLOW - full text)' if process_pdfs else 'No (FAST - metadata only)'}")
        print(f"\n{'='*60}\n")
        
        start_time = time.time()
        
        # Start progress monitoring in background thread
        stop_monitor = threading.Event()
        
        def monitor_progress():
            pbar = tqdm(total=total_lines, desc="Overall Progress", unit="lines", position=0)
            last_lines = 0
            while not stop_monitor.is_set():
                current_lines = progress_dict.get('lines', 0)
                current_papers = progress_dict.get('papers', 0)
                current_errors = progress_dict.get('errors', 0)
                
                # Update progress bar
                delta = current_lines - last_lines
                if delta > 0:
                    pbar.update(delta)
                    last_lines = current_lines
                
                # Update postfix with papers and errors
                pbar.set_postfix({
                    'papers': current_papers,
                    'errors': current_errors,
                    'rate': f"{current_papers/(time.time()-start_time):.1f}/s" if time.time() > start_time else "0/s"
                })
                
                time.sleep(0.5)  # Update twice per second
            
            # Final update
            pbar.n = total_lines
            pbar.refresh()
            pbar.close()
        
        monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
        monitor_thread.start()
        
        # Execute parallel processing
        with Pool(processes=workers, initializer=init_metadata_worker) as pool:
            results = pool.starmap(process_metadata_chunk, work_chunks)
        
        # Stop progress monitoring
        stop_monitor.set()
        monitor_thread.join(timeout=2)
        
        # Aggregate results
        total_processed = sum(r['processed'] for r in results)
        total_errors = sum(r['errors'] for r in results)
        duration = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"✅ Parallel processing complete!")
        print(f"{'='*60}")
        print(f"📊 Total processed: {total_processed:,} papers")
        print(f"❌ Total errors: {total_errors:,}")
        print(f"⏱️  Duration: {duration:.2f}s ({duration/60:.1f} min)")
        if duration > 0:
            print(f"⚡ Rate: {total_processed/duration:.2f} papers/sec")
        print(f"{'='*60}")
        
        return {
            'total_processed': total_processed,
            'total_errors': total_errors,
            'duration': duration,
            'rate': total_processed / duration if duration > 0 else 0
        }
        