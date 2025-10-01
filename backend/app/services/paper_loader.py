import arxiv
from pathlib import Path
import multiprocessing
import re
from grobid_client.grobid_client import GrobidClient
from langchain_community.document_loaders.parsers import GrobidParser
from langchain_community.document_loaders.generic import GenericLoader
from langchain.schema import Document
from langchain_community.document_loaders.base import BaseBlobParser
from app.db.models import Paper
from app.core.config import settings
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy import create_engine
from pydantic import BaseModel
from sqlalchemy.orm import Session
import yaml
from app.core.config import settings, PaperLoaderConfig
from .elasticsearch import ElasticsearchService
import json
from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime
from app.db.schema import ArxivPaper

def filter_json_stream(file_path, filter_values):
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

    # def metadata_workers(self, i: int, offset: int):
    #     with open(self.config.arxiv_metadata_path, 'r', encoding='utf-8') as file:
    #         file.seek(offset)
    #         for _ in range(self.config.batch_size):
    #             line = file.readline()


    # def load_by_metadata(self):
        # if not self.config.arxiv_metadata_path:
        #     raise ValueError("arxiv_metadata_path is required")
        # id_list = []
        # cnt = self.config.batch_size * self.config.workers
        # for item in filter_json_stream(self.config.arxiv_metadata_path, ['cs.CL']):
        #     id_list.append(item['id'])
        #     cnt -= 1
        #     if cnt == 0:
        #         break

        # client = arxiv.Client()
        # search = arxiv.Search(id_list=id_list)
        # results = client.results(search)
        # for result in results:
        #     file_name = re.sub(r'[<>:"/\\|?*]', '_', result.title.replace(" ", "_"))
        #     result.download_pdf(dirpath=self.config.output_dir, filename=file_name+".pdf")
    
    def load_by_metadata(self, categories_filter: List[str] = None, limit: int = None):
        """
        Load papers from JSON file and index them into Elasticsearch.
        
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
                    result = self.elasticsearch_service.add_papers_bulk(papers_batch)
                    print(f"Indexed batch: {result}")
                    papers_batch = []
                
                if limit and processed >= limit:
                    break
                    
            except Exception as e:
                print(f"Error processing paper {paper_data.get('id', 'unknown')}: {e}")
                continue
        
        # Index remaining papers
        if papers_batch:
            result = self.elasticsearch_service.add_papers_bulk(papers_batch)
            print(f"Indexed final batch: {result}")
        
        print(f"Total papers processed: {processed}")
        
        