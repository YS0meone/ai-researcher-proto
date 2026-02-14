import re
import requests

from langchain_core.documents import Document
from app.core.config import QdrantConfig
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
from qdrant_client.http.models import Distance, VectorParams
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.schema import ArxivPaper
from app.core.schema import S2Paper
import arxiv
from pathlib import Path
import logging
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import GrobidParser
from collections import defaultdict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever
from app.agent.RedisDocumentStore import RedisDocumentStore
from langchain_openai import OpenAIEmbeddings
from app.core.config import settings
from langchain_core.runnables import ConfigurableField
from typing import List


embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.OPENAI_API_KEY)
logger = logging.getLogger(__name__)
kv_store = RedisDocumentStore(redis_url="redis://localhost:6379")


class QdrantService:

    def __init__(self, config: QdrantConfig):
        self.config = config
        self.client = QdrantClient(url=self.config.url)

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            add_start_index=True
        )

        collection = self.config.collection
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(
                    size=self.config.vector_size,
                    distance=Distance[self.config.distance.upper()],
                ),
            )

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection,
            embedding=embeddings,
        )

        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vector_store,
            docstore=kv_store,
            child_splitter=self.child_splitter
        ).configurable_fields(
            search_kwargs=ConfigurableField(
                id="search_kwargs",
                name="Search Arguments",
                description="The search kwargs for the retriever"
            )
        )

    
    def download_pdf(self, paper: 'ArxivPaper'):
        client = arxiv.Client()
        result = list(client.results(arxiv.Search(id_list=[paper.id])))
        if not result:
            raise ValueError(f"No result found for paper {paper.id}")
        file_name = re.sub(r'[<>:"/\\|?*]', '_', result[0].title.replace(" ", "_"))
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        result[0].download_pdf(dirpath=self.config.output_dir, filename=file_name+".pdf")
        logging.info(f"Downloaded PDF for paper {paper.id} to {self.config.output_dir}/{file_name}.pdf")

    def download_pdf_batch(self, papers: list['ArxivPaper']):
        client = arxiv.Client()
        search = arxiv.Search(id_list=[paper.id for paper in papers])
        results = client.results(search)
        cnt = 0
        for result in results:
            file_name = re.sub(r'[<>:"/\\|?*]', '_', result.title.replace(" ", "_"))
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
            result.download_pdf(dirpath=self.config.output_dir, filename=file_name+".pdf")
            cnt += 1
        logging.info(f"Downloaded {cnt} papers to {self.config.output_dir}")

    def empty_pdf_folder(self):
        for file in Path(self.config.output_dir).glob("*.pdf"):
            file.unlink()
        logging.info(f"Emptied PDF folder {self.config.output_dir}")

    def add_paper(self, paper: ArxivPaper):
        self.empty_pdf_folder()
        self.download_pdf(paper)
        loader = GenericLoader.from_filesystem(
            self.config.output_dir,
            glob="*",
            suffixes=[".pdf"],
            parser=GrobidParser(segment_sentences=False),
        )
        docs = loader.load()
        if not docs:
            raise ValueError("No docs found")
        for i in range(len(docs)):
            docs[i].metadata.update(paper.model_dump())
        self.vector_store.add_documents(docs)
        logging.info(f"Added paper {paper.id} to Qdrant")
    
    def add_paper_with_chunks(self, paper: ArxivPaper, chunks: list[str]):
        print(paper.model_dump())
        docs = [Document(page_content=chunk, metadata=paper.model_dump()) for chunk in chunks]
        self.vector_store.add_documents(docs)
        logging.info(f"Added {len(docs)} chunks to Qdrant")

    
    def add_papers_batch(self, papers: list['ArxivPaper']):
        self.empty_pdf_folder()
        self.download_pdf_batch(papers)
        loader = GenericLoader.from_filesystem(
            self.config.output_dir,
            glob="*",
            suffixes=[".pdf"],
            parser=GrobidParser(segment_sentences=False),
        )
        docs = loader.load()
        if not docs:
            raise ValueError("No docs found")
        paper_idx = 0
        docs[0].metadata.update(papers[paper_idx].model_dump())
        for i in range(1, len(docs)):
            if docs[i].metadata["paper_title"] != docs[i-1].metadata["paper_title"]:
                paper_idx += 1
            docs[i].metadata.update(papers[paper_idx].model_dump())

        self.vector_store.add_documents(docs)
        logging.info(f"Added {len(papers)} papers to Qdrant")

    # ============================
    # S2Paper ingestion methods
    # ============================

    def _download_s2_pdf(self, paper: S2Paper) -> str:
        """Download open-access PDF for an S2Paper.
        
        Returns:
            The file path of the downloaded PDF.
        
        Raises:
            ValueError: If no open-access PDF URL is available or download fails.
        """
        if not paper.openAccessPdf or not paper.openAccessPdf.get("url"):
            raise ValueError(f"No open-access PDF URL for paper {paper.paperId}")

        pdf_url = paper.openAccessPdf["url"]
        safe_title = re.sub(r'[<>:"/\\|?*]', '_', (paper.title or paper.paperId).replace(" ", "_"))
        file_name = f"{safe_title}.pdf"

        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        file_path = Path(self.config.output_dir) / file_name

        response = requests.get(pdf_url, timeout=60)
        response.raise_for_status()

        file_path.write_bytes(response.content)
        logger.info(f"Downloaded S2 PDF for paper {paper.paperId} to {file_path}")
        return str(file_path)

    def _s2_paper_metadata(self, paper: S2Paper) -> dict:
        """Build a JSON-serializable metadata dict from an S2Paper."""
        data = paper.model_dump(mode="json")
        # Ensure paperId is also stored under 'id' for consistent filtering
        data["id"] = paper.paperId
        return data

    def add_s2_paper(self, file_name: str, paper_id: str) -> int:
        loader = GenericLoader.from_filesystem(
            self.config.output_dir,
            glob=file_name+".pdf",
            suffixes=[".pdf"],
            parser=GrobidParser(segment_sentences=False),
        )
        docs = loader.load()
        if not docs:
            raise ValueError(f"GROBID produced no chunks for paper {file_name}")

        section_docs = defaultdict(list)
        for doc in docs:
            section_docs[doc.metadata.get("section_title", "General")].append(doc)
        new_docs = []
        for docs in section_docs.values():
            metadata = docs[0].metadata.copy()
            del metadata["text"]
            del metadata["file_path"]
            del metadata["bboxes"]
            metadata["id"] = paper_id
            new_docs.append(Document(page_content="\n\n".join([doc.page_content for doc in docs]), metadata=metadata))
        
        self.retriever.add_documents(new_docs)

        logger.info(f"Added S2 paper {file_name} to Qdrant ({len(new_docs)} chunks)")
        return len(new_docs)

    def add_s2_paper_abstract_only(self, paper: S2Paper) -> int:
        """Fallback ingestion: embed title + abstract as a single document
        when no open-access PDF is available.

        Returns:
            Number of document chunks stored (always 1).
        """
        content_parts = []
        if paper.title:
            content_parts.append(paper.title)
        if paper.abstract:
            content_parts.append(paper.abstract)

        if not content_parts:
            raise ValueError(f"Paper {paper.paperId} has neither title nor abstract")

        content = "\n\n".join(content_parts)
        metadata = self._s2_paper_metadata(paper)

        doc = Document(page_content=content, metadata=metadata)
        self.retriever.add_documents([doc])
        logger.info(f"Added S2 paper {paper.paperId} abstract-only to Qdrant")
        return 1

    def search(self, query: str, k: int = 10, score_threshold: float = None) -> list[tuple[ArxivPaper, float]]:
        """
        Search for papers relevant to the query.
        
        Args:
            query: The search query string
            k: Number of results to return (default: 10)
            score_threshold: Optional minimum similarity score threshold
            
        Returns:
            List of tuples (ArxivPaper, score) with supporting_detail field filled with retrieved segments
        """
        # Perform similarity search with scores
        if score_threshold is not None:
            docs_and_scores = self.vector_store.similarity_search_with_score(
                query, 
                k=k,
                score_threshold=score_threshold
            )
        else:
            docs_and_scores = self.vector_store.similarity_search_with_score(query, k=k)
        
        # Convert documents to ArxivPaper objects
        results = []
        for doc, score in docs_and_scores:
            # Extract paper metadata
            metadata = doc.metadata.copy()
            
            # Add the retrieved segment as supporting_detail
            metadata['supporting_detail'] = doc.page_content
            
            # Create ArxivPaper object from metadata
            try:
                paper = ArxivPaper(**metadata)
                results.append((paper, score))
            except Exception as e:
                logger.warning(f"Failed to create ArxivPaper from metadata: {e}")
                continue
        
        logger.info(f"Found {len(results)} papers for query: {query[:50]}...")
        return results
    
    def search_selected_ids(self, ids: list[str], query: str, k: int = 10, score_threshold: float = None) -> List[Document]:
        filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.id",
                    match=MatchAny(any=ids)
                )
            ]
        )
        results = self.retriever.invoke(
            query,
            config={
                "configurable": {
                    "search_kwargs": {
                        "filter": filter,
                        "score_threshold": score_threshold,
                        "k": k,
                }
            }
            }
        )
        return results

    def check_paper_exists(self, paper_id: str) -> bool:
        result = self.client.count(
            collection_name=self.config.collection,
            count_filter=Filter(
                must=[
                    FieldCondition(
                        key="metadata.id",
                        match=MatchAny(any=[paper_id])
                    )
                ]
            ),
            exact=True,
        )
        logger.info(f"check_paper_exists({paper_id!r}): count={result.count}")
        return result.count > 0