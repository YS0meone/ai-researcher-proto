import re
from app.core.config import QdrantConfig
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_huggingface import HuggingFaceEmbeddings
from app.db.schema import ArxivPaper
import arxiv
from pathlib import Path
import logging
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import GrobidParser

logger = logging.getLogger(__name__)
embeddings = HuggingFaceEmbeddings(model_name="allenai-specter") 

class QdrantService:

    def __init__(self, config: QdrantConfig):
        self.config = config
        self.client = QdrantClient(url=self.config.url)

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
        