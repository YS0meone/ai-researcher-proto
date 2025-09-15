import arxiv
from pprint import pprint
import sqlalchemy
from pathlib import Path
import multiprocessing
import argparse
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


def create_db_session():
    engine = create_engine(settings.DATABASE_URL)
    Session = scoped_session(sessionmaker(bind=engine))
    return Session()

def init_worker():
    global session
    session = create_db_session()


def worker(query, output_dir, offset, n):
    global session
    # Construct the default API client.
    client = arxiv.Client()
    # search for the most recent 100 papers in the cs.LG category
    search = arxiv.Search(query=query, max_results=n, sort_by=arxiv.SortCriterion.SubmittedDate)
    results = client.results(search, offset=offset)

    # create folder if not exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {offset} to {n - 1}")
    papers = []
    for result in results:
        # Print all properties of the result
        # Clean the title to create a valid filename
        file_name = re.sub(r'[<>:"/\\|?*]', '_', result.title.replace(" ", "_"))
        result.download_pdf(dirpath=output_dir, filename=file_name+".pdf")
        loader = GenericLoader.from_filesystem(
            output_dir,
            glob=file_name+".pdf",
            suffixes=[".pdf"],
            parser= GrobidParser(segment_sentences=False)
        )
        docs = loader.load()
        full_text = '\n'.join(doc.page_content for doc in docs)
        print(f"Fully parsed {file_name} with {len(full_text)} characters")
        new_paper = Paper(
            paper_id=result.entry_id,
            title=result.title,
            abstract=result.summary,
            authors=[author.name for author in result.authors],
            published_date=result.published,
            url=result.pdf_url,
            full_text=full_text,
        )
        papers.append(new_paper)
    session.add_all(papers)
    session.commit()
    print(f"Saved {len(papers)} papers")

def download_papers(n, query, output_dir, processes):
    params = [None] * processes
    offset = 0
    for i in range(processes):
        params[i] = (query, output_dir+f"/{i}", offset, offset + n//processes)
        offset += n//processes
    with multiprocessing.Pool(processes=processes, initializer=init_worker) as pool:
        pool.starmap(worker, params)
    

def main():
    parser = argparse.ArgumentParser(description="Download papers from arXiv")
    parser.add_argument("-n", type=int, default=50, help="Number of papers to download")
    parser.add_argument("-q", type=str, default="cat:cs.CL AND all:nlp OR all:natural language processing", help="Search query for arXiv")
    parser.add_argument("-o", type=str, default="./papers", help="Output directory")
    parser.add_argument("-p", type=int, default=2, help="Number of processes to use")
    args = parser.parse_args()

    download_papers(args.n, args.q, args.o, args.p)

if __name__ == "__main__":
    main()