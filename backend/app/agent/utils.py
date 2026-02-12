from app.core.config import settings
import os
import time
from typing import List, Dict
from app.db.schema import S2Paper

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        return result
    return wrapper

def get_user_query(messages: list) -> str:
    """Extract the original user query from the message history."""
    user_msg = ""
    for m in messages:
        if hasattr(m, "type") and m.type == "human":
            user_msg = m.content
        elif isinstance(m, dict) and m.get("role") == "user" and m.get("content"):
            user_msg = m["content"]
    return user_msg 
    
def get_paper_info_text(papers: list[S2Paper]) -> str:
    """Get the text of the papers."""
    if not papers:
        return "No papers found yet"
    return "\n".join([f"Paper {paper.paperId}: {paper.title}\nAuthors: {paper.authors}\nPublication Date: {paper.publicationDate}\nAbstract: {paper.abstract}\n" for paper in papers])

def get_paper_abstract(papers: List[S2Paper], selected_paper_ids: List[str]) -> Dict[str, str]:
    abstracts = {}
    for paper in papers:
        if paper.paperId in selected_paper_ids:
            abstracts[paper.paperId] = paper.abstract
    return abstracts

def setup_langsmith():
    os.environ["LANGCHAIN_TRACING_V2"] = str(settings.LANGSMITH_TRACING_V2)
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
    