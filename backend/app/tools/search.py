from langchain.tools import tool
from typing import List, Dict, Any
from app.db.models import Paper
from app.db.session import AsyncSessionLocal


@tool
async def search_papers(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search academic papers using full-text search.
    
    Args:
        query: Search terms to find in papers (title, abstract, full text)
        limit: Maximum number of results to return (default: 10)
    
    Returns:
        List of papers with title, abstract, authors, and relevance score
    """
    # Create session directly using the sessionmaker
    session = AsyncSessionLocal()
    
    try:
        results = await Paper.search(session, query, limit)
        
        return [
            {
                "title": paper.title,
                "abstract": paper.abstract,
                "authors": paper.authors,
                "published_date": paper.published_date.isoformat(),
                "url": paper.url,
                "relevance_score": float(relevance)
            }
            for paper, relevance in results
        ]
    finally:
        await session.close()
