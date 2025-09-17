from langchain.tools import tool
from typing import List, Dict, Any
from app.db.models import Paper
from app.db.session import AsyncSessionLocal
from app.services.embedding import embedding_service


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

@tool
async def semantic_search_papers(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Semantic search academic papers using vector similarity.
    
    Args:
        query: Search terms to find in papers (title, abstract, full text)
        limit: Maximum number of results to return (default: 10)
    
    Returns:
        List of papers with title, abstract, authors, and similarity score
    """
    session = AsyncSessionLocal()
    
    try:
        # Generate embedding for the query
        query_embedding = await embedding_service.encode_async(query)
        
        # Perform semantic search
        results = await Paper.semantic_search(session, query_embedding, limit)
        
        return [
            {
                "title": paper.title,
                "abstract": paper.abstract,
                "authors": paper.authors,
                "published_date": paper.published_date.isoformat(),
                "url": paper.url,
                "similarity_score": 1.0 - float(distance)  # Convert distance to similarity
            }
            for paper, distance in results
        ]
    finally:
        await session.close()


@tool
async def hybrid_search_papers(query: str, limit: int = 10, semantic_weight: float = 0.7) -> List[Dict[str, Any]]:
    """Hybrid search combining full-text and semantic search for best results.
    
    Args:
        query: Search terms to find in papers
        limit: Maximum number of results to return (default: 10)
        semantic_weight: Weight for semantic vs text search (0.0-1.0, default: 0.7)
    
    Returns:
        List of papers with combined relevance scores
    """
    session = AsyncSessionLocal()
    
    try:
        # Generate embedding for the query
        query_embedding = await embedding_service.encode_async(query)
        
        # Perform hybrid search
        results = await Paper.hybrid_search(session, query, query_embedding, limit, semantic_weight)
        
        return [
            {
                "title": paper.title,
                "abstract": paper.abstract,
                "authors": paper.authors,
                "published_date": paper.published_date.isoformat(),
                "url": paper.url,
                "combined_score": float(relevance),
                "text_relevance": float(text_relevance),
                "semantic_similarity": float(semantic_similarity)
            }
            for paper, relevance, text_relevance, semantic_similarity in results
        ]
    finally:
        await session.close()