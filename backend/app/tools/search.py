from langchain.tools import tool
from typing import List, Dict, Any, Optional
from app.db.models import Paper
from app.db.session import AsyncSessionLocal
from app.services.elasticsearch import ElasticsearchService
from app.core.config import settings


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
def hybrid_search_papers(
    query: str, 
    limit: int = 10, 
    categories: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search academic papers using hybrid search (combines text search and semantic similarity).
    
    This tool provides the most comprehensive search by combining:
    - Traditional keyword/text matching for exact term matches
    - Semantic vector similarity for conceptual matches
    - Category filtering for domain-specific searches
    
    Args:
        query: Search query describing what papers you're looking for
        limit: Maximum number of results to return (default: 10, max: 50)
        categories: Optional comma-separated list of ArXiv categories to filter by 
                   (e.g., "cs.CL,cs.AI" for computational linguistics and AI papers)
    
    Returns:
        List of relevant papers with metadata, scores, and highlighted snippets
    """
    try:
        # Initialize Elasticsearch service
        es_service = ElasticsearchService(settings.elasticsearch_config)
        
        # Parse categories filter
        categories_filter = None
        if categories:
            categories_filter = [cat.strip() for cat in categories.split(',')]
        
        # Perform hybrid search
        results = es_service.hybrid_search(
            query=query,
            limit=min(limit, 50),  # Cap at 50 results
            categories_filter=categories_filter
        )
        
        # Format results for agent use
        formatted_results = []
        for result in results:
            source = result['source']
            formatted_result = {
                'arxiv_id': result['id'],
                'title': source.get('title', ''),
                'authors': source.get('authors', ''),
                'abstract': source.get('abstract', '')[:500] + '...' if len(source.get('abstract', '')) > 500 else source.get('abstract', ''),
                'categories': source.get('categories', ''),
                'doi': source.get('doi'),
                'journal_ref': source.get('journal-ref'),
                'search_score': round(result['score'], 3),
                'highlights': result.get('highlights', {}),
                'url': f"https://arxiv.org/abs/{result['id']}"
            }
            formatted_results.append(formatted_result)
        
        return formatted_results
        
    except Exception as e:
        return [{"error": f"Search failed: {str(e)}"}]


@tool
def semantic_search_papers(
    query: str, 
    limit: int = 10, 
    categories: Optional[str] = None,
    search_field: str = "title"
) -> List[Dict[str, Any]]:
    """
    Search papers using semantic similarity (finds conceptually similar papers).
    
    This tool uses AI embeddings to find papers that are conceptually similar
    to your query, even if they don't contain the exact keywords.
    
    Args:
        query: Describe the concepts or ideas you're looking for
        limit: Maximum number of results (default: 10, max: 30)
        categories: Optional comma-separated ArXiv categories to filter by
        search_field: Field to search against - "title" or "abstract" (default: "title")
    
    Returns:
        List of conceptually similar papers ranked by semantic similarity
    """
    try:
        es_service = ElasticsearchService(settings.elasticsearch_config)
        
        # Parse categories filter
        categories_filter = None
        if categories:
            categories_filter = [cat.strip() for cat in categories.split(',')]
        
        # Map search field to vector field
        vector_field = "title_vector" if search_field == "title" else "abstract_vector"
        
        # Perform semantic search
        results = es_service.semantic_search(
            query=query,
            limit=min(limit, 30),
            field=vector_field,
            categories_filter=categories_filter
        )
        
        # Format results
        formatted_results = []
        for result in results:
            source = result['source']
            formatted_result = {
                'arxiv_id': result['id'],
                'title': source.get('title', ''),
                'authors': source.get('authors', ''),
                'abstract': source.get('abstract', '')[:400] + '...' if len(source.get('abstract', '')) > 400 else source.get('abstract', ''),
                'categories': source.get('categories', ''),
                'similarity_score': round(result['score'], 3),
                'url': f"https://arxiv.org/abs/{result['id']}"
            }
            formatted_results.append(formatted_result)
        
        return formatted_results
        
    except Exception as e:
        return [{"error": f"Semantic search failed: {str(e)}"}]


@tool 
def keyword_search_papers(
    query: str, 
    limit: int = 10, 
    categories: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search papers using traditional keyword/text matching.
    
    This tool performs exact text matching and is best for:
    - Finding papers with specific terms or phrases
    - Author name searches
    - Exact title or journal searches
    
    Args:
        query: Keywords or phrases to search for
        limit: Maximum number of results (default: 10, max: 30)
        categories: Optional comma-separated ArXiv categories to filter by
    
    Returns:
        List of papers matching the keywords with highlighted text snippets
    """
    try:
        es_service = ElasticsearchService(settings.elasticsearch_config)
        
        # Parse categories filter
        categories_filter = None
        if categories:
            categories_filter = [cat.strip() for cat in categories.split(',')]
        
        # Perform text search
        results = es_service.text_search(
            query=query,
            limit=min(limit, 30),
            categories_filter=categories_filter
        )
        
        # Format results
        formatted_results = []
        for result in results:
            source = result['source']
            formatted_result = {
                'arxiv_id': result['id'],
                'title': source.get('title', ''),
                'authors': source.get('authors', ''),
                'abstract': source.get('abstract', '')[:400] + '...' if len(source.get('abstract', '')) > 400 else source.get('abstract', ''),
                'categories': source.get('categories', ''),
                'text_score': round(result['score'], 3),
                'highlights': result.get('highlights', {}),
                'url': f"https://arxiv.org/abs/{result['id']}"
            }
            formatted_results.append(formatted_result)
        
        return formatted_results
        
    except Exception as e:
        return [{"error": f"Keyword search failed: {str(e)}"}]


@tool
def search_papers_by_category(
    categories: str, 
    limit: int = 20,
    recent_only: bool = False
) -> List[Dict[str, Any]]:
    """
    Browse papers by ArXiv category.
    
    Useful for exploring papers in specific research domains.
    
    Args:
        categories: Comma-separated ArXiv categories (e.g., "cs.CL,cs.AI,cs.LG")
        limit: Maximum number of results (default: 20, max: 50)
        recent_only: If True, prioritize more recent papers (default: False)
    
    Returns:
        List of papers from the specified categories
    """
    try:
        es_service = ElasticsearchService(settings.elasticsearch_config)
        
        # Parse categories
        categories_filter = [cat.strip() for cat in categories.split(',')]
        
        # Use a broad query to get papers from these categories
        results = es_service.text_search(
            query="*",  # Match all papers
            limit=min(limit, 50),
            categories_filter=categories_filter
        )
        
        # Format results
        formatted_results = []
        for result in results:
            source = result['source']
            formatted_result = {
                'arxiv_id': result['id'],
                'title': source.get('title', ''),
                'authors': source.get('authors', ''),
                'abstract': source.get('abstract', '')[:300] + '...' if len(source.get('abstract', '')) > 300 else source.get('abstract', ''),
                'categories': source.get('categories', ''),
                'url': f"https://arxiv.org/abs/{result['id']}"
            }
            formatted_results.append(formatted_result)
        
        return formatted_results
        
    except Exception as e:
        return [{"error": f"Category search failed: {str(e)}"}]

@tool
def get_paper_details(arxiv_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Fetch full details for a list of arXiv IDs to ground synthesis.
    """
    es = ElasticsearchService(settings.elasticsearch_config)
    out: List[Dict[str, Any]] = []
    for pid in arxiv_ids:
        doc = es.get_paper(pid)
        if doc:
            out.append({
                "arxiv_id": pid,
                "title": doc.get("title"),
                "authors": doc.get("authors"),
                "abstract": doc.get("abstract"),
                "categories": doc.get("categories"),
                "doi": doc.get("doi"),
                "journal_ref": doc.get("journal-ref"),
                "url": f"https://arxiv.org/abs/{pid}",
            })
    return out