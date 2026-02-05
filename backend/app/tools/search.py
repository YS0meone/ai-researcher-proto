from langchain.tools import tool, ToolRuntime
from langgraph.prebuilt import InjectedState
from typing import List, Dict, Any, Optional, Annotated
from app.db.models import Paper
from app.db.session import AsyncSessionLocal
from app.services.elasticsearch import ElasticsearchService
from app.services.qdrant import QdrantService
from app.services.s2_client import S2Client
from app.core.config import settings
from pydantic import BaseModel, Field
from langgraph.types import Command
import os
from rerankers import Reranker, Document
from app.db.schema import S2Paper
from langchain_tavily import TavilySearch
from app.agent.utils import get_paper_info_text
from langchain_core.messages import ToolMessage
from pydantic import ConfigDict

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


def hybrid_search_papers_impl(
    query: str,
    limit: int = 10,
    categories: Optional[str] = None
) -> List[Dict[str, Any]]:
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
    return hybrid_search_papers_impl(query, limit, categories)


def semantic_search_papers_impl(
    query: str,
    limit: int = 10,
    categories: Optional[str] = None,
    search_field: str = "title"
) -> List[Dict[str, Any]]:
    """
    Search papers using semantic similarity (finds conceptually similar papers).
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
    
    return semantic_search_papers_impl(query, limit, categories, search_field)

def keyword_search_papers_impl(
    query: str,
    limit: int = 10,
    categories: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search papers using traditional keyword/text matching.
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
    
    return keyword_search_papers_impl(query, limit, categories)


def search_papers_by_category_impl(
    categories: str,
    limit: int = 20,
    recent_only: bool = False
) -> List[Dict[str, Any]]:
    """
    Browse papers by ArXiv category.
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
    
    return search_papers_by_category_impl(categories, limit, recent_only)

def get_paper_details_impl(arxiv_ids: List[str]) -> List[Dict[str, Any]]:
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

@tool
def get_paper_details(arxiv_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Fetch full details for a list of arXiv IDs to ground synthesis.
    """
    return get_paper_details_impl(arxiv_ids)


def vector_search_papers_impl(
    query: str,
    limit: int = 10,
    score_threshold: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Search papers using deep semantic vector search with full-text content retrieval.
    """
    try:
        # Initialize Qdrant service
        qdrant_service = QdrantService(settings.qdrant_config)
        
        # Perform vector search
        results = qdrant_service.search(
            query=query,
            k=min(limit, 30),
            score_threshold=score_threshold
        )
        
        # Format results
        formatted_results = []
        for paper, score in results:
            abstract = paper.abstract
            supporting_detail = paper.supporting_detail or ""
            
            formatted_result = {
                'arxiv_id': paper.id,
                'title': paper.title,
                'authors': paper.authors,
                'abstract': abstract,
                'categories': paper.categories,
                'supporting_detail': supporting_detail,
                'similarity_score': round(float(score), 3),
                'url': f"https://arxiv.org/abs/{paper.id}"
            }
            formatted_results.append(formatted_result)
        
        return formatted_results
        
    except Exception as e:
        return [{"error": f"Vector search failed: {str(e)}"}]

@tool
def vector_search_papers(
    query: str,
    limit: int = 10,
    score_threshold: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Search papers using deep semantic vector search with full-text content retrieval.
    
    This tool uses specialized scientific paper embeddings (SPECTER) to find papers
    by semantic similarity against full paper content (not just title/abstract).
    Returns relevant text segments from papers that match the query.
    
    Args:
        query: Describe the specific concepts, methods, or findings you're looking for
        limit: Maximum number of results to return (default: 10, max: 30)
        score_threshold: Optional minimum similarity threshold (0.0-1.0) to filter results
    
    Returns:
        List of papers with relevant text segments and similarity scores
    """
    return vector_search_papers_impl(query, limit, score_threshold)


def vector_search_papers_by_ids_impl(
    query: str,
    limit: int = 10,
    score_threshold: Optional[float] = None,
    ids: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Search papers using deep semantic vector search with full-text content retrieval.
    """
    try:
        # Initialize Qdrant service
        qdrant_service = QdrantService(settings.qdrant_config)
        
        # Perform vector search
        results = qdrant_service.search_selected_ids(
            ids=ids,
            query=query,
            k=min(limit, 30),
            score_threshold=score_threshold,
        )
        
        # Format results
        formatted_results = []
        for paper, score in results:
            abstract = paper.abstract
            supporting_detail = paper.supporting_detail or ""
            
            formatted_result = {
                'arxiv_id': paper.id,
                'title': paper.title,
                'authors': paper.authors,
                'abstract': abstract,
                'categories': paper.categories,
                'supporting_detail': supporting_detail,
                'similarity_score': round(float(score), 3),
                'url': f"https://arxiv.org/abs/{paper.id}"
            }
            formatted_results.append(formatted_result)
        
        return formatted_results
        
    except Exception as e:
        return [{"error": f"Vector search failed: {str(e)}"}]

@tool
def vector_search_papers_by_ids(
    query: str,
    limit: int = 10,
    score_threshold: Optional[float] = None,
    ids: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Search papers using deep semantic vector search with full-text content retrieval by arXiv IDs.
    
    Args:
        query: The search query string
        limit: Maximum number of results to return (default: 10, max: 30)
        score_threshold: Optional minimum similarity threshold (0.0-1.0) to filter results
        ids: List of arXiv IDs to search for
    
    Returns:
        List of papers with relevant text segments and similarity scores
    """
    return vector_search_papers_by_ids_impl(query, limit, score_threshold, ids)

def get_paper_abstract(arxiv_ids: List[str]) -> Dict[str, str]:
    """
    Retrieve paper abstracts by arXiv IDs.
    
    Args:
        arxiv_ids: List of arXiv IDs to fetch abstracts for
    
    Returns:
        Dictionary mapping paper_id to abstract text
    """
    es_service = ElasticsearchService(settings.elasticsearch_config)
    abstracts = {}
    
    for arxiv_id in arxiv_ids:
        paper = es_service.get_paper(arxiv_id)
        if paper and paper.get('abstract'):
            abstracts[arxiv_id] = paper['abstract']
    
    return abstracts


class S2SearchPapersRequest(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    runtime: ToolRuntime
    reasoning: str = Field(..., description="Explain your search strategy: what are you looking for and why this query formulation?")
    query: str = Field(
        ...,
        description="""Search query for Semantic Scholar. Follow these guidelines for effective queries:

QUERY LENGTH: Keep queries SHORT (2-5 keywords). The API matches against title and abstract.
- BAD: "What are the methods for improving efficiency of large language models through quantization and pruning techniques?"
- GOOD: "language model quantization pruning"

PHRASE MATCHING: Use quotes for exact multi-word phrases.
- "graph neural network" (matches exact phrase)
- "large language model" efficiency (phrase + keyword)

BOOLEAN OPERATORS: Use + (AND) and | (OR) to combine terms.
- transformer + attention (both required)
- BERT | RoBERTa | GPT (any of these)
- "neural network" + (pruning | quantization) (phrase AND either term)

KEYWORD STRATEGIES:
- Use technical terms, not natural language questions
- Include acronyms AND full names: "LLM" | "large language model"
- Prefer nouns over verbs: "image classification" not "classifying images"
- Omit common words: the, a, an, of, for, in, on

EXAMPLES:
- Finding a specific paper: Use match_title=True with the paper title
- Methodology papers: "transformer attention mechanism"
- Survey papers: "survey" + "deep learning"
- Recent techniques: Use year filter instead of adding "recent" to query"""
    )
    year: str = Field(
        None,
        description="Filter by publication year range. Examples: '2023' (single year), '2020-2024' (range). Use this instead of adding year terms to query."
    )
    venue: List[str] = Field(
        None,
        description="Filter by venue names (conferences/journals). Examples: ['NeurIPS', 'ICML', 'ACL']. Case-insensitive."
    )
    fields_of_study: List[str] = Field(
        None,
        description="Filter by field. Options: 'Computer Science', 'Mathematics', 'Physics', 'Biology', 'Medicine', etc. Use for broad domain filtering."
    )
    publication_date_or_year: str = Field(
        None,
        description="Filter by date range in format 'YYYY-MM-DD:YYYY-MM-DD' or 'YYYY:YYYY'. Example: '2023-01-01:2024-12-31'"
    )
    min_citation_count: int = Field(
        None,
        description="Minimum citations required. Use to find influential papers (e.g., 100 for well-cited, 1000 for seminal works)."
    )
    match_title: bool = Field(
        False,
        description="Set True to find ONE paper by exact title match. Use when you know the specific paper title."
    )

@tool(
    args_schema=S2SearchPapersRequest
    )
def s2_search_papers(
    runtime: ToolRuntime,
    reasoning: str,
    query: str,
    year: str = None,
    venue: List[str] = None,
    fields_of_study: List[str] = None,
    publication_date_or_year: str = None,
    min_citation_count: int = None,
    match_title: bool = False,
):
    """
    Search academic papers using Semantic Scholar's database of 200M+ papers.

    WHEN TO USE:
    - Finding papers on a specific research topic
    - Locating papers by methodology or technique
    - Finding influential/highly-cited papers in a field
    - Looking up a specific paper by title (use match_title=True)

    QUERY BEST PRACTICES:
    - Keep queries SHORT: 2-5 keywords, not full sentences
    - Use quotes for phrases: "graph neural network"
    - Use + for AND: transformer + efficiency
    - Use | for OR: BERT | GPT | T5
    - Combine with parentheses: "attention" + (vision | image)
    - Use FILTERS (year, venue, min_citation_count) instead of adding filter words to query

    EXAMPLES OF GOOD QUERIES:
    - "knowledge distillation" + transformer
    - "reinforcement learning" + robotics
    - BERT | RoBERTa (to find papers on either)
    - "graph neural network" + (node | link) + prediction

    EXAMPLES OF BAD QUERIES (too long/natural language):
    - "What are the recent advances in transformer architectures for NLP tasks?"
    - "Papers about how to make neural networks more efficient using pruning"

    Returns: List of papers with title, abstract, authors, year, and citation count.
    """
    # Get state from runtime (if available)
    state = runtime.state if runtime else {}
    tool_call_id = runtime.tool_call_id
    # Get new papers from S2
    try:
        s2_client = S2Client()
        new_results = s2_client.search_papers(
            query=query,
            year=year,
            venue=venue,
            fields_of_study=fields_of_study,
            publication_date_or_year=publication_date_or_year,
            min_citation_count=min_citation_count,
            match_title=match_title
        )
    except Exception as e:
        error_msg = str(e)
        if "Title match not found" in error_msg or "ObjectNotFoundException" in str(type(e)):
            # Handle title match failure gracefully
            return Command(
                update={"messages": [ToolMessage(content=f"No exact title match found for query: '{query}'. Try searching without title matching or rephrase the query.", tool_call_id=tool_call_id)]}
            )
        else:
            return Command(
                update={"messages": [ToolMessage(content=f"Error searching for papers: {error_msg}", tool_call_id=tool_call_id)]}
            )
    
    return Command(
        update={"new_papers": new_results, "messages": [
            ToolMessage(
                content=f"I found {len(new_results)} new papers for your query.",
                tool_call_id=tool_call_id
                )]}
    )

class TavilySearchRequest(BaseModel):
    reasoning: str = Field(..., description="Explain why you need to understand this research topic better and what you hope to learn")
    query: str = Field(..., description="A natural language query about the research topic or field you want to understand")


@tool(args_schema=TavilySearchRequest)
def tavily_research_overview(reasoning: str, query: str) -> str:
    """
    Search for general information about research topics using Tavily web search.
    
    **When to use this tool:**
    - When the research topic is GENERAL or UNFAMILIAR to you
    - When you need to understand what a research field or topic is about
    - When you want to identify the most FAMOUS/SEMINAL papers in a field
    - BEFORE searching academic databases, to avoid missing important foundational work
    - When you need context about trending research areas or breakthrough papers
    
    **This tool helps you:**
    - Understand the landscape of a research topic
    - Find names of influential papers, authors, and concepts
    - Identify key terms and methodologies to use in academic searches
    - Discover landmark papers that should not be missed
    
    **Example use cases:**
    - "What are the most important papers in transformer architectures?"
    - "What is the current state of research in quantum machine learning?"
    - "Who are the key researchers and what are the foundational papers in reinforcement learning?"
    
    Args:
        reasoning: Your reasoning for why you need this overview
        query: Natural language question about the research topic
        
    Returns:
        Web search results with information about the research topic, including
        mentions of famous papers, key researchers, and important concepts.
    """
    try:
        tavily = TavilySearch(
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=False,
            include_images=False
        )
        
        results = tavily.invoke({"query": query})
        
        # Format results for better readability
        if isinstance(results, list):
            formatted_output = []
            formatted_output.append(f"Research Topic Overview for: {query}\n")
            formatted_output.append("=" * 80 + "\n")
            
            for i, result in enumerate(results, 1):
                if isinstance(result, dict):
                    title = result.get('title', 'No title')
                    content = result.get('content', 'No content')
                    url = result.get('url', '')
                    
                    formatted_output.append(f"\n{i}. {title}")
                    formatted_output.append(f"   URL: {url}")
                    formatted_output.append(f"   Content: {content}\n")
            
            return "\n".join(formatted_output)
        else:
            return str(results)
            
    except Exception as e:
        return f"Error searching with Tavily: {str(e)}. You may proceed with direct academic database search."
    
@tool
def get_paper_details(
    runtime: ToolRuntime
) -> str:
    """
    Get the metadata of the papers in the paper list. Call it if you need to know the details of the papers in the paper list.

    **When to use this tool:**
    - When you want to check if certain paper in the paper list
    - When you want to know what papers you already have to decide what to do next
    
    Returns:
    The metadata of all papers in the paper list.
    """

    state = runtime.state if runtime else {}
    papers = state.get("papers", [])
    if len(papers) == 0:
        return "No paper in the paper list."

    paper_info_text = get_paper_info_text(papers)
    return paper_info_text


class ForwardSnowballRequest(BaseModel):
    """Request schema for forward snowball search."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    runtime: ToolRuntime
    reasoning: str = Field(
        ..., 
        description="Explain why you want to find papers that these seed papers cite/reference"
    )
    seed_paper_ids: List[str] = Field(
        ...,
        description="List of Semantic Scholar paper IDs to use as seed papers. These are the papers whose references you want to explore."
    )
    top_k: int = Field(
        10,
        description="Number of top referenced papers to return (default: 10, max: 50)"
    )


@tool(args_schema=ForwardSnowballRequest)
def forward_snowball(
    runtime: ToolRuntime,
    reasoning: str,
    seed_paper_ids: List[str],
    top_k: int = 10
):
    """
    Forward Snowball: Find papers that the specified seed papers CITE (their references).
    
    **What this does:**
    - Takes a list of paper IDs as seed papers
    - Fetches all papers that these seed papers reference/cite
    - Scores candidates based on citation count and how many seeds cite them
    - Returns the top-k most relevant referenced papers
    
    **When to use:**
    - You want to explore the foundational papers that influenced specific papers
    - You're tracing back to the original/parent works
    - You want to understand what prior work specific papers build upon
    
    **Example:**
    For papers about "GPT-3" and "BERT", this will find papers like 
    "Attention is All You Need" (Transformer paper) that they cite.
    
    Args:
        reasoning: Why you want to find referenced papers
        seed_paper_ids: List of Semantic Scholar paper IDs (e.g., ["paperId1", "paperId2"])
        top_k: Number of top papers to return (default: 10, max: 50)
        
    Returns:
        Top-k most relevant papers cited by the seed papers
    """
    tool_call_id = runtime.tool_call_id
    
    if not seed_paper_ids:
        return Command(
            update={"messages": [ToolMessage(
                content="No seed paper IDs provided. Please specify which papers to use for citation chasing.",
                tool_call_id=tool_call_id
            )]}
        )
    
    top_k = min(max(1, top_k), 50)  # Clamp between 1 and 50
    
    try:
        s2_client = S2Client()
        
        # Fetch references for all seed papers
        all_references = {}  # corpus_id -> {paper_data, num_seeds_citing: int}
        
        for paper_id in seed_paper_ids:
            try:
                references = s2_client.get_paper_references(
                    paper_id=paper_id,
                    fields=["paperId", "corpusId", "title", "abstract", "authors", 
                           "year", "citationCount", "influentialCitationCount"]
                )
                
                if not references:
                    continue
                
                for ref in references:
                    corpus_id = ref.get("corpusId")
                    if not corpus_id:
                        continue
                    
                    if corpus_id not in all_references:
                        all_references[corpus_id] = {
                            "paper": ref,
                            "num_seeds_citing": 0
                        }
                    
                    all_references[corpus_id]["num_seeds_citing"] += 1
                        
            except Exception as e:
                continue
        
        if not all_references:
            return Command(
                update={"messages": [ToolMessage(
                    content=f"Could not find references for the {len(seed_paper_ids)} seed papers.",
                    tool_call_id=tool_call_id
                )]}
            )
        
        # Score and sort candidates
        scored_candidates = []
        for corpus_id, data in all_references.items():
            paper = data["paper"]
            num_seeds = data["num_seeds_citing"]
            citation_count = paper.get("citationCount", 0)
            
            # Simple scoring: prioritize papers cited by multiple seeds and highly cited
            score = num_seeds * 10.0 + min(citation_count / 100, 10.0)
            
            scored_candidates.append({"score": score, "paper": paper})
        
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        top_candidates = scored_candidates[:top_k]
        
        # Convert to S2Paper objects
        result_papers = []
        for candidate in top_candidates:
            paper_data = candidate["paper"]
            try:
                s2_paper = S2Paper(
                    paperId=paper_data.get("paperId", ""),
                    corpusId=paper_data.get("corpusId"),
                    title=paper_data.get("title"),
                    abstract=paper_data.get("abstract"),
                    authors=paper_data.get("authors"),
                    year=paper_data.get("year"),
                    citationCount=paper_data.get("citationCount"),
                    influentialCitationCount=paper_data.get("influentialCitationCount"),
                    url=paper_data.get("url", f"https://www.semanticscholar.org/paper/{paper_data.get('paperId', '')}")
                )
                result_papers.append(s2_paper)
            except Exception:
                continue
        
        return Command(
            update={
                "new_papers": result_papers,
                "messages": [ToolMessage(
                    content=f"Found {len(result_papers)} papers cited by the {len(seed_paper_ids)} seed papers.",
                    tool_call_id=tool_call_id
                )]
            }
        )
        
    except Exception as e:
        return Command(
            update={"messages": [ToolMessage(
                content=f"Error during forward snowball: {str(e)}",
                tool_call_id=tool_call_id
            )]}
        )


class BackwardSnowballRequest(BaseModel):
    """Request schema for backward snowball search."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    runtime: ToolRuntime
    reasoning: str = Field(
        ..., 
        description="Explain why you want to find papers that cite these seed papers"
    )
    seed_paper_ids: List[str] = Field(
        ...,
        description="List of Semantic Scholar paper IDs to use as seed papers. These are the papers whose citations you want to explore."
    )
    top_k: int = Field(
        10,
        description="Number of top citing papers to return (default: 10, max: 50)"
    )


@tool(args_schema=BackwardSnowballRequest)
def backward_snowball(
    runtime: ToolRuntime,
    reasoning: str,
    seed_paper_ids: List[str],
    top_k: int = 10
):
    """
    Backward Snowball: Find papers that CITE the specified seed papers.
    
    **What this does:**
    - Takes a list of paper IDs as seed papers
    - Fetches all papers that cite these seed papers
    - Scores candidates based on citation count, recency, and how many seeds they cite
    - Returns the top-k most relevant citing papers
    
    **When to use:**
    - You want to find recent work building on specific foundational papers
    - You're looking for "child" papers that extend or apply seed papers
    - You want to discover how specific papers have been used or cited
    
    **Example:**
    For the "Attention is All You Need" paper, this will find papers like 
    "BERT", "GPT-3", and other transformer-based models that cite it.
    
    Args:
        reasoning: Why you want to find citing papers
        seed_paper_ids: List of Semantic Scholar paper IDs (e.g., ["paperId1", "paperId2"])
        top_k: Number of top papers to return (default: 10, max: 50)
        
    Returns:
        Top-k most relevant papers that cite the seed papers
    """
    tool_call_id = runtime.tool_call_id
    
    if not seed_paper_ids:
        return Command(
            update={"messages": [ToolMessage(
                content="No seed paper IDs provided. Please specify which papers to use for citation chasing.",
                tool_call_id=tool_call_id
            )]}
        )
    
    top_k = min(max(1, top_k), 50)  # Clamp between 1 and 50
    
    try:
        s2_client = S2Client()
        
        # Fetch citations for all seed papers
        all_citations = {}  # corpus_id -> {paper_data, num_seeds_cited: int}
        
        for paper_id in seed_paper_ids:
            try:
                citations = s2_client.get_paper_citations(
                    paper_id=paper_id,
                    fields=["paperId", "corpusId", "title", "abstract", "authors", 
                           "year", "citationCount", "influentialCitationCount"]
                )
                
                if not citations:
                    continue
                
                for cite in citations:
                    corpus_id = cite.get("corpusId")
                    if not corpus_id:
                        continue
                    
                    if corpus_id not in all_citations:
                        all_citations[corpus_id] = {
                            "paper": cite,
                            "num_seeds_cited": 0
                        }
                    
                    all_citations[corpus_id]["num_seeds_cited"] += 1
                        
            except Exception as e:
                continue
        
        if not all_citations:
            return Command(
                update={"messages": [ToolMessage(
                    content=f"Could not find citations for the {len(seed_paper_ids)} seed papers.",
                    tool_call_id=tool_call_id
                )]}
            )
        
        # Score and sort candidates
        scored_candidates = []
        current_year = 2026
        
        for corpus_id, data in all_citations.items():
            paper = data["paper"]
            num_seeds = data["num_seeds_cited"]
            citation_count = paper.get("citationCount", 0)
            year = paper.get("year", 2000)
            
            # Recency bonus for papers from last 3 years
            recency_bonus = max(0, (year - (current_year - 3)) / 3.0) * 2.0
            
            # Simple scoring: prioritize papers citing multiple seeds, highly cited, and recent
            score = num_seeds * 10.0 + min(citation_count / 100, 10.0) + recency_bonus
            
            scored_candidates.append({"score": score, "paper": paper})
        
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        top_candidates = scored_candidates[:top_k]
        
        # Convert to S2Paper objects
        result_papers = []
        for candidate in top_candidates:
            paper_data = candidate["paper"]
            try:
                s2_paper = S2Paper(
                    paperId=paper_data.get("paperId", ""),
                    corpusId=paper_data.get("corpusId"),
                    title=paper_data.get("title"),
                    abstract=paper_data.get("abstract"),
                    authors=paper_data.get("authors"),
                    year=paper_data.get("year"),
                    citationCount=paper_data.get("citationCount"),
                    influentialCitationCount=paper_data.get("influentialCitationCount"),
                    url=paper_data.get("url", f"https://www.semanticscholar.org/paper/{paper_data.get('paperId', '')}")
                )
                result_papers.append(s2_paper)
            except Exception:
                continue
        
        return Command(
            update={
                "new_papers": result_papers,
                "messages": [ToolMessage(
                    content=f"Found {len(result_papers)} papers that cite the {len(seed_paper_ids)} seed papers.",
                    tool_call_id=tool_call_id
                )]
            }
        )
        
    except Exception as e:
        return Command(
            update={"messages": [ToolMessage(
                content=f"Error during backward snowball: {str(e)}",
                tool_call_id=tool_call_id
            )]}
        )

