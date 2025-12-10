from typing import Annotated, List, Dict, Any, Optional, Literal, TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict, total=False):
    """State for paper research agent.
    
    Shared state between paper finding and QA subgraphs.
    Using total=False allows partial updates from nodes.
    """
    # Core message history (required)
    messages: Annotated[list, add_messages]
    
    # Paper finding state
    papers: List[Dict[str, Any]]  # Retrieved papers from search
    search_queries: List[str]     # Queries executed so far
    iter: int                     # Current search iteration
    max_iters: int                # Max search iterations allowed
    coverage_score: float         # How well papers cover the query
    route: Optional[Literal["search", "synthesize", "qa"]]  # Routing decision
    
    # QA state
    selected_ids: List[str]       # Paper IDs selected for QA (by user or from retrieval)
    retrieved_segments: List[Dict[str, Any]]  # Vector search results from selected papers
    qa_query: Optional[str]       # The specific question for QA mode