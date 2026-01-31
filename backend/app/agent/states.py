from typing import Annotated, List, Dict, Any, Optional, Literal, TypedDict, Tuple
from langgraph.graph.message import add_messages
import operator


class State(TypedDict, total=False):
    """State for paper research agent.

    Shared state between paper finding and QA subgraphs.
    Using total=False allows partial updates from nodes.
    """
    # Core message history (required)
    messages: Annotated[list, add_messages]

    optimized_query: Optional[str]  # Query optimized for search
    paper_search_iteration: int     # Current paper search iteration (max 3)
    is_clear: Optional[bool]        # Whether the user's query is clear

    # Paper finding state
    plan_steps: List[str]
    plan_reasoning: str
    papers: List[Dict[str, Any]]  # Retrieved papers from search
    search_queries: List[str]     # Queries executed so far
    iter: int                     # Current search iteration
    max_iters: int                # Max search iterations allowed
    coverage_score: float         # How well papers cover the query

    # QA state
    selected_ids: List[str]
    retrieved_segments: List[str]
    limitation: List[str]
    qa_query: Optional[str]
    retrieval_queries: List[str]  # The queries used for retrieval
    sufficient_evidence: bool     # Whether the evidence is sufficient to answer the user's question
    qa_iteration: int             # Current QA iteration
    rd_reason: str

class PaperFinderState(TypedDict):
    messages: Annotated[list, add_messages]
    optimized_query: Optional[str]
    plan_steps: List[str]       
    completed_steps: Annotated[List[Tuple[str, str]], operator.add]
    plan_reasoning: str
    papers: List[Dict[str, Any]]
    iter: int
    goal_achieved: bool