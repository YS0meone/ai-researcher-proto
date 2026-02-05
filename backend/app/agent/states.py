from typing import Annotated, List, Dict, Any, Optional, Literal, TypedDict, Tuple, Sequence
from langgraph.graph.message import add_messages
from langgraph.graph.ui import AnyUIMessage, ui_message_reducer
import operator
from langchain.agents import AgentState
from app.db.schema import S2Paper

class State(AgentState):
    """State for paper research agent.

    Shared state between paper finding and QA subgraphs.
    Using total=False allows partial updates from nodes.
    """

    optimized_query: Optional[str]  # Query optimized for search
    is_clear: Optional[bool]
    papers: List[S2Paper]
    ui: Annotated[Sequence[AnyUIMessage], ui_message_reducer]  # UI messages channel 

    # plan_steps: List[str]
    # plan_reasoning: str
    
    # search_queries: List[str]     # Queries executed so far
    # iter: int                     # Current search iteration
    # max_iters: int                # Max search iterations allowed
    # coverage_score: float         # How well papers cover the query

    # QA state
    # selected_ids: List[str]
    # retrieved_segments: List[str]
    # limitation: List[str]
    # qa_query: Optional[str]
    # retrieval_queries: List[str]  # The queries used for retrieval
    # sufficient_evidence: bool     # Whether the evidence is sufficient to answer the user's question
    # qa_iteration: int             # Current QA iteration
    # rd_reason: str

class PaperFinderState(TypedDict):
    messages: Annotated[list, add_messages]
    optimized_query: Optional[str]
    plan_steps: List[str]       
    completed_steps: Annotated[List[Tuple[str, str]], operator.add]
    plan_reasoning: str
    papers: List[Dict[str, Any]]
    iter: int
    goal_achieved: bool