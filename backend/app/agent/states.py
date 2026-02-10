from typing import Annotated, List, Dict, Any, Optional, Literal, TypedDict, Tuple, Sequence
from langgraph.graph.message import add_messages
from langgraph.graph.ui import AnyUIMessage, ui_message_reducer
import operator
from langchain.agents import AgentState
from app.db.schema import S2Paper
from langgraph.types import Document
from langgraph.graph.message import MessageState

class State(AgentState):
    optimized_query: Optional[str]
    is_clear: Optional[bool]
    papers: List[S2Paper]
    ui: Annotated[Sequence[AnyUIMessage], ui_message_reducer]
    plan_steps: List[Literal["answer_question", "find_papers", "end"]]
    selected_paper_ids: List[str]


class PaperFinderState(MessageState):
    optimized_query: Optional[str]
    plan_steps: List[str]       
    completed_steps: Annotated[List[Tuple[str, str]], operator.add]
    plan_reasoning: str
    papers: List[Dict[str, Any]]
    iter: int
    goal_achieved: bool

class QAAgentState(MessageState):
    evidences: List[Document]
    limitation: str
    qa_iteration: int
    selected_paper_ids: List[str]
    sufficient_evidence: bool
    user_query: str
    papers: List[S2Paper]
