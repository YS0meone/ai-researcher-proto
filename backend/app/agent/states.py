from typing import Annotated, List, Dict, Any, Optional, Literal, Tuple, Sequence
from langgraph.graph.ui import AnyUIMessage, ui_message_reducer
from langchain.messages import AIMessage
import operator
from langchain.agents import AgentState
from app.core.schema import S2Paper, Step
from langchain_core.documents import Document
from langgraph.graph.message import MessagesState

class SupervisorState(AgentState):
    optimized_query: Optional[str]
    is_clear: Optional[bool]
    papers: List[S2Paper]
    ui: Annotated[Sequence[AnyUIMessage], ui_message_reducer]
    plan_steps: List[Literal["answer_question", "find_papers", "end"]]
    selected_paper_ids: List[str]
    ui_tracking_message: AIMessage
    ui_tracking_id: str
    steps: List[Step]


class PaperFinderState(MessagesState):
    optimized_query: Optional[str]
    plan_steps: List[str]       
    completed_steps: Annotated[List[Tuple[str, str]], operator.add]
    plan_reasoning: str
    papers: List[Dict[str, Any]]
    iter: int
    goal_achieved: bool

class QAAgentState(MessagesState):
    evidences: Annotated[List[Document], operator.add]
    limitation: str
    qa_iteration: int
    selected_paper_ids: List[str]
    sufficient_evidence: bool
    user_query: str
    papers: List[S2Paper]
    final_answer: str