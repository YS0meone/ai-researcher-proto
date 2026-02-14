from typing import TypedDict, Literal
from enum import Enum


class Step(TypedDict):
    id: str
    label: str
    status: str
    description: str


class StepName(Enum):
    QUERY_CLARIFICATION = "Query clarification"
    QUERY_OPTIMIZATION = "Query optimization"
    PLAN = "Plan"
    FIND_PAPERS = "Find papers"
    RETRIEVE_AND_ANSWER_QUESTION = "Answer question"


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
