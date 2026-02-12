from typing import Literal
from langchain_core.messages import AIMessage
from langgraph.graph.ui import push_ui_message 


def get_query_clarification_step(status: Literal["running", "unclear", "clear"]):
    QUERY_CLARIFICATION_RUNNING = {
    "id": "Query clarification",
    "label": "Query clarification",
    "status": "running",
    "description": f"Checking if query is clear...",
    }

    QUERY_CLARIFICATION_UNCLEAR = {
        "id": "Query clarification",
        "label": "Query clarification",
        "status": "completed",
        "description": f"Asking for clarification...",
    }

    QUERY_CLARIFICATION_CLEAR = {
        "id": "Query clarification",
        "label": "Query clarification",
        "status": "completed",
        "description": f"The user query is clear.",
    }

    if status == "running":
        return QUERY_CLARIFICATION_RUNNING
    elif status == "unclear":
        return QUERY_CLARIFICATION_UNCLEAR
    elif status == "clear":
        return QUERY_CLARIFICATION_CLEAR
    else:   
        raise ValueError(f"Invalid status: {status}")

def get_query_optimization_step(status: Literal["running", "completed"], optimized_query: str | None = None):
    QUERY_OPTIMIZATION_RUNNING = {
    "id": "Query optimization",
    "label": "Query optimization",
    "status": "running",
    "description": f"Optimizing query...",
    }

    QUERY_OPTIMIZATION_COMPLETED = {
        "id": "Query optimization",
        "label": "Query optimization",
        "status": "completed",
        "description": f"The optimized query is: {optimized_query}",
    }
    if status == "running":
        return QUERY_OPTIMIZATION_RUNNING
    elif status == "completed":
        return QUERY_OPTIMIZATION_COMPLETED
    else:
        raise ValueError(f"Invalid status: {status}")

def get_plan_step(status: Literal["running", "completed"], plan_choice: Literal["find_then_qa", "find_only", "qa_only"] | None = None):
    PLAN_RUNNING = {
    "id": "Plan",
    "label": "Plan",
    "status": "running",
    "description": f"Supervisor agent planning the next steps...",
    }

    if status == "running":
        return PLAN_RUNNING

    # status == "completed" - need plan_choice
    description = ""
    if plan_choice == "find_then_qa":
        description = "Based on the user's intent, first retrieve the papers then answer the user's question"
    elif plan_choice == "find_only":
        description = "Based on the user's intent, only retrieve the papers"
    elif plan_choice == "qa_only":
        description = "Based on the user's intent, only answer the user's question"
    else:
        raise ValueError(f"Invalid plan choice: {plan_choice}")

    PLAN_COMPLETED = {
        "id": "Plan",
        "label": "Plan",
        "status": "completed",
        "description": f"{description}",
    }
    return PLAN_COMPLETED

def get_find_papers_step(status: Literal["running", "completed"], num_papers: int | None = None):
    FIND_PAPERS_RUNNING = {
    "id": "Find papers",
    "label": "Find papers",
    "status": "running",
    "description": f"Delegating to find papers agent, this might take a while...",
    }

    if status == "running":
        return FIND_PAPERS_RUNNING

    # status == "completed" - need num_papers
    if num_papers is None:
        raise ValueError("num_papers is required when status is 'completed'")

    FIND_PAPERS_COMPLETED = {
        "id": "Find papers",
        "label": "Find papers",
        "status": "completed",
        "description": f"Found {num_papers} papers.",
    }
    return FIND_PAPERS_COMPLETED

def get_retrieve_and_answer_question_step(status: Literal["running", "completed"]):
    ANSWER_QUESTION_RUNNING = {
    "id": "Answer question",
    "label": "Answer question",
    "status": "running",
    "description": f"Retrieving evidence to answer the user's question, this might take a while...",
    }
    if status == "running":
        return ANSWER_QUESTION_RUNNING

    ANSWER_QUESTION_COMPLETED = {
        "id": "Answer question",
        "label": "Answer question",
        "status": "completed",
        "description": f"Finished retrieving evidence to answer the user's question.",
    }
    return ANSWER_QUESTION_COMPLETED

step_map = {
    "query_clarification": (0, get_query_clarification_step),
    "query_optimization": (1, get_query_optimization_step),
    "plan": (2, get_plan_step),
    "find_papers": (3, get_find_papers_step),
    "retrieve_and_answer_question": (4, get_retrieve_and_answer_question_step),
}

class UIManager:
    def __init__(self, step_tracking_message: AIMessage, step_tracking_ui_id: str):
        self.steps = [None] * len(step_map)
        self.step_tracking_message = step_tracking_message
        self.step_tracking_ui_id = step_tracking_ui_id


    def update_step(self, step_name: str, step_status: str, *args: str):
        step_idx, step_generator = step_map[step_name]
        self.steps[step_idx] = step_generator(step_status, *args)
        ui_steps = [step for step in self.steps if step is not None]
        push_ui_message(
            "steps",
            {
                "steps": ui_steps
            },
            message=self.step_tracking_message,
            id=self.step_tracking_ui_id
        )