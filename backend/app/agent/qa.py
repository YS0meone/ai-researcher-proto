from typing import Dict, List, Any, Optional
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from app.agent.states import State
from app.agent.utils import get_user_query, setup_langsmith
from app.core.config import settings         
from app.tools.search import retrieve_evidence_from_selected_papers, get_paper_abstract
import logging
import sys
import json
from typing import Union
from app.agent.states import QAAgentState
from app.db.schema import S2Paper
from langgraph.graph import StateGraph, END, START

logger = logging.getLogger(__name__)
setup_langsmith()

qa_model = init_chat_model(model=settings.QA_AGENT_MODEL_NAME, api_key=settings.OPENAI_API_KEY)


def get_paper_abstract(papers: List[S2Paper], selected_paper_ids: List[str]) -> Dict[str, str]:
    abstracts = {}
    for paper in papers:
        if paper.paperId in selected_paper_ids:
            abstracts[paper.paperId] = paper.abstract
    return abstracts

def qa_retrieve(state: QAAgentState) -> QAAgentState:
    user_query = state.get("user_query", "")
    selected_paper_ids = state.get("selected_paper_ids", [])
    papers = state.get("papers", [])
    if not selected_paper_ids:
        print("WARNING: No papers selected for QA!", file=sys.stderr)
        return {
            "evidences": [],
            "messages": [AIMessage(content="No papers have been selected for Q&A. Please select papers first or use the paper finding mode.")]
        }


    QA_RETRIEVAL_SYSTEM = """You are an expert in evidence retrieval for academic paper QA.
    Goal:
    - You need to generate optimal tool calls to retrieve more evidence to answer the user's question.

    General guide:
    - The generated tool calls should be based on the following context: the chat history, the paper abstracts, the retrieved evidence and the limitation of the retrieved evidence.
    - The user question is usually a research question about several selected papers and the paper abstracts are the ones of the selected papers.
    - Use the chat history to understand what strategies you have tried and why you have tried them.
    - Use the paper abstract and the retrieved evidence to better understand the context and the user's question.
    - Use the limitation to guide you to generate the optimal tool calls.
    - Generate no more than 3 tool calls focus on the tool call quality.
    - You can use the search tool to help you understand the user's question better instead of directly answering the user's question.
    """

    abstracts = get_paper_abstract(papers=papers, selected_paper_ids=selected_paper_ids)
    evidences = state.get("evidences", [])
    evidences_text = "\n\n".join([
        f"Evidence {i+1}:\n{evidence.page_content}"
        for i, evidence in enumerate(evidences)
    ]) if evidences else "No evidence retrieved yet."
    limitation = state.get("limitation", "This is the first retrieval attempt, use search tools to retrieve more evidence.")

    abstracts_text = "\n".join([
        f"Paper {paper_id}:\n{abstract}"
        for paper_id, abstract in abstracts.items()
    ])
    
    retrieval_prompt = f"""User query: {user_query}

Paper abstracts:
{abstracts_text}

Retrieved evidences:
{evidences_text}

Limitation of the retrieved evidence:
{limitation}
"""

    tool_model = qa_model.bind_tools([retrieve_evidence_from_selected_papers], tool_choice="retrieve_evidence_from_selected_papers")
    tool_response = tool_model.invoke([
        SystemMessage(content=QA_RETRIEVAL_SYSTEM),
        *state.get("messages", []),
        HumanMessage(content=retrieval_prompt)
    ])
    return {
        "messages": [tool_response],
    }

def qa_evaluate(state: QAAgentState) -> QAAgentState:
    evaluation_system = """
    You are an expert in evaluating the relevance of retrieved evidence for answering a research question.
    You are given a chat history, user query, paper abstracts, retrieved evidence.
    The user query is usually a research question about several selected papers and the paper abstracts are the ones of the selected papers.
    The system have retrieved evidence to answer the user question.
    
    Goal:
    - You need to evaluate the relevance of the retrieved evidence to the user query and decide whether we should answer the question or not.
    - You can either choose to move on the answer the question or to ask for more evidence.
    - If you choose to answer the question you need to provide a very concise reasoning for your choice.
    - If you choose to ask for more evidence you need to provide a the limitation of the current retrieved evidence to help with the next retrieval attempt.

    General Guidelines:
    - If there is no limitation of the retrieved evidence, you should decide that the retrieved evidence is sufficient to answer the user query.
    - If there is major limitation of the retrieved evidence, you should decide that the retrieved evidence is not sufficient to answer the user query.
    - Sometimes the user's question might not be present in the paper, you just determine that the question is not answerable. If you retrieved a lot of evidences
    and none of them is remotely relevant to the user's question, you should decide that the question is not answerable, choose to answer the question by telling the user that the question is not answerable.
    """

    user_query = state.get("user_query", "")
    abstracts = get_paper_abstract(state.get("papers", []), state.get("selected_paper_ids", []))
    abstracts_text = "\n".join([
        f"Paper {paper_id}:\n{abstract}"
        for paper_id, abstract in abstracts.items()
    ])
    evidences = state.get("evidences", [])
    evidences_text = "\n\n".join([
        f"Evidence {i+1}:\n{evidence.page_content}"
        for i, evidence in enumerate(evidences)
    ]) if evidences else "No evidence retrieved yet."
    evaluation_prompt = f"""User query: {user_query}
    Paper abstracts:
    {abstracts_text}
    Retrieved evidences:
    {evidences_text}
    """

    class AskForMoreEvidence(BaseModel):
        limitation: str = Field(
            description="The limitation of the current retrieved evidence to help with the next retrieval attempt")
    
    class AnswerQuestion(BaseModel):
        reasoning: str = Field(
            description="The reasoning for why we should answer the user's question based on the retrieved evidence")

    class Evaluation(BaseModel):
        decision: Union[AskForMoreEvidence, AnswerQuestion] = Field(
            description="The decision for whether to retrieve more evidence or to answer the user's question")

    structured_model = qa_model.with_structured_output(Evaluation)
    decision_response = structured_model.invoke([
        SystemMessage(content=evaluation_system),
        HumanMessage(content=evaluation_prompt)
    ])

    if isinstance(decision_response.decision, AskForMoreEvidence):
        return {
            "messages": [AIMessage(content=decision_response.decision.limitation)],
            "limitation": decision_response.decision.limitation,
            "sufficient_evidence": False,
            "qa_iteration": state.get("qa_iteration", 0) + 1
        }
    elif isinstance(decision_response.decision, AnswerQuestion):
        return {
            "messages": [AIMessage(content=decision_response.decision.reasoning)],
            "sufficient_evidence": True,
            "qa_iteration": state.get("qa_iteration", 0) + 1
        }
    else:
        return {
            "messages": [AIMessage(content="Invalid decision")],
            "limitation": "Invalid decision",
            "sufficient_evidence": False,
            "qa_iteration": state.get("qa_iteration", 0) + 1
        }

def qa_answer(state: QAAgentState) -> QAAgentState:
    """
    Generate a final answer based on retrieved segments and reasoning.
    Combines all evidence and provides a concise yet complete response.
    """
    user_query = state.get("user_query", "")
    
    # Get all accumulated evidence
    evidences = state.get("evidences", [])
    evidences_text = "\n".join([
        f"Evidence {i}:\n{evidence.page_content}"
        for i, evidence in enumerate(evidences)
    ])
    
    limitation = state.get("limitation", "No limitation")
    
    # Get paper abstracts for context
    abstracts = get_paper_abstract(state.get("papers", []), state.get("selected_paper_ids", []))
    abstracts_text = "\n".join([
        f"Paper {paper_id}:\n{abstract}"
        for paper_id, abstract in abstracts.items()
    ])
    
    answer_prompt = f"""User question: {user_query}

Paper abstracts:
{abstracts_text}

Limitation of the retrieved evidence:
{limitation}

Retrieved evidences:
{evidences_text}

Based on the above evidence and analysis, provide a concise yet complete answer to the user's question. If the evidence is insufficient, acknowledge the limitations."""

    answer_system = """You are an expert research assistant that helps answer user questions.
    The user question is usually a research question about several selected papers and the paper abstracts are the ones of the selected papers.
    The system have retrieved evidence to answer the user question and the potential limitation of the retrieved evidence if any.
    
    Goal:
    - You need to provide a concise yet complete answer to the user's question.
    - You need to acknowledge the limitations of the evidence if the evidence is insufficient.
    - You need to provide a follow-up suggestions if the evidence is insufficient.

    General Strategy:
    - YOU SHOULD DIRECTLY ANSWER THE USER'S QUESTION FIRST, THEN PROVIDE THE EVIDENCE TO SUPPORT YOUR ANSWER.
    - If the answer is present in the retrieved evidence, you extract the answer from the evidence.
    - BE BRIEF, CONCISE AND FACTUALLY CONSISTENT, DON'T PROVIDE UNNECESSARY INFORMATION.
    - If the question is simple or can be extracted from the evidence. Answer the question as short as possible.
    """

    response = qa_model.invoke([
        SystemMessage(content=answer_system),
        HumanMessage(content=answer_prompt)
    ])
    
    return {
        "messages": [AIMessage(content=response.content)]
    }

def should_answer(state: QAAgentState) -> str:
    if state.get("sufficient_evidence", False):
        return "answer"
    elif state.get("qa_iteration", 0) >= 3:
        return "answer"
    else:
        return "retrieve"


def build_qa_graph():
    """
    Build the QA subgraph for paper question answering.

    Flow:
    1. qa_prepare: Prepare state for QA (select papers if needed)
    1. qa_retrieve: Search within selected papers for relevant segments
    2. qa_assess_quality: Check if evidence is sufficient
    3. If insufficient: qa_refine_retrieval (max 1 refinement)
    4. qa_answer: Generate grounded answer
    """
    

    qa_builder = StateGraph(QAAgentState)

    # Add nodes
    qa_builder.add_node("qa_retrieve", qa_retrieve)
    qa_builder.add_node("tools", ToolNode([retrieve_evidence_from_selected_papers]))
    qa_builder.add_node("qa_evaluate", qa_evaluate)
    qa_builder.add_node("qa_answer", qa_answer)

    # Add edges
    qa_builder.add_edge(START, "qa_retrieve")
    qa_builder.add_edge("qa_retrieve", "tools")
    qa_builder.add_edge("tools", "qa_evaluate")
    qa_builder.add_conditional_edges("qa_evaluate", should_answer, {
        "answer": "qa_answer",
        "retrieve": "qa_retrieve",
    })
    qa_builder.add_edge("qa_answer", END)

    return qa_builder.compile()


# Export the compiled QA graph
qa_graph = build_qa_graph()
