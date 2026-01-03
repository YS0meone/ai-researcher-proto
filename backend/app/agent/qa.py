"""
QA Agent Module - Agentic RAG for answering questions about selected papers.

This module implements:
1. Vector search within user-selected papers to retrieve relevant segments
2. Question answering grounded in the retrieved evidence
3. Iterative refinement if initial retrieval is insufficient
"""

from typing import Dict, List, Any, Optional
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from app.agent.states import State
from app.agent.utils import get_user_query, setup_langsmith
from app.core.config import settings         
from app.tools.search import vector_search_papers_by_ids, get_paper_abstract
import logging
import sys
import json

logger = logging.getLogger(__name__)
setup_langsmith()
qa_model = init_chat_model(model=settings.AGENT_MODEL_NAME, api_key=settings.OPENAI_API_KEY)

class RetrievalDecision(BaseModel):
    """Structured output for retrieval decision."""
    reasoning: str = Field(
        description="The reasoning for the decision")
    should_answer: bool = Field(
        description="Whether the current evidence is sufficient to answer the user's question")
    search_queries: List[str] = Field(
        description="The search queries to be executed")

QA_RETRIEVAL_SYSTEM = """You are an expert in evidence retrieval for academic paper QA.
You role is to determine whether the current evidence is sufficient to answer the user's question or we need to retrieve more evidence.

General Strategy:
- You should think step by step and reason about the user's query and the evidence.

Routing Strategy:
- You decision should be based on the following context: the user's query, the paper abstracts, the retrieved evidence and the past reasoning.  
- You should only decide to answer the user when you completely understand the user's query and the evidence is sufficient to answer the question or you are certain that the database does not contain any information that could answer the question.

Retrieval Strategy:
- If the evidence is insufficient, you should generate 1-3 focused search queries to find relevant evidence within specific papers.
- You should always understand the user's query first. If you are not sure about certain concept you should generate search queries to help you understand the user's query better.
"""


def qa_prepare(state: State) -> Dict:
    """
    If user hasn't selected specific papers, select top papers from search results.
    """
    selected_ids = state.get("selected_ids", [])
    papers = state.get("papers", [])
    intent = state.get("intent")

    # If QA-only mode and user has selected papers, use them
    if intent == "qa_only" and selected_ids:
        print(
            f"QA mode with {len(selected_ids)} pre-selected papers", file=sys.stderr)
        return {"sufficient_evidence": False}

    # If search_then_qa mode, select top papers for QA
    if intent == "search_then_qa" and papers and not selected_ids:
        # Select top 5 papers for detailed QA
        top_papers = papers[:5]
        selected_ids = [p["arxiv_id"] for p in top_papers]

        print(
            f"Auto-selecting top {len(selected_ids)} papers for QA", file=sys.stderr)

        return {
            "selected_ids": selected_ids,
            "qa_query": state.get("original_query"),
            "messages": [AIMessage(content=f"Selected top {len(selected_ids)} papers for detailed analysis.")],
            "sufficient_evidence": False
        }

    return {"sufficient_evidence": False}


def qa_retrieve(state: State) -> Dict:
    """
    Retrieve relevant segments from selected papers using vector search.

    Uses the selected_ids from state to scope the vector search to only
    the papers the user has chosen to ask questions about.
    """
    user_msg = get_user_query(state["messages"])
    selected_ids = state.get("selected_ids", [])

    if not selected_ids:
        print("WARNING: No papers selected for QA!", file=sys.stderr)
        return {
            "retrieved_segments": [],
            "messages": [AIMessage(content="No papers have been selected for Q&A. Please select papers first or use the paper finding mode.")]
        }

    abstracts = get_paper_abstract(selected_ids)
    segments = state.get("retrieved_segments", [])
    segments_text = "\n".join(segments)
    past_summaries = state.get("summaries", [])
    past_summaries_text = "\n".join(past_summaries)

    abstracts_text = "\n".join([
        f"Paper {paper_id}:\n{abstract}"
        for paper_id, abstract in abstracts.items()
    ])
    
    retrieval_prompt = f"""User question: {user_msg}

Selected papers to search: {selected_ids}

Paper abstracts:
{abstracts_text}

Past summaries:
{past_summaries_text}

Retrieved evidence:
{segments_text}"""

    structured_model = qa_model.with_structured_output(RetrievalDecision)
    plan = structured_model.invoke([
        SystemMessage(content=QA_RETRIEVAL_SYSTEM),
        HumanMessage(content=retrieval_prompt)
    ])

    if plan.should_answer:
        return {
            "messages": [AIMessage(content=plan.reasoning)],
            "sufficient_evidence": True
        }
    
    tool_model = qa_model.bind_tools([vector_search_papers_by_ids])
    response = tool_model.invoke([
        SystemMessage(content="Create vector search tool calls based on the search queries"),
        HumanMessage(content=f"Search queries: {plan.search_queries}, Selected papers: {selected_ids}")
    ])

    return {
        "messages": [response],
        "sufficient_evidence": False
    }

def qa_rerank(state: State) -> Dict:
    """
    Extract tool results and rerank segments based on relevance.
    Uses LLM to evaluate which segments are most relevant to the user's query.
    """
    messages = state.get("messages", [])
    user_msg = get_user_query(messages)
    selected_ids = state.get("selected_ids", [])
    
    # Extract tool call results from the last tool messages
    new_segments = []
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            # Tool message content is a JSON string of search results
            content = msg.content
            try:
                if isinstance(content, str):
                    parsed = json.loads(content)
                    if isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, dict) and item.get("supporting_detail") and item["supporting_detail"] not in new_segments:
                                new_segments.append(item["supporting_detail"])
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("supporting_detail") and item["supporting_detail"] not in new_segments:
                            new_segments.append(item["supporting_detail"])
            except json.JSONDecodeError:
                # If not JSON, append as is
                new_segments.append(content)
        elif hasattr(msg, 'tool_calls') and msg.tool_calls:
            # Stop when we hit the tool call request
            break
    
    if not new_segments:
        return {"messages": [AIMessage(content="No new segments found. Please try again.")]}
    # Get context for reranking
    abstracts = get_paper_abstract(selected_ids)
    abstracts_text = "\n".join([
        f"Paper {paper_id}:\n{abstract}"
        for paper_id, abstract in abstracts.items()
    ])
    
    past_summaries = state.get("summaries", [])
    past_summaries_text = "\n".join(past_summaries)
    
    new_segments_text = "\n".join([f"Segment {i}:\n{segment}" for i, segment in enumerate(new_segments)])
    
    rerank_system = """You are an expert in evaluating the relevance of retrieved evidence for answering a research question.
    Your would be presented with a user question, paper abstracts, past summaries and new retrieved segments.
    The user question is usually a research question about several selected papers and the paper abstracts are the ones of the selected papers.
    The system have retrieved evidence to answer the user question or to help you understand the user question better.
    The past summaries are the summaries of the system about the user question and the retrieved evidence.

    Goal:
    - You need to select the index of the segements that can help you answer the user question, and order them by their relevance to the user question.
    - You need to summarize the information you learned from the segments that are helpful to answer the user question.

    General Strategy:
    - You should think in step by step manner.
    - The selected segments should only be the ones that are the direct evidence to answer the user question.
    - For selected segements, you should only generate the selected index of the segments. The index is present at the beginning of the segment text in format of "Segment x:" where x is the index of the segment.
    - The summary you generate should be covering the useful information you learned from the segements and help the system to generate a better search query or answer the user question.
    - You should not select any segment that is not directly related to the user question, it's ok you don't select any segment.
    - You should output the selected index of the segments in the order of their relevance to the user question.

    Tips:
    - The retrieved segments may not be directly related to the user question, but they may help you understand the user question better and generate better search queries.
    - For the segments that are not directly related to the user question, you should summarize the information you learned from them and don't select them in the final output.
    - Some segments may be very short and don't contain any useful information, you should not select them."""

    # Use LLM to evaluate and extract reasoning about relevance
    rerank_prompt = f"""User question: {user_msg}

Paper abstracts:
{abstracts_text}
Past Summaries:
{past_summaries_text}

New retrieved segments:
{new_segments_text}

Select the relevant segments and generate a summary of the information you learned from them."""

    class BatchRerankResult(BaseModel):
        selected_idx: List[int] = Field(description="The index of the selected segments")
        summary: str = Field(description="The summary of the selected segments")

    structured_model = qa_model.with_structured_output(BatchRerankResult)
    response = structured_model.invoke([
        SystemMessage(content=rerank_system),
        HumanMessage(content=rerank_prompt)
    ])

    existing_segments = state.get("retrieved_segments", [])
    selected_segments = [new_segments[i] for i in response.selected_idx if new_segments[i] not in existing_segments]
    # Store the new segments and reasoning
    existing_summaries = state.get("summaries", [])
    qa_iteration = state.get("qa_iteration", 0) + 1
    
    return {
        "retrieved_segments": existing_segments + selected_segments,
        "summaries": existing_summaries + [response.summary],
        "qa_iteration": qa_iteration
    }

def qa_answer(state: State) -> Dict:
    """
    Generate a final answer based on retrieved segments and reasoning.
    Combines all evidence and provides a concise yet complete response.
    """
    messages = state.get("messages", [])
    user_msg = get_user_query(messages)
    selected_ids = state.get("selected_ids", [])
    
    # Get all accumulated evidence
    segments = state.get("retrieved_segments", [])
    segments_text = "\n\n".join(segments) if segments else "No evidence retrieved."
    
    summaries = state.get("summaries", [])
    summaries_text = "\n\n".join(summaries) if summaries else "No analysis available."
    
    # Get paper abstracts for context
    abstracts = get_paper_abstract(selected_ids)
    abstracts_text = "\n".join([
        f"Paper {paper_id}:\n{abstract}"
        for paper_id, abstract in abstracts.items()
    ])
    
    answer_prompt = f"""User question: {user_msg}

Paper abstracts:
{abstracts_text}

Summaries:
{summaries_text}

Retrieved evidence:
{segments_text}

Based on the above evidence and analysis, provide a concise yet complete answer to the user's question. If the evidence is insufficient, acknowledge the limitations."""

    response = qa_model.invoke([
        SystemMessage(content="You are an expert research assistant. Provide accurate, well-grounded answers based on the evidence provided. Cite specific findings from the papers when relevant."),
        HumanMessage(content=answer_prompt)
    ])
    
    return {
        "messages": [AIMessage(content=response.content)]
    }

def should_answer(state: State) -> str:
    if state.get("sufficient_evidence", False):
        return "answer"
    elif state.get("qa_iteration", 0) >= 3:
        return "answer"
    else:
        return "tools"


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
    from langgraph.graph import StateGraph, END

    qa_builder = StateGraph(State)

    # Add nodes
    qa_builder.add_node("qa_prepare", qa_prepare)
    qa_builder.add_node("qa_retrieve", qa_retrieve)
    qa_builder.add_node("tools", ToolNode([vector_search_papers_by_ids]))
    qa_builder.add_node("qa_rerank", qa_rerank)
    qa_builder.add_node("qa_answer", qa_answer)

    # Add edges
    qa_builder.set_entry_point("qa_prepare")
    qa_builder.add_edge("qa_prepare", "qa_retrieve")
    qa_builder.add_conditional_edges("qa_retrieve", should_answer, {
        "answer": "qa_answer",
        "tools": "tools",
    })
    qa_builder.add_edge("tools", "qa_rerank")
    qa_builder.add_edge("qa_rerank", "qa_retrieve")
    qa_builder.add_edge("qa_answer", END)

    return qa_builder.compile()


# Export the compiled QA graph
qa_graph = build_qa_graph()
