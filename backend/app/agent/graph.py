"""
Main Graph - Unified multi-agent workflow for AI research assistant.

This module orchestrates:
1. Intent analysis and query optimization (orchestrator)
2. Paper search with iterative refinement (paper_finder)
3. Question answering on selected papers (qa)
"""

from typing import Annotated, List, Dict, Any, Optional, Literal, Sequence
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from app.core.config import settings
from app.tools.search import (
    hybrid_search_papers,
    semantic_search_papers,
    keyword_search_papers,
    search_papers_by_category,
    vector_search_papers,
)
from pydantic import BaseModel, Field
import logging
from app.agent.states import State
from app.agent.utils import get_user_query

# Import orchestrator functions
from app.agent.orchestrator import (
    orchestrator_intent_analysis,
    orchestrator_evaluate_papers,
    orchestrator_route_decision,
    orchestrator_prepare_qa,
    should_evaluate_papers,
)

# Import paper finder functions
from app.agent.paper_finder import (
    search_agent,
    merge_and_rerank,
    increment_iter,
)

# Import QA subgraph
from app.agent.qa import qa_graph

logger = logging.getLogger(__name__)

model = init_chat_model(model=settings.MODEL_NAME,
                        api_key=settings.OPENAI_API_KEY)


# ============================================================================
# BUILD UNIFIED GRAPH
# ============================================================================

graph_builder = StateGraph(State)

# Add all nodes
graph_builder.add_node("orchestrator", orchestrator_intent_analysis)
graph_builder.add_node("search_agent", search_agent)
graph_builder.add_node("tools", ToolNode([
    hybrid_search_papers,
    semantic_search_papers,
    keyword_search_papers,
    search_papers_by_category,
    vector_search_papers,
]))
graph_builder.add_node("merge_and_rerank", merge_and_rerank)
graph_builder.add_node("increment_iter", increment_iter)
graph_builder.add_node("evaluate_papers", orchestrator_evaluate_papers)
graph_builder.add_node("prepare_qa", orchestrator_prepare_qa)
graph_builder.add_node("qa_agent", qa_graph)  # QA subgraph

# Entry point: orchestrator analyzes intent
graph_builder.add_edge(START, "orchestrator")

# From orchestrator, route based on intent
graph_builder.add_conditional_edges(
    "orchestrator",
    orchestrator_route_decision,
    {
        "search": "search_agent",
        "qa": "prepare_qa",
    }
)

# Search flow: search_agent -> tools -> merge_and_rerank
graph_builder.add_edge("search_agent", "tools")
graph_builder.add_edge("tools", "merge_and_rerank")

# After merge_and_rerank, decide whether to evaluate or go to QA
graph_builder.add_conditional_edges(
    "merge_and_rerank",
    should_evaluate_papers,
    {
        "evaluate": "evaluate_papers",
        "qa": "prepare_qa",
    }
)

# After evaluation, route based on sufficiency
graph_builder.add_conditional_edges(
    "evaluate_papers",
    orchestrator_route_decision,
    {
        "search": "increment_iter",  # Need more papers - increment and iterate
        "qa": "prepare_qa",           # Papers sufficient - proceed to QA
    }
)

# Iteration loop: increment_iter -> search_agent
graph_builder.add_edge("increment_iter", "search_agent")

# Prepare QA and then run QA agent
graph_builder.add_edge("prepare_qa", "qa_agent")

# QA agent goes directly to END
graph_builder.add_edge("qa_agent", END)

# Compile the graph
graph = graph_builder.compile()
