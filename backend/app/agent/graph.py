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
    orchestrator_route_decision_entry,
    orchestrator_route_decision_after_paper_finder,
)

# Import paper finder subgraph
from app.agent.paper_finder import paper_finder

# Import QA subgraph
from app.agent.qa import qa_graph

logger = logging.getLogger(__name__)

model = init_chat_model(model=settings.AGENT_MODEL_NAME,
                        api_key=settings.OPENAI_API_KEY)


# ============================================================================
# BUILD UNIFIED GRAPH
# ============================================================================

graph_builder = StateGraph(State)

# Add all nodes
graph_builder.add_node("orchestrator", orchestrator_intent_analysis)
graph_builder.add_node("paper_finder", paper_finder)
graph_builder.add_node("qa_agent", qa_graph)  # QA subgraph

# Entry point: orchestrator analyzes intent
graph_builder.add_edge(START, "orchestrator")

# From orchestrator, route based on intent
graph_builder.add_conditional_edges(
    "orchestrator",
    orchestrator_route_decision_entry,
    {
        "search": "paper_finder",
        "qa": "qa_agent",
        "refusal": END,
    }
)

# After paper finder, decide whether to end or go to QA
graph_builder.add_conditional_edges(
    "paper_finder",
    orchestrator_route_decision_after_paper_finder,
    {
        "end": END,              # No QA needed - end here
        "qa": "qa_agent",      # Papers sufficient - proceed to QA
    }
)

# QA agent goes directly to END
graph_builder.add_edge("qa_agent", END)

# Compile the graph
graph = graph_builder.compile()
