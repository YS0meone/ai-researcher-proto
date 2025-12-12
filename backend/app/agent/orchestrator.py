"""
Orchestrator Module - Central coordinator for the AI research assistant.

This module implements:
1. Intent analysis - Determine user's goal (search, QA, or both)
2. Query optimization - Refine queries for better search results
3. Paper evaluation - Assess if retrieved papers are sufficient
4. Workflow coordination - Route between paper finder and QA agent
"""

from typing import Dict, Optional
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
import logging
import sys

from app.agent.states import State
from app.agent.utils import get_user_query
from app.agent.prompts import OrchestratorPrompts
from app.core.config import settings

logger = logging.getLogger(__name__)

orchestrator_model = init_chat_model(
    model=settings.MODEL_NAME,
    api_key=settings.OPENAI_API_KEY
)


# ============================================================================
# STRUCTURED OUTPUT MODELS
# ============================================================================

class IntentDecision(BaseModel):
    """Orchestrator intent analysis output."""
    intent: str = Field(
        pattern="^(search_then_qa|qa_only|search_only|non_cs_query)$")
    reasoning: str


class QueryOptimization(BaseModel):
    """Query optimization output."""
    optimized_query: str
    reasoning: str


class PaperEvaluation(BaseModel):
    """Paper evaluation output."""
    sufficient: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    missing_aspects: Optional[list[str]] = None
    refined_query: Optional[str] = None


# ============================================================================
# ORCHESTRATOR NODES
# ============================================================================

def orchestrator_intent_analysis(state: State) -> Dict:
    """
    Analyze user intent and decide the workflow.

    Returns:
        - intent: "search_then_qa", "qa_only", "search_only", or "non_cs_query"
        - optimized_query: If search is needed, optimize the query
        - original_query: Store for reference
    """
    user_msg = get_user_query(state["messages"])
    num_papers = len(state.get("papers", []))
    selected_ids = state.get("selected_ids", [])
    coverage_score = state.get("coverage_score", 0.0)

    print(
        f"Orchestrator analyzing intent for: {user_msg[:50]}...", file=sys.stderr)
    print(
        f"State: papers={num_papers}, selected_ids={len(selected_ids)}, coverage={coverage_score}", file=sys.stderr)

    # Analyze intent
    intent_prompt = OrchestratorPrompts.format_intent_analysis(
        user_msg=user_msg,
        num_papers=num_papers,
        selected_ids=selected_ids,
        coverage_score=coverage_score
    )

    structured_model = orchestrator_model.with_structured_output(
        IntentDecision)

    try:
        intent_result = structured_model.invoke([
            SystemMessage(content=OrchestratorPrompts.INTENT_SYSTEM),
            *state.get("messages", []),
            HumanMessage(content=intent_prompt)
        ])
    except Exception as e:
        print(f"ERROR in intent analysis: {e}", file=sys.stderr)
        intent_result = None

    # Handle None or failed response with fallback
    if intent_result is None or not hasattr(intent_result, 'intent'):
        print(
            "WARNING: Intent analysis failed, defaulting to search_then_qa", file=sys.stderr)
        intent = "search_then_qa"
        reasoning = "Fallback due to LLM error"
    else:
        intent = intent_result.intent
        reasoning = intent_result.reasoning

    print(
        f"Intent determined: {intent} - {reasoning}", file=sys.stderr)

    # Initialize state updates
    updates = {
        "intent": intent,
        "original_query": user_msg,
        "paper_search_iteration": 0
    }
    print(f"Intent is {intent}", file=sys.stderr)

    # If non-CS query detected, add message and return early
    if intent == "non_cs_query":
        updates["messages"] = [AIMessage(content=f"I'm an AI research assistant focused on computer science topics. Your query appears to be outside the computer science domain. I can help you with questions about machine learning, algorithms, software engineering, systems, NLP, computer vision, and other CS topics. Please ask a computer science-related question.")]
        return updates

    # If search is needed, optimize the query
    if intent in ["search_then_qa", "search_only"]:
        previous_queries = state.get("search_queries", [])

        opt_prompt = OrchestratorPrompts.format_query_optimization(
            user_msg=user_msg,
            previous_queries=previous_queries
        )

        opt_model = orchestrator_model.with_structured_output(
            QueryOptimization)

        try:
            opt_result = opt_model.invoke([
                SystemMessage(
                    content=OrchestratorPrompts.QUERY_OPTIMIZATION_SYSTEM),
                HumanMessage(content=opt_prompt)
            ])
        except Exception as e:
            print(f"ERROR in query optimization: {e}", file=sys.stderr)
            opt_result = None

        # Handle None or failed response with fallback
        if opt_result is None or not hasattr(opt_result, 'optimized_query'):
            print(
                "WARNING: Query optimization failed, using original query", file=sys.stderr)
            updates["optimized_query"] = user_msg
        else:
            updates["optimized_query"] = opt_result.optimized_query
            print(
                f"Query optimized: '{user_msg}' → '{opt_result.optimized_query}'", file=sys.stderr)
            print(f"Reasoning: {opt_result.reasoning}", file=sys.stderr)

    return updates


def orchestrator_route_decision_entry(state: State) -> str:
    """
    Decide the next step based on intent and current state.

    Returns:
        - "search": Go to paper finder
        - "qa": Go to QA agent
        - "refusal": End due to non-CS query
    """
    intent = state.get("intent")

    if intent == "qa_only":
        print("Intent is qa_only, routing to qa", file=sys.stderr)
        return "qa"
    elif intent == "non_cs_query":
        print("Intent is non_cs_query, routing to END", file=sys.stderr)
        return "refusal"
    elif intent in ["search_then_qa", "search_only"]:
        print(f"Intent is {intent}, routing to search", file=sys.stderr)
        return "search"
    else:
        # Default to search
        print("No clear intent, defaulting to search", file=sys.stderr)
        return "search"


def orchestrator_route_decision_after_paper_finder(state: State) -> str:
    """
    Decide the next step based on intent and current state.

    Returns:
        - "qa": Go to QA agent
        - "end": End if no QA needed
    """
    intent = state.get("intent")

    if intent == "search_only":
        print(f"Intent is {intent}, routing to end", file=sys.stderr)
        return "end"
    elif intent == "search_then_qa":
        print(f"Intent is {intent}, routing to qa", file=sys.stderr)
        return "qa"
    else:
        # Default to qa
        print("No clear intent, defaulting to qa", file=sys.stderr)
        return "qa"
