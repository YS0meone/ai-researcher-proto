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
    intent: str = Field(pattern="^(search_then_qa|qa_only|search_only)$")
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
        - intent: "search_then_qa", "qa_only", or "search_only"
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


def orchestrator_evaluate_papers(state: State) -> Dict:
    """
    Evaluate if retrieved papers are sufficient to answer the query.

    Returns:
        - If sufficient: proceed to QA
        - If insufficient and iterations < 3: refine query and search again
        - If max iterations reached: proceed with what we have
    """
    user_msg = state.get("original_query") or get_user_query(state["messages"])
    papers = state.get("papers", [])
    coverage_score = state.get("coverage_score", 0.0)
    iteration = state.get("paper_search_iteration", 0)

    print(
        f"Evaluating papers: count={len(papers)}, coverage={coverage_score}, iteration={iteration}", file=sys.stderr)

    # Quick heuristic checks
    if len(papers) >= 5 and coverage_score >= 0.65:
        print("Quick check: Sufficient papers found (heuristic)", file=sys.stderr)
        return {
            "route": "qa",
            "messages": [AIMessage(content=f"Found {len(papers)} relevant papers with coverage score {coverage_score:.2f}. Proceeding to answer your question.")]
        }

    if iteration >= 3:
        print("Max iterations reached, proceeding with available papers",
              file=sys.stderr)
        return {
            "route": "qa",
            "messages": [AIMessage(content=f"Searched 3 times. Found {len(papers)} papers. Proceeding with available information.")]
        }

    # Detailed LLM evaluation
    eval_prompt = OrchestratorPrompts.format_paper_evaluation(
        user_msg=user_msg,
        papers=papers,
        coverage_score=coverage_score,
        iteration=iteration
    )

    eval_model = orchestrator_model.with_structured_output(PaperEvaluation)

    try:
        eval_result = eval_model.invoke([
            SystemMessage(content=OrchestratorPrompts.PAPER_EVALUATION_SYSTEM),
            HumanMessage(content=eval_prompt)
        ])
    except Exception as e:
        print(f"ERROR in paper evaluation: {e}", file=sys.stderr)
        eval_result = None

    # Handle None or failed response with fallback
    if eval_result is None or not hasattr(eval_result, 'sufficient'):
        print("WARNING: Paper evaluation failed, proceeding to QA with available papers", file=sys.stderr)
        return {
            "route": "qa",
            "messages": [AIMessage(content=f"Proceeding with {len(papers)} papers found.")]
        }

    print(
        f"Evaluation: sufficient={eval_result.sufficient}, confidence={eval_result.confidence}", file=sys.stderr)
    print(f"Reasoning: {eval_result.reasoning}", file=sys.stderr)

    if eval_result.sufficient or eval_result.confidence >= 0.6:
        return {
            "route": "qa",
            "messages": [AIMessage(content=f"Retrieved papers are sufficient. {eval_result.reasoning}")]
        }
    else:
        # Need to refine and search again
        refined_query = eval_result.refined_query or state.get(
            "optimized_query") or user_msg

        print(
            f"Papers insufficient. Refining query: {refined_query}", file=sys.stderr)
        if eval_result.missing_aspects:
            print(
                f"Missing aspects: {eval_result.missing_aspects}", file=sys.stderr)

        return {
            "route": "search",
            "optimized_query": refined_query,
            "paper_search_iteration": iteration + 1,
            "messages": [AIMessage(content=f"Need more papers. Missing: {', '.join(eval_result.missing_aspects or [])}. Refining search...")]
        }


def orchestrator_route_decision(state: State) -> str:
    """
    Decide the next step based on intent and current state.

    Returns:
        - "search": Go to paper finder
        - "qa": Go to QA agent
        - "evaluate": Evaluate papers after search
    """
    intent = state.get("intent")
    route = state.get("route")

    # If route is explicitly set by evaluation, use it
    if route:
        print(f"Using explicit route: {route}", file=sys.stderr)
        return route

    # Otherwise, decide based on intent
    if intent == "qa_only":
        print("Intent is qa_only, routing to qa", file=sys.stderr)
        return "qa"
    elif intent in ["search_then_qa", "search_only"]:
        print(f"Intent is {intent}, routing to search", file=sys.stderr)
        return "search"
    else:
        # Default to search
        print("No clear intent, defaulting to search", file=sys.stderr)
        return "search"


def orchestrator_prepare_qa(state: State) -> Dict:
    """
    Prepare state for QA agent.

    If user hasn't selected specific papers, select top papers from search results.
    """
    selected_ids = state.get("selected_ids", [])
    papers = state.get("papers", [])
    intent = state.get("intent")

    # If QA-only mode and user has selected papers, use them
    if intent == "qa_only" and selected_ids:
        print(
            f"QA mode with {len(selected_ids)} pre-selected papers", file=sys.stderr)
        return {}

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
            "messages": [AIMessage(content=f"Selected top {len(selected_ids)} papers for detailed analysis.")]
        }

    return {}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def should_evaluate_papers(state: State) -> str:
    """
    After paper search, decide if we should evaluate or continue.

    Returns:
        - "evaluate": Evaluate papers
        - "qa": Skip evaluation, go straight to QA
    """
    intent = state.get("intent")

    # If search_only, don't evaluate - just return results
    if intent == "search_only":
        print("Search-only mode, skipping evaluation", file=sys.stderr)
        return "qa"

    # If search_then_qa, evaluate papers
    if intent == "search_then_qa":
        print("Search-then-QA mode, evaluating papers", file=sys.stderr)
        return "evaluate"

    # Default: evaluate
    return "evaluate"
