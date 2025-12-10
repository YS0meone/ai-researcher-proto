from app.agent.qa import qa_graph
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
from app.agent.prompts import (
    RerankingPrompts,
    SynthesisPrompts,
    SearchAgentPrompts,
    OrchestratorPrompts,
)
from app.agent.orchestrator import (
    orchestrator_intent_analysis,
    orchestrator_evaluate_papers,
    orchestrator_route_decision,
    orchestrator_prepare_qa,
    should_evaluate_papers,
)
from pydantic import BaseModel, Field
import logging
from app.agent.states import State
from app.agent.utils import get_user_query

logger = logging.getLogger(__name__)

model = init_chat_model(model=settings.MODEL_NAME,
                        api_key=settings.OPENAI_API_KEY)

# ============================================================================
# STRUCTURED OUTPUT MODELS
# ============================================================================


class RouterDecision(BaseModel):
    """Structured output for routing decisions."""
    route: Literal["search", "synthesize"]
    short_reason: str


class QueryGeneration(BaseModel):
    """Structured output for query generation."""
    queries: List[str] = Field(max_length=4)


class SynthesisDecision(BaseModel):
    """Structured output for synthesis planning."""
    needs_deep_search: bool
    reasoning: str
    search_query: Optional[str] = None


# ============================================================================
# GRAPH NODES
# ============================================================================

def search_agent(state: State) -> Dict:
    """
    Unified search agent that generates queries and selects tools.
    Returns ONE AIMessage with MULTIPLE tool_calls to avoid LangGraph tool response mismatch.
    Uses optimized query from orchestrator if available.
    """
    # Use optimized query from orchestrator if available, otherwise get from messages
    user_msg = state.get("optimized_query") or get_user_query(
        state["messages"])

    # Get search plan with structured output
    prompt = SearchAgentPrompts.format_planning(
        user_msg=user_msg,
        search_queries=state.get("search_queries", [])
    )

    from app.agent.paper_finder import SearchPlan
    structured_model = model.with_structured_output(SearchPlan)
    plan = structured_model.invoke([
        SystemMessage(content=SearchAgentPrompts.SYSTEM),
        *state["messages"],
        HumanMessage(content=prompt)
    ])

    # Build tool instructions from the plan
    tool_instructions = f"Execute the following search strategy: {plan.strategy}\n\nSearches to perform:\n"
    for tc in plan.tool_calls:
        tool_instructions += f"- {tc.tool_name}(query='{tc.query}', limit={tc.limit}) - {tc.reasoning}\n"

    # Bind tools and get ONE AIMessage with MULTIPLE tool_calls
    tool_bound = model.bind_tools([
        hybrid_search_papers,
        semantic_search_papers,
        keyword_search_papers,
        search_papers_by_category,
        vector_search_papers,
    ])

    # Invoke with explicit tool call instructions
    ai_message = tool_bound.invoke([
        SystemMessage(
            content="You are a search execution agent. Execute the planned searches by calling the appropriate tools."),
        HumanMessage(content=tool_instructions)
    ])

    # Extract queries for state tracking
    queries = [tc.query for tc in plan.tool_calls]
    search_queries = state.get("search_queries", [])
    seen = set(q.lower() for q in search_queries)
    merged = search_queries + \
        [q for q in queries if q and q.lower() not in seen]

    return {
        "search_queries": merged[:6],
        "messages": [ai_message]  # Single AIMessage with multiple tool_calls!
    }


def merge_and_rerank(state: State) -> Dict:
    # Collect the last tool outputs from messages
    import sys
    import json
    candidates: List[Dict[str, Any]] = []

    for idx, m in enumerate(state["messages"]):
        msg_type = getattr(m, 'type', None) or (
            m.get('type') if isinstance(m, dict) else None)
        msg_class = m.__class__.__name__

        # Check if this is a tool message
        is_tool = (isinstance(m, dict) and m.get("type") == "tool") or (
            hasattr(m, "type") and m.type == "tool") or msg_class == "ToolMessage"
        if is_tool:
            content = m.get("content") if isinstance(
                m, dict) else getattr(m, "content", None)

            # Try to extract results
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("arxiv_id"):
                        candidates.append(item)
            elif isinstance(content, str):
                # Tool content might be JSON string
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, dict) and item.get("arxiv_id"):
                                candidates.append(item)
                except:
                    pass

    papers = state.get("papers", [])
    print(f"Total candidates found: {len(candidates)}", file=sys.stderr)
    print(f"Existing papers in state: {len(papers)}", file=sys.stderr)

    # Deduplicate by arxiv_id and keep best score
    seen = {p["arxiv_id"]: p for p in papers}
    for c in candidates:
        aid = c["arxiv_id"]
        prev = seen.get(aid)
        if not prev:
            seen[aid] = c
        else:
            def best(a, b, key):
                sa, sb = a.get(key), b.get(key)
                if sa is None:
                    return b
                if sb is None:
                    return a
                return a if sa >= sb else b
            seen[aid] = best(prev, c, "search_score" if "search_score" in c else (
                "similarity_score" if "similarity_score" in c else "text_score"))

    merged = list(seen.values())[:100]

    print(f"Merged papers count: {len(merged)}", file=sys.stderr)

    # Rerank with LLM (title+abstract snippet) and compute coverage heuristic
    user_msg = state.get("optimized_query") or state.get(
        "original_query") or get_user_query(state["messages"])
    short_list = merged[:30]

    print(
        f"Short list for reranking: {len(short_list)} papers", file=sys.stderr)

    if len(short_list) == 0:
        print("WARNING: No papers to rerank! Returning empty results.", file=sys.stderr)
        return {"papers": [], "coverage_score": 0.0}

    prompt = RerankingPrompts.format_reranking(
        user_msg=user_msg,
        candidates=short_list
    )

    from app.agent.paper_finder import RerankingResult
    structured_model = model.with_structured_output(RerankingResult)
    result = structured_model.invoke([
        SystemMessage(content=RerankingPrompts.SYSTEM),
        *state["messages"],
        HumanMessage(content=prompt)
    ])

    order = result.order
    coverage = result.coverage_score

    # Fallback if no valid order returned
    if not order:
        print("WARNING: No valid order returned, using default", file=sys.stderr)
        order = [p.get("arxiv_id") for p in short_list]

    id_to_p = {p["arxiv_id"]: p for p in merged}
    reranked = [id_to_p[i] for i in order if i in id_to_p] + \
        [p for p in merged if p["arxiv_id"] not in set(order)]

    print(f"Final reranked papers: {len(reranked[:50])}", file=sys.stderr)

    return {"papers": reranked[:50], "coverage_score": coverage, "messages": [HumanMessage(content=f"Reranked {len(reranked[:50])} papers, coverage: {coverage:.2f}")]}


def synthesize(state: State) -> Dict:
    import sys
    user_msg = state.get("original_query") or get_user_query(state["messages"])
    papers = state.get("papers", [])

    # First, determine if we need deep content search
    decision_prompt = SynthesisPrompts.format_decision(
        user_msg=user_msg,
        num_papers=len(papers)
    )

    structured_model = model.with_structured_output(SynthesisDecision)
    decision = structured_model.invoke([
        SystemMessage(content=SynthesisPrompts.DECISION_SYSTEM),
        *state["messages"],
        HumanMessage(content=decision_prompt)
    ])

    needs_deep_search = decision.needs_deep_search
    search_query = decision.search_query or user_msg
    print(
        f"Synthesize decision: needs_deep_search={needs_deep_search}, reasoning={decision.reasoning}", file=sys.stderr)

    # If deep search is needed, call vector_search_papers
    detailed_segments = []
    if needs_deep_search and papers:
        print(
            f"Performing vector search with query: {search_query}", file=sys.stderr)
        try:
            # Call vector_search_papers tool directly
            results = vector_search_papers.invoke(
                {"query": search_query, "limit": 5})
            if isinstance(results, list) and results:
                detailed_segments = results
                print(
                    f"Vector search returned {len(detailed_segments)} results with detailed segments", file=sys.stderr)
        except Exception as e:
            print(f"Vector search failed: {e}", file=sys.stderr)

    # Build paper details from state
    paper_details = []
    for p in papers[:10]:
        paper_details.append({
            "arxiv_id": p.get("arxiv_id"),
            "title": p.get("title"),
            "abstract": p.get("abstract"),
            "authors": p.get("authors"),
            "categories": p.get("categories"),
        })

    # Produce final grounded answer
    if detailed_segments:
        answer_prompt = SynthesisPrompts.format_answer_with_details(
            user_msg=user_msg,
            paper_details=paper_details[:5],
            detailed_segments=detailed_segments
        )
    else:
        answer_prompt = SynthesisPrompts.format_answer_basic(
            user_msg=user_msg,
            paper_details=paper_details
        )

    final = model.invoke([
        SystemMessage(content=SynthesisPrompts.ANSWER_SYSTEM),
        *state["messages"],
        HumanMessage(content=answer_prompt)
    ])
    return {"messages": [final]}


# ============================================================================
# BUILD GRAPH WITH ORCHESTRATOR COORDINATION
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
graph_builder.add_node("evaluate_papers", orchestrator_evaluate_papers)
graph_builder.add_node("prepare_qa", orchestrator_prepare_qa)
graph_builder.add_node("qa_agent", qa_graph)  # QA subgraph
graph_builder.add_node("synthesize", synthesize)

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
        "search": "search_agent",  # Need more papers
        "qa": "prepare_qa",        # Papers sufficient
    }
)

# Prepare QA and then run QA agent
graph_builder.add_edge("prepare_qa", "qa_agent")

# QA agent can either synthesize or end
graph_builder.add_conditional_edges(
    "qa_agent",
    lambda s: "synthesize" if s.get("intent") == "search_then_qa" else "end",
    {
        "synthesize": "synthesize",
        "end": END,
    }
)

graph_builder.add_edge("synthesize", END)

graph = graph_builder.compile()
