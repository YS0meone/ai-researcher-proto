from typing import List, Dict, Any, Optional, Literal
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from app.core.config import settings
from app.tools.search import (
    hybrid_search_papers,
    semantic_search_papers,
    keyword_search_papers,
    vector_search_papers,
)
from app.agent.prompts import (
    RerankingPrompts,
    SearchAgentPrompts,
)
from pydantic import BaseModel, Field
import logging
from app.agent.states import State
from app.agent.utils import get_user_query

logger = logging.getLogger(__name__)

paper_finder_model = init_chat_model(
    model=settings.MODEL_NAME, api_key=settings.OPENAI_API_KEY)


class QueryGeneration(BaseModel):
    """Structured output for query generation."""
    queries: List[str] = Field(max_length=4)


class SynthesisDecision(BaseModel):
    """Structured output for synthesis planning."""
    needs_deep_search: bool
    reasoning: str
    search_query: Optional[str] = None


class SearchToolCall(BaseModel):
    """Structured output for a single search tool call."""
    tool_name: Literal["hybrid_search_papers", "semantic_search_papers",
                       "keyword_search_papers", "vector_search_papers",
                       "search_papers_by_category"]
    query: str
    limit: int = 15
    reasoning: str


class SearchPlan(BaseModel):
    """Structured output for search planning."""
    tool_calls: List[SearchToolCall] = Field(min_length=1, max_length=5)
    strategy: str


class RerankingResult(BaseModel):
    """Structured output for paper reranking."""
    order: List[str]
    coverage_score: float = Field(ge=0.0, le=1.0)
    brief_reasoning: Optional[str] = None


def search_agent(state: State) -> Dict:
    """
    Unified search agent that generates queries and selects tools.
    Returns ONE AIMessage with MULTIPLE tool_calls to avoid LangGraph tool response mismatch.
    """
    user_msg = get_user_query(state["messages"])

    # Get search plan with structured output
    prompt = SearchAgentPrompts.format_planning(
        user_msg=user_msg,
        search_queries=state.get("search_queries", [])
    )

    paper_finder_model = paper_finder_model.with_structured_output(SearchPlan)
    plan = paper_finder_model.invoke([
        SystemMessage(content=SearchAgentPrompts.SYSTEM),
        *state["messages"],
        HumanMessage(content=prompt)
    ])

    # Build tool instructions from the plan
    tool_instructions = f"Execute the following search strategy: {plan.strategy}\n\nSearches to perform:\n"
    for tc in plan.tool_calls:
        tool_instructions += f"- {tc.tool_name}(query='{tc.query}', limit={tc.limit}) - {tc.reasoning}\n"

    # Bind tools and get ONE AIMessage with MULTIPLE tool_calls
    tool_bound = paper_finder_model.bind_tools([
        hybrid_search_papers,
        semantic_search_papers,
        keyword_search_papers,
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
    user_msg = get_user_query(state["messages"])
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

    paper_finder_model = paper_finder_model.with_structured_output(
        RerankingResult)
    result = paper_finder_model.invoke([
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


def decide_next(state: State):
    """Decide whether to continue searching or synthesize results."""
    coverage = state.get("coverage_score", 0.0)
    iter_count = state.get("iter", 0)
    max_iters = state.get("max_iters", 3)

    if coverage >= 0.65 or iter_count + 1 >= max_iters:
        return "synthesize"
    return "search_more"


def increment_iter(state: State) -> Dict:
    return {"iter": state.get("iter", 0) + 1}


pf_graph_builder = StateGraph(State)
pf_graph_builder.add_node("search_agent", search_agent)
pf_graph_builder.add_node("tools", ToolNode([
    hybrid_search_papers,
    semantic_search_papers,
    keyword_search_papers,
    vector_search_papers,
]))
pf_graph_builder.add_node("merge_and_rerank", merge_and_rerank)
pf_graph_builder.add_node("increment_iter", increment_iter)

pf_graph_builder.add_edge(START, "search_agent")

pf_graph_builder.add_edge("search_agent", "tools")
pf_graph_builder.add_edge("tools", "merge_and_rerank")

pf_graph_builder.add_conditional_edges("merge_and_rerank", decide_next, {
    "search_more": "increment_iter",
    "synthesize": END,
})
pf_graph_builder.add_edge("increment_iter", "search_agent")

pf_graph = pf_graph_builder.compile()
