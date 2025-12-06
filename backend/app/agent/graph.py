from typing import Annotated, List, Dict, Any, Optional, Literal, Sequence
from langchain_deepseek import ChatDeepSeek
from langchain.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from app.core.config import settings
from app.tools.search import (
    hybrid_search_papers,
    semantic_search_papers,
    keyword_search_papers,
    search_papers_by_category,
    search_papers,
    get_paper_details,
    vector_search_papers,
)
from app.agent.prompts import (
    RouterPrompts,
    RerankingPrompts,
    SynthesisPrompts,
    SearchAgentPrompts,
)
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

model = ChatDeepSeek(
    model=settings.MODEL_NAME,
    api_key=settings.OPENAI_API_KEY,
    temperature=0,  # 使用0温度以获得更确定性的输出
    max_retries=3
)

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

class RerankingResult(BaseModel):
    """Structured output for paper reranking."""
    order: List[str]
    coverage_score: float = Field(ge=0.0, le=1.0)
    brief_reasoning: Optional[str] = None

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
    query: Optional[str] = None
    limit: int = 15
    reasoning: Optional[str] = None  # Why this tool for this query

class SearchPlan(BaseModel):
    """Structured output for search planning."""
    tool_calls: List[SearchToolCall] = Field(min_items=1, max_items=5)
    strategy: str  # Overall search strategy explanation

# ============================================================================
# GRAPH STATE
# ============================================================================

class State(BaseModel):
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    papers: List[Dict[str, Any]] = Field(default_factory=list)
    search_queries: List[str] = Field(default_factory=list)
    iter: int = Field(default=0)
    max_iters: int = Field(default=3)
    coverage_score: float = Field(default=0.0)
    route: Optional[Literal["search","synthesize"]] = Field(default=None)

def get_user_query(messages: list) -> str:
    """Extract the original user query from the message history."""
    user_msg = ""
    for m in messages:
        if hasattr(m, "type") and m.type == "human":
            user_msg = m.content
        elif isinstance(m, dict) and m.get("role") == "user" and m.get("content"):
            user_msg = m["content"]
    return user_msg


def router(state: State) -> Dict:
    user_msg = get_user_query(state.messages)
    prompt = RouterPrompts.format_decision(
        user_msg=user_msg,
        num_papers=len(state.papers),
        coverage_score=state.coverage_score,
        search_queries=state.search_queries
    )
    
    structured_model = model.with_structured_output(RouterDecision)
    result = structured_model.invoke([
        SystemMessage(content=RouterPrompts.SYSTEM),
        *state.messages,
        HumanMessage(content=prompt)
    ])
    
    return {"route": result.route, "messages": [HumanMessage(content=f"Routing decision: {result.route}. Reason: {result.short_reason}")]}

def search_agent(state: State) -> Dict:
    """
    Unified search agent that generates queries and selects tools.
    Returns ONE AIMessage with MULTIPLE tool_calls to avoid LangGraph tool response mismatch.
    """
    user_msg = get_user_query(state.messages)
    
    # Get search plan with structured output
    prompt = SearchAgentPrompts.format_planning(
        user_msg=user_msg,
        search_queries=state.search_queries
    )
    
    structured_model = model.with_structured_output(SearchPlan)
    try:
        plan = structured_model.invoke([
            SystemMessage(content=SearchAgentPrompts.SYSTEM),
            *state.messages,
            HumanMessage(content=prompt)
        ])
    except Exception as e:
        logger.error(f"Error generating search plan: {e}")
        # Fallback to simple hybrid search
        plan = SearchPlan(
            tool_calls=[SearchToolCall(
                tool_name="hybrid_search_papers",
                query=user_msg,
                limit=15,
                reasoning="Fallback to basic hybrid search"
            )],
            strategy="Using fallback hybrid search due to planning error"
        )
    
    # Filter out invalid tool calls (missing query)
    valid_tool_calls = [tc for tc in plan.tool_calls if tc.query and tc.query.strip()]
    
    if not valid_tool_calls:
        # If no valid tool calls, use the user query directly
        valid_tool_calls = [SearchToolCall(
            tool_name="hybrid_search_papers",
            query=user_msg,
            limit=15,
            reasoning="Using user query directly"
        )]
    
    # Limit to max 3 tool calls to avoid overwhelming the system
    valid_tool_calls = valid_tool_calls[:3]
    
    # Build tool instructions from the plan
    tool_instructions = f"Execute the following search strategy: {plan.strategy}\n\nSearches to perform:\n"
    for tc in valid_tool_calls:
        tool_instructions += f"- {tc.tool_name}(query='{tc.query}', limit={tc.limit}) - {tc.reasoning or 'N/A'}\n"
    
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
        SystemMessage(content="You are a search execution agent. Execute the planned searches by calling the appropriate tools."),
        HumanMessage(content=tool_instructions)
    ])
    
    # Extract queries for state tracking
    queries = [tc.query for tc in valid_tool_calls if tc.query]
    seen = set(q.lower() for q in state.search_queries)
    merged = state.search_queries + [q for q in queries if q and q.lower() not in seen]
    
    return {
        "search_queries": merged[:6],
        "messages": [ai_message]  # Single AIMessage with multiple tool_calls!
    }

def merge_and_rerank(state: State) -> Dict:
    # Collect the last tool outputs from messages
    import sys
    import json
    candidates: List[Dict[str, Any]] = []
    
    for idx, m in enumerate(state.messages):
        msg_type = getattr(m, 'type', None) or (m.get('type') if isinstance(m, dict) else None)
        msg_class = m.__class__.__name__
        
        # Check if this is a tool message
        is_tool = (isinstance(m, dict) and m.get("type") == "tool") or (hasattr(m, "type") and m.type == "tool") or msg_class == "ToolMessage"
        if is_tool:
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
            
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
    
    print(f"Total candidates found: {len(candidates)}", file=sys.stderr)
    print(f"Existing papers in state: {len(state.papers)}", file=sys.stderr)

    # Deduplicate by arxiv_id and keep best score
    seen = {p["arxiv_id"]: p for p in state.papers}
    for c in candidates:
        aid = c["arxiv_id"]
        prev = seen.get(aid)
        if not prev:
            seen[aid] = c
        else:
            def best(a, b, key):
                sa, sb = a.get(key), b.get(key)
                if sa is None: return b
                if sb is None: return a
                return a if sa >= sb else b
            seen[aid] = best(prev, c, "search_score" if "search_score" in c else ("similarity_score" if "similarity_score" in c else "text_score"))

    merged = list(seen.values())[:100]
    
    print(f"Merged papers count: {len(merged)}", file=sys.stderr)

    # Rerank with LLM (title+abstract snippet) and compute coverage heuristic
    user_msg = get_user_query(state.messages)
    short_list = merged[:30]
    
    print(f"Short list for reranking: {len(short_list)} papers", file=sys.stderr)
    
    if len(short_list) == 0:
        print("WARNING: No papers to rerank! Returning empty results.", file=sys.stderr)
        return {"papers": [], "coverage_score": 0.0}
    
    prompt = RerankingPrompts.format_reranking(
        user_msg=user_msg,
        candidates=short_list
    )
    
    structured_model = model.with_structured_output(RerankingResult)
    result = structured_model.invoke([
        SystemMessage(content=RerankingPrompts.SYSTEM),
        *state.messages,
        HumanMessage(content=prompt)
    ])
    
    order = result.order
    coverage = result.coverage_score
    
    # Fallback if no valid order returned
    if not order:
        print("WARNING: No valid order returned, using default", file=sys.stderr)
        order = [p.get("arxiv_id") for p in short_list]

    id_to_p = {p["arxiv_id"]: p for p in merged}
    reranked = [id_to_p[i] for i in order if i in id_to_p] + [p for p in merged if p["arxiv_id"] not in set(order)]
    
    print(f"Final reranked papers: {len(reranked[:50])}", file=sys.stderr)
    
    return {"papers": reranked[:50], "coverage_score": coverage, "messages": [HumanMessage(content=f"Reranked {len(reranked[:50])} papers, coverage: {coverage:.2f}")]}

def decide_next(state: State):
    """Decide whether to continue searching or synthesize results."""
    if state.coverage_score >= 0.65 or state.iter + 1 >= state.max_iters:
        return "synthesize"
    return "search_more"

def increment_iter(state: State) -> Dict:
    return {"iter": state.iter + 1}

def synthesize(state: State) -> Dict:
    import sys
    user_msg = get_user_query(state.messages)
    
    # First, determine if we need deep content search
    decision_prompt = SynthesisPrompts.format_decision(
        user_msg=user_msg,
        num_papers=len(state.papers)
    )
    
    structured_model = model.with_structured_output(SynthesisDecision)
    decision = structured_model.invoke([
        SystemMessage(content=SynthesisPrompts.DECISION_SYSTEM),
        *state.messages,
        HumanMessage(content=decision_prompt)
    ])
    
    needs_deep_search = decision.needs_deep_search
    search_query = decision.search_query or user_msg
    print(f"Synthesize decision: needs_deep_search={needs_deep_search}, reasoning={decision.reasoning}", file=sys.stderr)
    
    # If deep search is needed, call vector_search_papers
    detailed_segments = []
    if needs_deep_search and state.papers:
        print(f"Performing vector search with query: {search_query}", file=sys.stderr)
        try:
            # Call vector_search_papers tool directly
            results = vector_search_papers.invoke({"query": search_query, "limit": 5})
            if isinstance(results, list) and results:
                detailed_segments = results
                print(f"Vector search returned {len(detailed_segments)} results with detailed segments", file=sys.stderr)
        except Exception as e:
            print(f"Vector search failed: {e}", file=sys.stderr)
    
    # Build paper details from state
    paper_details = []
    for p in state.papers[:10]:
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
        *state.messages,
        HumanMessage(content=answer_prompt)
    ])
    return {"messages": [final]}

# Build graph
graph_builder = StateGraph(State)
graph_builder.add_node("router", router)
graph_builder.add_node("search_agent", search_agent)
graph_builder.add_node("tools", ToolNode([
    hybrid_search_papers,
    semantic_search_papers,
    keyword_search_papers,
    search_papers_by_category,
    search_papers,
    get_paper_details,
    vector_search_papers,
]))
graph_builder.add_node("merge_and_rerank", merge_and_rerank)
graph_builder.add_node("increment_iter", increment_iter)
graph_builder.add_node("synthesize", synthesize)

graph_builder.add_edge(START, "router")

def route_branch(state: State):
    return "search" if state.route == "search" else "synthesize"

graph_builder.add_conditional_edges("router", route_branch, {
    "search": "search_agent",
    "synthesize": "synthesize",
})

# Search loop
graph_builder.add_edge("search_agent", "tools")
graph_builder.add_edge("tools", "merge_and_rerank")

graph_builder.add_conditional_edges("merge_and_rerank", decide_next, {
    "search_more": "increment_iter",
    "synthesize": "synthesize",
})
graph_builder.add_edge("increment_iter", "search_agent")

graph_builder.add_edge("synthesize", END)
graph = graph_builder.compile()