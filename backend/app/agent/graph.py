from typing import Annotated, List, Dict, Any, Optional, Literal
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langchain.schema import SystemMessage
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
    get_paper_details,  # new
)

model = init_chat_model(model="gpt-4o-mini", api_key=settings.OPENAI_API_KEY)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    papers: List[Dict[str, Any]]
    search_queries: List[str]
    iter: int
    max_iters: int
    coverage_score: float
    route: Optional[Literal["search","synthesize"]]

BASE_SYS = "You are an AI research assistant. Think step-by-step, minimize cost, and avoid redundant searches."

def filter_tool_messages(messages: list) -> list:
    """
    Filter out tool-related messages to avoid OpenAI API constraint violation.
    OpenAI requires that assistant messages with tool_calls be immediately followed
    by tool response messages, so we filter these out when making new LLM calls.
    """
    clean_messages = []
    for m in messages:
        # Skip tool response messages (check both 'type' and 'role' fields)
        if isinstance(m, dict):
            if m.get("type") == "tool" or m.get("role") == "tool":
                continue
            # Skip assistant messages with tool_calls
            if m.get("tool_calls"):
                continue
        # Skip assistant messages with tool_calls (LangChain message objects)
        if hasattr(m, "type") and m.type == "tool":
            continue
        if hasattr(m, "tool_calls") and m.tool_calls:
            continue
        # Additional check: for ToolMessage class
        if m.__class__.__name__ == "ToolMessage":
            continue
        clean_messages.append(m)
    
    # Debug: print message types being kept
    import sys
    for i, msg in enumerate(clean_messages):
        msg_type = getattr(msg, 'type', None) or (msg.get('type') if isinstance(msg, dict) else None)
        msg_role = getattr(msg, 'role', None) or (msg.get('role') if isinstance(msg, dict) else None)
        has_tool_calls = bool(getattr(msg, 'tool_calls', None) or (msg.get('tool_calls') if isinstance(msg, dict) else None))
        # print(f"DEBUG: Message {i}: type={msg_type}, role={msg_role}, has_tool_calls={has_tool_calls}, class={msg.__class__.__name__}", file=sys.stderr)
    
    return clean_messages

def get_user_query(messages: list) -> str:
    """Extract the original user query from the message history."""
    user_msg = ""
    for m in messages:
        if hasattr(m, "type") and m.type == "human":
            user_msg = m.content
        elif isinstance(m, dict) and m.get("role") == "user" and m.get("content"):
            user_msg = m["content"]
    return user_msg

def init_state(state: State) -> State:
    msgs = state["messages"]
    if not msgs or not isinstance(msgs[0], SystemMessage):
        msgs = [SystemMessage(content=BASE_SYS)] + msgs
    # Defaults
    state.setdefault("papers", [])
    state.setdefault("search_queries", [])
    state.setdefault("iter", 0)
    state.setdefault("max_iters", 3)
    state.setdefault("coverage_score", 0.0)
    state["messages"] = msgs
    return state

def router(state: State) -> Dict:
    user_msg = get_user_query(state["messages"])
    prompt = f"""
Decide whether to SEARCH or SYNTHESIZE now.
Consider: if the user asks for facts grounded in papers or unknown coverage, choose SEARCH.
Return JSON with fields: route ("search"|"synthesize"), short_reason.
User: {user_msg}
"""
    clean_messages = filter_tool_messages(state["messages"])
    out = model.invoke([*clean_messages, {"role":"user","content":prompt}])
    route = "search" if "search" in str(out.content).lower() else "synthesize"
    return {"route": route, "messages": [out]}

def generate_queries(state: State) -> Dict:
    user_msg = get_user_query(state["messages"])
    prompt = f"""
Generate 2-4 diversified search queries for tools (hybrid/semantic/keyword/category) for the user intent.
Return as JSON list 'queries'. Avoid duplicates and overly generic terms.
User: {user_msg}
Current queries: {state.get("search_queries")}
"""
    clean_messages = filter_tool_messages(state["messages"])
    out = model.invoke([SystemMessage(content=BASE_SYS), *clean_messages, {"role":"user","content":prompt}])
    # naive parse; improve with structured output parser if desired
    import json, re
    text = str(out.content)
    try:
        queries = json.loads(re.search(r"\[.*\]", text, re.S).group(0))
    except Exception:
        queries = []
        for line in text.splitlines():
            if line.strip() and len(queries) < 4:
                queries.append(line.strip("- ").strip())
    # merge & dedupe
    seen = set(q.lower() for q in state["search_queries"])
    merged = state["search_queries"] + [q for q in queries if q and q.lower() not in seen]
    return {"search_queries": merged[:6], "messages": [out]}

search_tools = ToolNode([
    hybrid_search_papers,
    semantic_search_papers,
    keyword_search_papers,
    search_papers_by_category,
    search_papers,
])

def call_search_tools(state: State) -> Dict:
    # Let the model pick tools per query sequentially
    # In each round, we nudge it to choose the best tool and produce a single tool call
    queries = state["search_queries"][-2:] or state["search_queries"]
    tool_bound = model.bind_tools([
        hybrid_search_papers,
        semantic_search_papers,
        keyword_search_papers,
        search_papers_by_category,
        search_papers,
    ])
    all_results: List[Dict[str, Any]] = []
    tool_msgs = []
    clean_messages = filter_tool_messages(state["messages"])
    for q in queries:
        choose = f"""
Choose ONE best tool for this query and call it. Query: "{q}"
Tool guidance: hybrid (default), semantic (conceptual), keyword (exact names/phrases), category (browse).
Return only the tool call.
"""
        m = tool_bound.invoke([SystemMessage(content=BASE_SYS), *clean_messages, {"role":"user","content":choose}])
        tool_msgs.append(m)
        if not m.tool_calls:
            continue
        # Let ToolNode execute; emulate minimal execution here by returning the tool-calls
        # LangGraph will route to ToolNode next
    return {"messages": tool_msgs}

def merge_and_rerank(state: State) -> Dict:
    # Collect the last tool outputs from messages
    import sys
    candidates: List[Dict[str, Any]] = []
    
    # print(f"\n=== MERGE_AND_RERANK DEBUG ===", file=sys.stderr)
    # print(f"Total messages in state: {len(state['messages'])}", file=sys.stderr)
    
    for idx, m in enumerate(state["messages"]):
        msg_type = getattr(m, 'type', None) or (m.get('type') if isinstance(m, dict) else None)
        msg_class = m.__class__.__name__
        # print(f"Message {idx}: type={msg_type}, class={msg_class}", file=sys.stderr)
        
        # Check if this is a tool message
        is_tool = (isinstance(m, dict) and m.get("type") == "tool") or (hasattr(m, "type") and m.type == "tool") or msg_class == "ToolMessage"
        if is_tool:
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
            # print(f"  -> TOOL MESSAGE found! Content type: {type(content)}", file=sys.stderr)
            # print(f"  -> Content preview: {str(content)[:200]}", file=sys.stderr)
            
            # Try to extract results
            if isinstance(content, list):
                # print(f"  -> Content is a list with {len(content)} items", file=sys.stderr)
                for item in content:
                    if isinstance(item, dict) and item.get("arxiv_id"):
                        candidates.append(item)
                        # print(f"  -> Added paper: {item.get('arxiv_id')}", file=sys.stderr)
            elif isinstance(content, str):
                # Tool content might be JSON string
                try:
                    import json
                    parsed = json.loads(content)
                    if isinstance(parsed, list):
                        # print(f"  -> Parsed JSON list with {len(parsed)} items", file=sys.stderr)
                        for item in parsed:
                            if isinstance(item, dict) and item.get("arxiv_id"):
                                candidates.append(item)
                                # print(f"  -> Added paper: {item.get('arxiv_id')}", file=sys.stderr)
                except:
                    pass
    
    print(f"Total candidates found: {len(candidates)}", file=sys.stderr)
    print(f"Existing papers in state: {len(state['papers'])}", file=sys.stderr)

    # Deduplicate by arxiv_id and keep best score
    seen = {p["arxiv_id"]: p for p in state["papers"]}
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
    user_msg = get_user_query(state["messages"])
    short_list = merged[:30]
    
    print(f"Short list for reranking: {len(short_list)} papers", file=sys.stderr)
    
    if len(short_list) == 0:
        print("WARNING: No papers to rerank! Returning empty results.", file=sys.stderr)
        return {"papers": [], "coverage_score": 0.0}
    
    prompt = f"""
Rerank these candidate papers for answering the user query.
Return JSON with fields:
- order: list of arxiv_id sorted by descending relevance
- coverage_score: float between 0 and 1 indicating how sufficient the top-10 are to answer.

User: {user_msg}
Candidates (id, title, abstract<=400):
{[{ 'id': p.get('arxiv_id'), 'title': p.get('title'), 'abstract': (p.get('abstract','') or '')[:400] } for p in short_list]}
"""
    clean_messages = filter_tool_messages(state["messages"])
    out = model.invoke([SystemMessage(content=BASE_SYS), *clean_messages, {"role":"user","content":prompt}])
    
    # print(f"LLM response: {str(out.content)[:500]}", file=sys.stderr)
    
    import json, re
    order: List[str] = []
    coverage = 0.0
    try:
        j = json.loads(re.search(r"\{.*\}", str(out.content), re.S).group(0))
        order = j.get("order", []) or []
        coverage = float(j.get("coverage_score", 0))
        # print(f"Parsed order: {len(order)} papers, coverage: {coverage}", file=sys.stderr)
    except Exception as e:
        print(f"Failed to parse LLM response: {e}", file=sys.stderr)
        order = [p.get("arxiv_id") for p in short_list]
        print(f"Using default order with {len(order)} papers", file=sys.stderr)

    id_to_p = {p["arxiv_id"]: p for p in merged}
    reranked = [id_to_p[i] for i in order if i in id_to_p] + [p for p in merged if p["arxiv_id"] not in set(order)]
    
    print(f"Final reranked papers: {len(reranked[:50])}", file=sys.stderr)
    print(f"=== END MERGE_AND_RERANK DEBUG ===\n", file=sys.stderr)
    
    return {"papers": reranked[:50], "coverage_score": coverage, "messages": [out]}

def decide_next(state: State):
    if state["coverage_score"] >= 0.65 or state["iter"] + 1 >= state["max_iters"]:
        return "synthesize"
    return "search_more"

def increment_iter(state: State) -> Dict:
    return {"iter": state["iter"] + 1}

def synthesize(state: State) -> Dict:
    # Build paper details directly from state instead of using tool calls
    paper_details = []
    for p in state["papers"][:10]:
        paper_details.append({
            "arxiv_id": p.get("arxiv_id"),
            "title": p.get("title"),
            "abstract": p.get("abstract"),
            "authors": p.get("authors"),
            "categories": p.get("categories"),
        })
    
    # Produce final grounded answer
    answer_prompt = f"""
Write a concise, well-structured answer grounded in the provided papers. Cite arXiv IDs inline like [arXiv:XXXX.XXXXX].
If evidence is weak, state limitations and suggest follow-ups.

Top relevant papers:
{paper_details}
"""
    clean_messages = filter_tool_messages(state["messages"])
    final = model.invoke([SystemMessage(content=BASE_SYS), *clean_messages, {"role":"user","content": answer_prompt}])
    return {"messages": [final]}

# Build graph
graph_builder = StateGraph(State)
graph_builder.add_node("init", init_state)
graph_builder.add_node("router", router)
graph_builder.add_node("generate_queries", generate_queries)
graph_builder.add_node("call_search_tools", call_search_tools)
graph_builder.add_node("tools", ToolNode([
    hybrid_search_papers,
    semantic_search_papers,
    keyword_search_papers,
    search_papers_by_category,
    search_papers,
    get_paper_details,
]))
graph_builder.add_node("merge_and_rerank", merge_and_rerank)
graph_builder.add_node("increment_iter", increment_iter)
graph_builder.add_node("synthesize", synthesize)

graph_builder.add_edge(START, "init")
graph_builder.add_edge("init", "router")

def route_branch(state: State):
    return "search" if state.get("route") == "search" else "synthesize"

graph_builder.add_conditional_edges("router", route_branch, {
    "search": "generate_queries",
    "synthesize": "synthesize",
})

# Search loop
graph_builder.add_edge("generate_queries", "call_search_tools")
graph_builder.add_edge("call_search_tools", "tools")
graph_builder.add_edge("tools", "merge_and_rerank")

graph_builder.add_conditional_edges("merge_and_rerank", decide_next, {
    "search_more": "increment_iter",
    "synthesize": "synthesize",
})
graph_builder.add_edge("increment_iter", "generate_queries")

graph_builder.add_edge("synthesize", END)
graph = graph_builder.compile()