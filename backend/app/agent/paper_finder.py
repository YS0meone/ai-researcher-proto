from langgraph.graph import START, END, StateGraph
from app.agent.states import State
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from app.tools.search import s2_search_papers
import os
from app.core.config import settings
from app.agent.utils import setup_langsmith, get_user_query
from langchain_core.messages import ToolMessage
from rerankers import Reranker, Document
from app.db.schema import S2Paper

setup_langsmith()

ranker = Reranker("cohere", api_key=os.environ.get("COHERE_API_KEY"))

tools = [s2_search_papers]
tool_node = ToolNode(tools)


def agent(state: State):
    search_agent_prompt = f"""
    You are a search agent for academic papers.
    """
    model = ChatOpenAI(model=settings.AGENT_MODEL_NAME, api_key=settings.OPENAI_API_KEY).bind_tools(tools)
    return {"messages": [model.invoke(state["messages"])]}


def rerank(state: State):
    raw_papers = []
    
    for message in reversed(state["messages"]):
        if isinstance(message, ToolMessage):
            if message.artifact:
                raw_papers.extend(message.artifact)
        else:
            break

    if not raw_papers:
        return {"papers": []}

    unique_papers = {p.paper_id: p for p in raw_papers}
    deduped_list = list(unique_papers.values())

    docs = []
    for paper in deduped_list:
        content_text = f"Title: {paper.title}\nAbstract: {paper.abstract}\nAuthors: {paper.authors}"
        
        docs.append(Document(
            text=content_text, 
            doc_id=str(paper.paper_id), 
            metadata=paper.model_dump()
        ))

    user_query = get_user_query(state["messages"])

    try:
        reranked_results = ranker.rank(query=user_query, docs=docs)
    except Exception as e:
        print(f"Reranking failed: {e}")
        return {"papers": deduped_list[:10]}

    top_matches = reranked_results.top_k(k=10)
    
    final_papers = []
    for match in top_matches:
        paper_obj = S2Paper.model_validate(match.document.metadata)
        final_papers.append(paper_obj)

    return {"papers": final_papers}

def should_continue(state: State):
    last = state["messages"][-1]
    return "tools" if hasattr(last, "tool_calls") and last.tool_calls else "end"


paper_finder_builder = StateGraph(State)
paper_finder_builder.add_node("agent", agent)
paper_finder_builder.add_node("tools", tool_node)
paper_finder_builder.add_node("rerank", rerank)
paper_finder_builder.add_edge(START, "agent")
paper_finder_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
paper_finder_builder.add_edge("tools", "rerank")
paper_finder_builder.add_edge("rerank", "agent")
paper_finder = paper_finder_builder.compile()

