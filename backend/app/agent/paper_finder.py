from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage, AIMessage
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
from pydantic import BaseModel, Field
from typing import List

setup_langsmith()

ranker = Reranker("cohere", api_key=os.environ.get("COHERE_API_KEY"))

tools = [s2_search_papers]
tool_node = ToolNode(tools)

model = init_chat_model(model=settings.AGENT_MODEL_NAME, api_key=settings.GEMINI_API_KEY)
tool_reasoning_model = model.bind_tools(tools, tool_choice="none")
tool_model = model.bind_tools(tools)

def query_clarification(state: State):
    system_prompt = f"""
    You are an expert in clarifying user queries for a search agent for academic papers.
    You need to first decide if the user's query is clear. 
    If it is not clear, you need to mark the is_clear field as False and provide a clarification for the user's query. 
    If it is clear, mark the is_clear field as True and leave the clarification field empty.
    Think step by step and provide the reasoning for your decision and the clarification if needed.
    """

    class QueryClarificationOutput(BaseModel):
        reasoning: str = Field(description="The reasoning for your decision and the clarification if needed")
        is_clear: bool = Field(description="Whether the user's query is clear")
        clarification: str = Field(description="The clarification for the user's query")

    structured_model = model.with_structured_output(QueryClarificationOutput)
    response = structured_model.invoke([
        SystemMessage(content=system_prompt)
    ] + state["messages"])
    if response.is_clear:
        return {"messages": [AIMessage(content="The user's query is clear.")], "is_clear": True}
    else:
        return {"messages": [AIMessage(content=response.clarification)], "is_clear": False}

def query_optimization(state: State):
    system_prompt = f"""
    You are an expert in optimizing user queries for a search agent for academic papers.
    Your goals is to rephrase the user query to be more specific and to be more likely to return relevant papers.
    There might be some clarification happened before this node, you should take that into account.
    If the user's query is good enough, you may repeat the user's query as the optimized query or change it slightly to be more specific.
    If the user's query is not good enough, you should optimize it.
    The optimized query should be understandable by a search agent without any additional context.
    Think step by step and provide the reasoning for your optimization.
    """

    class QueryOptimizationOutput(BaseModel):
        reasoning: str = Field(description="The reasoning for your optimization")
        optimized_query: str = Field(description="The optimized query for the user's query")

    structured_model = model.with_structured_output(QueryOptimizationOutput)
    response = structured_model.invoke([
        SystemMessage(content=system_prompt)
    ] + state["messages"])

    message = f"The optimized query is: {response.optimized_query}\n\nReasoning: {response.reasoning}"
    
    return {"messages": [AIMessage(content=message)], "optimized_query": response.optimized_query}

def evaluation(state: State):
    system_prompt = f"""
    You are an expert in evaluating the relevance of papers to a user's query.
    You are provided with a list of papers and a user's query.
    You need to evaluate the relevance of the papers to the user's query.
    Think step by step and provide the reasoning for your evaluation.
    """


def search_agent(state: State):
    search_query_prompt = """
    You are a search agent for academic papers. You are provided with a versatile search tool that can search for papers
    by general query, specific year, venue, fields of study, publication date or year, minimum citation count, and title matching.
    Your role is to generate optimized search queries to search for papers that are most likely to be relevant to the user's query.
    Think step by step and generate 3-4 search queries along with its reasoning. 
    """

    class SearchQuery(BaseModel):
        reasoning: str = Field(description="The reasoning for the search query")
        search_query: str = Field(description="The search query for the user's query")

    class SearchQueryOutput(BaseModel):
        search_queries: List[SearchQuery] = Field(description="The search queries for the user's query")

    structured_model = tool_reasoning_model.with_structured_output(SearchQueryOutput)

    response = structured_model.invoke([
        SystemMessage(content=search_query_prompt),
        HumanMessage(content=f"Generate search queries for this: {state['optimized_query']}")
    ])
    search_queries = [query.search_query for query in response.search_queries]
    search_query_text = "\n".join([f"Search query {i+1}: {query}" for i, query in enumerate(search_queries)])

    tool_prompt = """
    You are a search agent for academic papers. You are provided with a versatile search tool that can search for papers.
    Generate tool call messages based on the given search queries:
    """
    
    response = tool_model.invoke([
        SystemMessage(content=tool_prompt),
        HumanMessage(content=search_query_text)
    ])
    return {"messages": [response]}

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

    unique_papers = {p.paperId: p for p in raw_papers}
    deduped_list = list(unique_papers.values())

    docs = []
    for paper in deduped_list:
        content_text = f"Title: {paper.title}\nAbstract: {paper.abstract}\nAuthors: {paper.authors}"
        
        docs.append(Document(
            text=content_text, 
            doc_id=str(paper.paperId), 
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

def should_clarify(state: State):
    is_clear = state.get("is_clear", True)
    route = "optimize" if is_clear else "end"
    print(f"[DEBUG] should_clarify: is_clear={is_clear}, routing to '{route}'")
    return route

def should_continue(state: State):
    last = state["messages"][-1]
    return "tools" if hasattr(last, "tool_calls") and last.tool_calls else "end"


paper_finder_builder = StateGraph(State)

paper_finder_builder.add_node("query_clarification", query_clarification)
paper_finder_builder.add_node("query_optimization", query_optimization)
paper_finder_builder.add_node("search_agent", search_agent)
paper_finder_builder.add_node("tools", tool_node)
paper_finder_builder.add_node("rerank", rerank)

paper_finder_builder.add_edge(START, "query_clarification")
paper_finder_builder.add_conditional_edges("query_clarification", should_clarify, {"optimize": "query_optimization", "end": END})
paper_finder_builder.add_edge("query_optimization", "search_agent")
paper_finder_builder.add_conditional_edges("search_agent", should_continue, {"tools": "tools", "end": END})
paper_finder_builder.add_edge("tools", "rerank")
paper_finder_builder.add_edge("rerank", END)

paper_finder = paper_finder_builder.compile()

