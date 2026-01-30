from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import START, END, StateGraph
from app.agent.states import State, PaperFinderState
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from app.tools.search import s2_search_papers, tavily_research_overview
import os
from app.core.config import settings
from app.agent.utils import setup_langsmith, get_user_query, get_paper_info_text
from langchain_core.messages import ToolMessage
from rerankers import Reranker, Document
from app.db.schema import S2Paper
from pydantic import BaseModel, Field
from typing import List

setup_langsmith()

ranker = Reranker("cohere", api_key=os.environ.get("COHERE_API_KEY"))

tools = [tavily_research_overview, s2_search_papers]
tool_node = ToolNode(tools)

model = init_chat_model(model=settings.AGENT_MODEL_NAME, api_key=settings.GEMINI_API_KEY)
tool_reasoning_model = model.bind_tools(tools, tool_choice="none")
search_agent_model = model.bind_tools(tools)

def planner(state: PaperFinderState):
    system_prompt = """
    You are a senior researcher. The goal is to create a plan for your research assistant to find the most relevant papers to the user query.
    You are provided with a user query and a list of papers known to the research assistant.
    You need to plan the best way to find the most relevant papers to the user query.
    There are three different methods for your assistant to find papers:
    1. General web search. This is important to understand the general context for the user query and find the most relevant papers from the web.
    2. Access to academic databases. This is helpful to find actual papers with metadata filters and keyword matching.
    3. Citation chase. The tool that helps the assistant to traverse through the citation network.

    Guidelines:
    - Think like a real researcher who would give different plans based on different scenarios:
        For example:
        - If the user query is about a general topic. You might want the assistant to use the general web search and then use the academic databases to find the most relevant papers.
        - If the user query is about a specific paper. You might want the assistant to use the academic database with certain keywords or metadata filters to search for that specific paper
        - User query is asking about those papers being related to certain anchor paper. You already have that anchor paper in the context. You can just use the citation chase tool to find the relevant papers. 
        If not you can use general web search or academic data database Find the anchor paper first
    - The granularity of each step should be adequate for the assistant to finish within one execution.
    Think step by step and provide the reasoning for your plan.
    """

    paper_info_text = get_paper_info_text(state["papers"])
    user_query = state["optimized_query"]
    user_prompt = f"""
    User query: {user_query}
    Papers information:
    {paper_info_text}
    """
    class Plan(BaseModel):
        plan_reasoning: str = Field(description="The reasoning for the plan")
        plan_steps: List[str] = Field(description="The steps of the plan")

    structured_model = model.with_structured_output(Plan)

    response = structured_model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    return {"plan_steps": response.plan_steps, "plan_reasoning": response.plan_reasoning}


def evaluation(state: PaperFinderState):
    system_prompt = f"""
    You are an expert in evaluating the relevance of papers to a user's query.
    You are provided with a list of papers and a user's query
    You need to evaluate the relevance of the papers to the user's query.
    Think step by step and provide the reasoning for your evaluation.
    """


def search_agent(state: PaperFinderState):
    search_query_prompt = """
    You are a senior research assistant who helps finding academic papers based on a user query.
    You are provided with a plan for your search from your mentor.
    Your goal is to utilize the provided tools to finish the current step of the plan.
    You are provided with three methods to find papers:
    1. General web search. This is important to understand the general context for the user query and find the most relevant papers from the web.
    2. Access to academic databases. This is helpful to find actual papers with metadata filters and keyword matching.
    3. Citation chase. The tool that helps the assistant to traverse through the citation network.
    Call the tools that you think are most relevant to the current step of the plan.
    Think step by step and generate the tool calls. 
    """

    response = search_agent_model.invoke([
        SystemMessage(content=search_query_prompt),
        HumanMessage(content=f"Generate search tool calls for this user query: {state['optimized_query']}")
    ])
    return {"messages": [response]}


def rerank(state: PaperFinderState):
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

def should_clarify(state: PaperFinderState):
    is_clear = state.get("is_clear", True)
    route = "optimize" if is_clear else "end"
    print(f"[DEBUG] should_clarify: is_clear={is_clear}, routing to '{route}'")
    return route

def should_continue(state: PaperFinderState):
    last = state["messages"][-1]
    return "tools" if hasattr(last, "tool_calls") and last.tool_calls else "end"


paper_finder_builder = StateGraph(PaperFinderState)

paper_finder_builder.add_node("search_agent", search_agent)
paper_finder_builder.add_node("tools", tool_node)
paper_finder_builder.add_node("rerank", rerank)

paper_finder_builder.add_edge(START, "search_agent")
paper_finder_builder.add_conditional_edges("search_agent", should_continue, {"tools": "tools", "end": END})
paper_finder_builder.add_edge("tools", "rerank")
paper_finder_builder.add_edge("rerank", END)

paper_finder = paper_finder_builder.compile()
