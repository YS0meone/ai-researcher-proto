from langchain_core.tools import tool
from app.db.schema import S2Paper
from app.agent.paper_finder import paper_finder
from app.agent.qa import qa_graph
from typing import Tuple
from langgraph.graph import StateGraph
from app.agent.states import State
import logging
from app.core.config import settings
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode, START, END
from app.agent.utils import setup_langsmith
from pydantic import BaseModel, Field
from langchain.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.agents import create_react_agent


setup_langsmith()
logger = logging.getLogger(__name__)

@tool(response_format="content_and_artifact")
def find_papers(user_query: str, papers: list[S2Paper]) -> Tuple[str, list[S2Paper]]:
    """
    Find papers related to the user query.
    Args: 
        user_query: The user query to find papers for.
        papers: The papers to find papers for.
    Returns:
        A tuple containing the number of papers found and the list of papers found.
        A list of papers related to the user query.
    """
    state = {"messages": [{"role": "user", "content": user_query}], "papers": papers}
    result = paper_finder.invoke(state)
    return f"I found {len(result['papers'])} papers for your query.", result["papers"]

@tool
def answer_question(user_query: str, papers: list[S2Paper]) -> str:
    """
    Answer the user question based on the papers.
    Args:
        user_query: The user query to answer.
        papers: The papers to answer the question for.
    Returns:
        The answer to the user question.
    """
    state = {"messages": [{"role": "user", "content": user_query}], "papers": papers}
    result = qa_graph.invoke(state)
    return result["messages"][-1].content


def query_clarification(state: State):
    system_prompt = f"""
    You are an expert in clarifying user queries for a research assistant.
    You need to first decide if the user's query is clear. 
    If it is not clear, you need to mark the is_clear field as False and provide a clarification for the user's query. 
    If it is clear, mark the is_clear field as True and leave the clarification field empty.
    Think step by step and provide the reasoning for your decision and the clarification if needed.
    """

    class QueryClarificationOutput(BaseModel):
        reasoning: str = Field(description="The reasoning for your decision and the clarification if needed")
        is_clear: bool = Field(description="Whether the user's query is clear")
        clarification: str = Field(description="The clarification for the user's query")

    structured_model = supervisor_model.with_structured_output(QueryClarificationOutput)
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
    Your goals is to rephrase the user query to be more specific and to be more likely to help the subagent find the most relevant papers and answer the user's question.
    There might be some clarification happened before this node, you should take that into account.
    If the user's query is good enough, you may repeat the user's query as the optimized query or change it slightly to be more specific.
    If the user's query is not good enough, you should optimize it.
    The optimized query should be self-contained and should not require any additional context.
    Think step by step and provide the reasoning for your optimization.
    """

    class QueryOptimizationOutput(BaseModel):
        reasoning: str = Field(description="The reasoning for your optimization")
        optimized_query: str = Field(description="The optimized query for the user's query")

    structured_model = supervisor_model.with_structured_output(QueryOptimizationOutput)
    response = structured_model.invoke([
        SystemMessage(content=system_prompt)
    ] + state["messages"])

    message = f"The optimized query is: {response.optimized_query}\n\nReasoning: {response.reasoning}"
    
    return {"messages": [AIMessage(content=message)], "optimized_query": response.optimized_query}

def should_clarify(state: State):
    is_clear = state.get("is_clear", True)
    route = "optimize" if is_clear else "end"
    return route

supervisor_model = init_chat_model(model=settings.AGENT_MODEL_NAME, api_key=settings.GEMINI_API_KEY)
tools = [find_papers, answer_question]

executor_prompt = """
You are an expert in executing tasks for a research assistant.
You are provided with two tools: find_papers and answer_question.
The find_papers tool is used to find papers related to the user query.
The answer_question tool is used to answer the user question based on the papers.
You need to execute the task based on the user query and the papers.

Based on different user queries, you need to decide which tool to use or combine them to answer the user question.
If the user query   
You need to use the find_papers tool to find the most relevant papers and then use the answer_question tool to answer the user question.
Think step by step and provide the reasoning for your execution.
"""

executor_node = create_react_agent(supervisor_model, tools, tool_choice="any")


supervisor = StateGraph(State)
supervisor.add_node("query_clarification", query_clarification)
supervisor.add_node("query_optimization", query_optimization)
supervisor.add_node("executor", executor_node)

supervisor.add_edge(START, "query_clarification")
supervisor.add_conditional_edges("query_clarification", should_clarify, {"optimize": "query_optimization", "end": END})
supervisor.add_edge("query_optimization", "executor")
supervisor.add_edge("executor", "tool_node")
supervisor.add_edge("tool_node", END)
supervisor.add_edge(END, END)
supervisor = supervisor.compile()