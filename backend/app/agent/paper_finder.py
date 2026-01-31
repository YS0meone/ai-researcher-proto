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
from typing import List, Tuple
from langchain.agents import create_agent

setup_langsmith()

ranker = Reranker("cohere", api_key=os.environ.get("COHERE_API_KEY"))

tools = [tavily_research_overview, s2_search_papers]
tool_node = ToolNode(tools)

MAX_ITER = 3

model = init_chat_model(model=settings.AGENT_MODEL_NAME, api_key=settings.GEMINI_API_KEY)
tool_reasoning_model = model.bind_tools(tools, tool_choice="none")
search_agent_model = model.bind_tools(tools)

def planner(state: PaperFinderState):
    system_prompt = """
    You are a senior researcher. The goal is to create a plan for your research assistant to find the most relevant papers to the user query.
    You are provided with a user query and a list of papers known to the research assistant.
    You need to plan the best way to find the most relevant papers to the user query.
    There are two different methods for your assistant to find papers:
    1. General web search. This is important to understand the general context for the user query and find the most relevant papers from the web.
    2. Access to academic databases. This is helpful to find actual papers with metadata filters and keyword matching.

    Guidelines:
    - Think like a real researcher who would give different plans based on different scenarios:
        For example:
        - If the user query is about a general topic. You might want the assistant to use the general web search and then use the academic databases to find the most relevant papers.
        - If the user query is about a specific paper. You might want the assistant to use the academic database with certain keywords or metadata filters to search for that specific paper
        If not you can use general web search or academic data database Find the anchor paper first
    - The granularity of each step should be adequate for the assistant to finish within one execution.
    Think step by step and provide the reasoning for your plan.
    Limit the number of steps to 2 or less.
    """

    paper_info_text = get_paper_info_text(state.get("papers", []))
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

    try:
        response = structured_model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        return {"plan_steps": response.plan_steps, "plan_reasoning": response.plan_reasoning}
    except Exception as e:
        print(f"Error in planner: {e}")
        # Fallback plan
        return {
            "plan_steps": [
                "Use web search to understand the research topic",
                "Search academic database for relevant papers",
                "Review and refine search results"
            ],
            "plan_reasoning": "Using default plan due to planning error"
        }

def completed_steps_formatter(completed_steps: List[Tuple[str, str]]) -> List[str]:
    return ["\n".join([f"Task:{task}\nResult:{result}" for task, result in completed_steps])]

def replan_agent(state: PaperFinderState):
    system_prompt = """
    You are a senior researcher. The goal is to update a plan for your research assistant to find the most relevant papers to the user query.
    You are provided with a user query, the current plan to retrieve the papers and the steps your assistant has completed.
    You need to first determine if goal is achieved or not. If the goal is achieved, you can stop and mark the goal as achieved.
    If the goal is not achieved, you need to update the plan to find the most relevant papers to the user query and return the new plan.

    There are two different methods for your assistant to find papers:
    1. General web search. This is important to understand the general context for the user query and find the most relevant papers from the web.
    2. Access to academic databases. This is helpful to find actual papers with metadata filters and keyword matching.

    Guidelines:
    - You should first determine if the goal is achieved or not. If the goal is achieved, you can stop and mark the goal as achieved and leave the plan_steps, and plan_reasoning empty. If not you need to update the plan.
    - Think like a real researcher who would give different plans based on different scenarios:
        For example:
        - If the user query is about a general topic. You might want the assistant to use the general web search and then use the academic databases to find the most relevant papers.
        - If the user query is about a specific paper. You might want the assistant to use the academic database with certain keywords or metadata filters to search for that specific paper
        If not you can use general web search or academic data database Find the anchor paper first
    - The granularity of each step should be adequate for the assistant to finish within one execution.
    - Take into account the steps that your assistant has already completed and the results of those steps.
    - If you think the current plan is good enough, you can simply remove the steps that are already completed and keep the rest of the plan.
    - The completed steps should not be included in the new plan.
    Limit the number of new steps to 2 or less.
    Think step by step and provide the reasoning for your plan.
    """

    paper_info_text = get_paper_info_text(state.get("papers", []))
    user_query = state.get("optimized_query", "")
    user_prompt = f"""
    User query: {user_query}
    Papers information:
    {paper_info_text}
    Completed Steps: 
    {completed_steps_formatter(state.get("completed_steps", []))}
    Current Plan: {state.get("plan_steps", [])}
    """

    class ReplanPlan(BaseModel):
        goal_achieved: bool = Field(description="Whether the goal is achieved or not")
        plan_reasoning: str = Field(description="The reasoning for the new plan")
        plan_steps: List[str] = Field(description="The steps of the new plan")

    structured_model = model.with_structured_output(ReplanPlan)

    try:
        response = structured_model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        return {"plan_steps": response.plan_steps, "plan_reasoning": response.plan_reasoning, "goal_achieved": response.goal_achieved}
    except Exception as e:
        print(f"Error in replan_agent: {e}")
        # Continue with existing plan
        current_plan = state.get("plan_steps", [])
        return {
            "plan_steps": current_plan[1:] if len(current_plan) > 1 else ["Search for more relevant papers"],
            "plan_reasoning": "Continuing with adjusted plan due to replanning error",
            "goal_achieved": False
        }

def search_agent(state: PaperFinderState):
    search_query_prompt = """
    You are a senior research assistant who helps finding academic papers based on a user query.
    You are provided with a plan for your search from your mentor.

    Your goal is to utilize the provided tools to finish the current step of the plan.
    You are provided with two methods to find papers:
    1. General web search. This is important to understand the general context for the user query and find the most relevant papers from the web.
    2. Access to academic databases. This is helpful to find actual papers with metadata filters and keyword matching.
    Call the tools that you think are most relevant to the current step of the plan.
    Reflect on the past action, and completed steps and decide what to do next to finish the goal.
    If you think you got the desired results, you can stop and summarize what you found.
    The summarization should be concise and to the point.
    Think step by step.
    """

    iter = state.get("iter", 0)
    plan_steps = state.get("plan_steps", [])
    
    # Handle empty plan_steps
    if not plan_steps:
        current_goal = "Search for relevant papers based on the user query"
    else:
        current_goal = plan_steps[0]

    search_agent = create_agent(model=search_agent_model, tools=tools)

    user_prompt = f"""
    User query: {state.get("optimized_query", "")}
    Current Goal: {current_goal}
    Completed Steps: {state.get("completed_steps", [])}
    """

    response = search_agent.invoke({"messages": [SystemMessage(content=search_query_prompt), HumanMessage(content=user_prompt)]})
    if isinstance(response["messages"][-1].content, list):
        content = " ".join([item["text"] for item in response["messages"][-1].content])
        print(content)
    elif isinstance(response["messages"][-1].content, str):
        content = response["messages"][-1].content
        print(content)
        print(type(content))

    else:
        content = str(response["messages"][-1].content)
        print(type(content))
        print(f"the content is: {content}")
    step_summary = (current_goal, content)
    return {"completed_steps": [step_summary], "iter": iter + 1}


def should_clarify(state: PaperFinderState):
    is_clear = state.get("is_clear", True)
    route = "optimize" if is_clear else "end"
    print(f"[DEBUG] should_clarify: is_clear={is_clear}, routing to '{route}'")
    return route

def should_continue(state: PaperFinderState):
    last = state["messages"][-1]
    return "tools" if hasattr(last, "tool_calls") and last.tool_calls else "end"

def should_reply(state: PaperFinderState):
    goal_achieved = state.get("goal_achieved", False)
    iter = state.get("iter", 0)
    return END if goal_achieved or iter >= MAX_ITER else "search_agent"

paper_finder_builder = StateGraph(PaperFinderState)
paper_finder_builder.add_node("planner", planner)
paper_finder_builder.add_node("replan_agent", replan_agent)
paper_finder_builder.add_node("search_agent", search_agent)
paper_finder_builder.add_node("tools", tool_node)

paper_finder_builder.add_edge(START, "planner")
paper_finder_builder.add_edge("planner", "search_agent")
paper_finder_builder.add_edge("search_agent", "replan_agent")
paper_finder_builder.add_conditional_edges("replan_agent", should_reply)

paper_finder = paper_finder_builder.compile()
