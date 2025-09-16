from typing import Annotated
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langchain.schema import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from app.tools.search import search_papers

from app.core.config import settings

model = init_chat_model(model="gpt-4o-mini", api_key=settings.OPENAI_API_KEY)
model = model.bind_tools([search_papers])

SYSTEM_PROMPT = """You are an AI research assistant specialized in academic paper search and analysis.

Your capabilities:
- Search through a database of academic papers using full-text search
- Answer questions about research topics, methodologies, and findings
- Provide summaries and insights from academic literature

When users ask about research topics, papers, or academic questions:
1. Use the search_papers tool to find relevant papers
2. Analyze and synthesize the results
3. Provide clear, informative responses with paper references

Be helpful, accurate, and cite the papers you reference."""

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def chatbot(state: State):
    # Add system prompt if this is the first interaction
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    
    return {"messages": [model.invoke(messages)]}

def should_continue(state: State):
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return END

# Define the tool node
tools = ToolNode([search_papers])

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tools)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", should_continue)
graph_builder.add_edge("tools", "chatbot")

graph = graph_builder.compile()