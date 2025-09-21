from typing import Annotated
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langchain.schema import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from app.tools.search import (
    search_papers, 
    hybrid_search_papers, 
    semantic_search_papers, 
    keyword_search_papers,
    search_papers_by_category
)

from app.core.config import settings

model = init_chat_model(model="gpt-4o-mini", api_key=settings.OPENAI_API_KEY)
model = model.bind_tools([
    hybrid_search_papers,  # Primary search tool
    semantic_search_papers,  # For conceptual searches
    keyword_search_papers,   # For exact term matching
    search_papers_by_category,  # For category browsing
    search_papers  # Keep the original for backward compatibility
])

SYSTEM_PROMPT = """You are an AI research assistant specialized in academic paper search and analysis, with access to a comprehensive database of ArXiv papers.

## Your Advanced Search Capabilities:

### 🔍 **hybrid_search_papers** (PRIMARY TOOL - Use this most often)
- Combines keyword matching + semantic similarity for best results
- Perfect for comprehensive research on any topic
- Use for: General research questions, finding papers on specific topics
- Example: "machine learning interpretability", "transformer attention mechanisms"

### 🧠 **semantic_search_papers** (Conceptual Discovery)
- Finds papers by meaning/concepts, not just keywords
- Great for discovering related work and novel approaches
- Use when: Exploring concepts, finding similar methodologies, research inspiration
- Can search "title" or "abstract" fields specifically

### 🎯 **keyword_search_papers** (Exact Matching)
- Traditional keyword/phrase search with highlighting
- Best for: Author searches, specific terms, exact paper titles
- Use when: Looking for specific researchers, exact methodologies, precise terms

### 📚 **search_papers_by_category** (Domain Exploration)
- Browse papers by ArXiv categories (cs.CL, cs.AI, cs.LG, etc.)
- Perfect for: Domain exploration, staying current in specific fields
- Categories: cs.CL (NLP), cs.AI (AI), cs.LG (ML), cs.CV (Vision), etc.

## Smart Search Strategy:

**For Research Questions:**
1. Start with `hybrid_search_papers` for comprehensive coverage
2. Use `semantic_search_papers` to find conceptually related work
3. Use `keyword_search_papers` for specific authors or exact terms

**Category Filtering:**
- Always consider using category filters: "cs.CL,cs.AI" for NLP+AI papers
- Common categories: cs.CL, cs.AI, cs.LG, cs.CV, cs.RO, stat.ML

**Response Guidelines:**
- Cite papers with: Title, Authors, ArXiv ID, and URL
- Summarize key findings and methodologies
- Highlight relevance scores when comparing results
- Suggest follow-up searches for deeper exploration
- Always provide ArXiv URLs for easy access

Be thorough, insightful, and help users discover the most relevant research for their needs."""

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

# Define the tool node with all available search tools
tools = ToolNode([
    hybrid_search_papers,
    semantic_search_papers, 
    keyword_search_papers,
    search_papers_by_category,
    search_papers
])

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tools)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", should_continue)
graph_builder.add_edge("tools", "chatbot")

graph = graph_builder.compile()