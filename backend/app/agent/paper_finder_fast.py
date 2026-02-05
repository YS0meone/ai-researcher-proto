from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode
from app.tools.search import s2_search_papers, tavily_research_overview, get_paper_details, forward_snowball, backward_snowball
from app.core.config import settings
from app.agent.utils import setup_langsmith
from rerankers import Reranker, Document
from app.db.schema import S2Paper
from pydantic import BaseModel
from typing import List, Annotated
from langchain.agents import AgentState
from langgraph.prebuilt import tools_condition


setup_langsmith()
MAX_PAPER_LIST_LENGTH = 20

model = init_chat_model(model=settings.AGENT_MODEL_NAME, api_key=settings.GEMINI_API_KEY)

tools = [tavily_research_overview, s2_search_papers, get_paper_details, forward_snowball, backward_snowball]
search_agent_model = model.bind_tools(tools)

# Initialize Cohere reranker
if not settings.COHERE_API_KEY:
    print("⚠️  WARNING: COHERE_API_KEY not set in .env. Reranking will be skipped.")
    ranker = None
else:
    try:
        ranker = Reranker("cohere", api_key=settings.COHERE_API_KEY)
        print(f"✅ Cohere reranker initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Cohere reranker: {e}")
        print(f"⚠️  Reranking will be skipped.")
        ranker = None


class ClearItem(BaseModel):
    """sentinel item to clear the list"""
    pass

def new_paper_reducer(current: list, update: list | ClearItem) -> list:
    if isinstance(update, ClearItem):
        return []
    return current + update

class SearchAgentState(AgentState):
    optimized_query: str
    papers: List[S2Paper]
    iter: int
    new_papers: Annotated[List[S2Paper], new_paper_reducer]



def search_agent_node(state: SearchAgentState):
    search_query_prompt = """
    You are a senior research assistant who helps finding academic papers based on a user query.

    Your goal is to utilize the provided tools to help user find the most relevant papers to the user query.
    
    You have access to multiple search methods:
    1. General web search (tavily_research_overview): Use this when the research topic is general or unfamiliar. 
       This helps you understand the research landscape and identify famous/seminal papers you shouldn't miss.
    
    2. Academic database search (s2_search_papers): Search Semantic Scholar's database of 200M+ papers.
       Use keyword queries, filters by year, venue, citation count, etc. to find relevant papers.
    
    3. Citation chasing tools:
       - forward_snowball: Find papers that your seed papers CITE (their references/foundations)
       - backward_snowball: Find papers that CITE your seed papers (recent work building on them)
       Use these when you've found good papers and want to explore their citation network.
    
    4. Paper details (get_paper_details): Check what papers are currently in your paper list.
    
    Strategy tips:
    - Start with web search if topic is unfamiliar to get context
    - Use academic database for targeted searches with filters
    - Use citation chasing to expand from good seed papers you've found
    - Check paper details to avoid redundant searches
    
    Reflect on past actions and completed steps to decide what to do next.
    If you have sufficient results, stop and provide a concise summary of what you found.
    """

    response = search_agent_model.invoke([
        SystemMessage(content=search_query_prompt),
        *state.get("messages", [])
    ])
    return {"messages": [response]}

search_tool_node = ToolNode(tools)

def rerank_node(state: SearchAgentState):
    if len(state.get("new_papers", [])) == 0:
        return {}
    
    existing_papers = state.get("papers", [])
    
    all_papers = list(existing_papers) + list(state.get("new_papers", []))
    unique_papers = {p.paperId: p for p in all_papers}
    deduped_list = list(unique_papers.values())
    
    if len(deduped_list) > 0:
        user_query = state.get("optimized_query", "")
        
        # Skip reranking if no query or no ranker
        if not user_query or not user_query.strip():
            print("⚠️  Skipping rerank: no query provided")
            final_papers = deduped_list[:MAX_PAPER_LIST_LENGTH]
        elif ranker is None:
            print("⚠️  Skipping rerank: COHERE_API_KEY not set")
            final_papers = deduped_list[:MAX_PAPER_LIST_LENGTH]
        else:
            try:
                docs = []
                for paper in deduped_list:
                    # Handle None values in paper data
                    title = paper.title or "No title"
                    abstract = paper.abstract or "No abstract"
                    authors = paper.authors or []
                    content_text = f"Title: {title}\nAbstract: {abstract}\nAuthors: {authors}"
                    docs.append(Document(
                        text=content_text,
                        doc_id=str(paper.paperId),
                        metadata=paper.model_dump()
                    ))
                
                print(f"🔄 Reranking {len(docs)} papers with query: {user_query[:50]}...")
                
                # Debug: Try to catch the actual API response
                try:
                    reranked_results = ranker.rank(query=user_query, docs=docs)
                    top_matches = reranked_results.top_k(k=MAX_PAPER_LIST_LENGTH)
                except KeyError as ke:
                    print(f"🔍 KeyError in reranking: {ke}")
                    print(f"🔍 This suggests Cohere API returned an error response")
                    print(f"🔍 Check your COHERE_API_KEY environment variable")
                    raise  # Re-raise to go to outer except block
                
                final_papers = []
                for match in top_matches:
                    paper_obj = S2Paper.model_validate(match.document.metadata)
                    final_papers.append(paper_obj)
                print(f"✅ Reranking successful: {len(final_papers)} papers")
            except Exception as e:
                print(f"❌ Reranking failed: {type(e).__name__}: {e}")
                print(f"📋 Falling back to original order (top {MAX_PAPER_LIST_LENGTH})")
                final_papers = deduped_list[:MAX_PAPER_LIST_LENGTH]
    else:
        final_papers = []
    
    return {"papers": final_papers, "new_papers": ClearItem(), "iter": state.get("iter", 0) + 1}

def my_tools_condition(state: SearchAgentState):
    if state.get("iter", 0) > 3:
        return "__end__"
    return tools_condition(state)

paper_finder_fast_graph = StateGraph(SearchAgentState)
paper_finder_fast_graph.add_node("search_agent", search_agent_node)
paper_finder_fast_graph.add_node("search_tool", search_tool_node)
paper_finder_fast_graph.add_node("rerank", rerank_node)

paper_finder_fast_graph.add_edge(START, "search_agent")
paper_finder_fast_graph.add_conditional_edges("search_agent", my_tools_condition, {"tools": "search_tool", "__end__": END})
paper_finder_fast_graph.add_edge("search_tool", "rerank")
paper_finder_fast_graph.add_edge("rerank", "search_agent")
paper_finder_fast_graph = paper_finder_fast_graph.compile()
