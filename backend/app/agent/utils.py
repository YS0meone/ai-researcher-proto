from app.core.config import settings
import os
def get_user_query(messages: list) -> str:
    """Extract the original user query from the message history."""
    user_msg = ""
    for m in messages:
        if hasattr(m, "type") and m.type == "human":
            user_msg = m.content
        elif isinstance(m, dict) and m.get("role") == "user" and m.get("content"):
            user_msg = m["content"]
    return user_msg 
    
def setup_langsmith():
    os.environ["LANGCHAIN_TRACING_V2"] = str(settings.LANGSMITH_TRACING_V2)
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
    