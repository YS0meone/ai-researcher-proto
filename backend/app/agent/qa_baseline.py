from langgraph.graph import StateGraph
from langgraph.graph import END
from typing import Annotated, List, TypedDict
from langgraph.graph.message import add_messages
import pandas as pd
from pathlib import Path
from pprint import pprint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.core.config import settings
from langchain.chat_models import init_chat_model
from pydantic import BaseModel
from pydantic import Field

qa_baseline_model = init_chat_model(model=settings.QA_BASELINE_MODEL_NAME)

train_df = pd.read_parquet(Path(__file__).parent.parent.parent / "eval" / "data" / "train.parquet")
test_df = pd.read_parquet(Path(__file__).parent.parent.parent / "eval" / "data" / "test.parquet")
validation_df = pd.read_parquet(Path(__file__).parent.parent.parent / "eval" / "data" / "validation.parquet")
papers_df = pd.concat([train_df, test_df, validation_df])


class BaseLineState(TypedDict):
    messages: Annotated[list, add_messages]
    selected_ids: List[str]
    reasoning: str
    abstracts: list[str]

def to_full_text(sections: list[str], paragraphs: list[list[str]]) -> str:
    if len(sections) != len(paragraphs):
        flattened_paragraphs = [para for section in paragraphs for para in section]
        return "\n".join(flattened_paragraphs)
    return "\n\n".join([f"{section}\n\n{'\n'.join(paragraph)}" for section, paragraph in zip(sections, paragraphs)])

def get_papers_info(ids: List[str]) -> list[str]:
    papers_info = {}
    for id in ids:
        title = papers_df[papers_df['id'] == id]['title'].values.tolist()
        abstract = papers_df[papers_df['id'] == id]['abstract'].values.tolist()
        paragraphs = papers_df[papers_df['id'] == id]['full_text'].values.tolist()
        papers_info[id] = {
            'title': title,
            'abstract': abstract,
            'full_text': to_full_text(paragraphs[0]['section_name'], paragraphs[0]['paragraphs'])
        }
    return papers_info

def qa_baseline(state: BaseLineState) -> BaseLineState:
    papers_info = get_papers_info(state["selected_ids"])
    papers_info_text = "\n".join([f"Paper {id}: {info['title']}\nAbstract: {info['abstract']}\nFull Text: {info['full_text']}" for id, info in papers_info.items()])
    BASELINE_PROMPT = f"""
    You are an expert in answering question about scientific papers.
    You will be given a user question and a list of papers and their information.
    
    Goal: 
    - Answer the user question based on the papers information.
    - Give the reasoning for your answer.

    General Strategy:
    - You should think step by step and reason about the user question and the papers information.
    - The user question is usually based on the papers information.
    - You should first understand the user's question based on the shown paper abstracts.
    - The answer should be concise and based on the papers information.
    - Some answers can be directly found in the papers information, you should directly answer the question based on the papers information.
    - If the answer is not found in the papers information, you should provide a concise answer based on the papers information.
    - If the answer is not answerable, you should say that the answer is not found in the papers information.
    """

    USER_PROMPT = f"""
    User question: {state["messages"]}
    Papers information:
    {papers_info_text}
    """

    class BaseLineOutput(BaseModel):
        reasoning: str = Field(description="The reasoning for your answer")
        answer: str = Field(description="The answer to the user question")

    structured_model = qa_baseline_model.with_structured_output(BaseLineOutput)
    response = structured_model.invoke([
        SystemMessage(content=BASELINE_PROMPT),
        HumanMessage(content=USER_PROMPT)
    ])
    return {
        "messages": [AIMessage(content=response.answer)],
        "reasoning": response.reasoning,
        "abstracts": [info['abstract'] for info in papers_info.values()]
    }

qa_baseline_builder = StateGraph(BaseLineState)

qa_baseline_builder.add_node("qa_baseline", qa_baseline)

qa_baseline_builder.set_entry_point("qa_baseline")

qa_baseline_builder.add_edge("qa_baseline", END)

qa_baseline = qa_baseline_builder.compile()
