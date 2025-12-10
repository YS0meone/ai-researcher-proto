from pprint import pprint
from typing import Literal, Optional, TypedDict
from huggingface_hub import login
from app.core.config import settings
from mlcroissant import Dataset
import pandas as pd
from pathlib import Path
from app.db.schema import ArxivPaper
from app.services.elasticsearch import ElasticsearchService
from app.services.qdrant import QdrantService
from app.core.config import settings
from numpy.typing import NDArray


es_service = ElasticsearchService(settings.elasticsearch_config)
qdrant_service = QdrantService(settings.qdrant_config)

class FullText(TypedDict):
    section_name: NDArray[str]
    paragraphs: NDArray[NDArray[str]]

class Answer(TypedDict):
    unanswerable: bool
    extractive_spans: NDArray[str]
    yes_no: Optional[Literal["yes", "no"]]
    free_form_answer: str
    evidence: NDArray[str]
    highlighted_evidence: NDArray[str]
    annotation_id: str
    worker_id: str

class QAS(TypedDict):
    question: NDArray[str]
    question_id: NDArray[str]
    nlp_background: NDArray[str]
    topic_background: NDArray[str]
    paper_read: NDArray[str]
    search_query: NDArray[str]
    question_writer: NDArray[str]
    answers: NDArray[NDArray[Answer]]

class FiguresAndTables(TypedDict):
    caption: NDArray[str]
    file: NDArray[str]

class QasperPaper(TypedDict):
    id: str
    title: str
    abstract: str
    full_text: FullText
    qas: QAS
    figures_and_tables: FiguresAndTables

def convert_to_ArxivPaper(d: QasperPaper) -> ArxivPaper:
    return ArxivPaper(
        id=d["id"],
        title=d["title"],
        abstract=d["abstract"]
    )

def load_qasper_to_db(papers_df: pd.DataFrame) -> ArxivPaper:
    first = True
    for _, row in papers_df.iterrows():
        if first:
            first = False
        else:
            break
        d = row.to_dict()
        paper = convert_to_ArxivPaper(d)
        flattened_paragraphs = [para for section in d["full_text"]["paragraphs"] for para in section]
        es_service.add_paper(paper)
        qdrant_service.add_paper_with_chunks(paper, flattened_paragraphs)
        

def main():
    train_df = pd.read_parquet(Path(__file__).parent / "data" / "train.parquet")
    test_df = pd.read_parquet(Path(__file__).parent / "data" / "test.parquet")
    validation_df = pd.read_parquet(Path(__file__).parent / "data" / "validation.parquet")
    papers_df = pd.concat([train_df, test_df, validation_df])
    load_qasper_to_db(papers_df)

if __name__ == "__main__":
    main()



