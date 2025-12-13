from tqdm import tqdm
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
from langsmith import Client
from app.agent.utils import setup_langsmith

setup_langsmith()
client = Client()

dataset_name = "qasper-qa-e2e"

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
    for _, row in tqdm(papers_df.iterrows(), total=len(papers_df)):
        d = row.to_dict()
        paper = convert_to_ArxivPaper(d)
        flattened_paragraphs = [para for section in d["full_text"]["paragraphs"] for para in section]
        es_service.add_paper(paper)
        qdrant_service.add_paper_with_chunks(paper, flattened_paragraphs)

def create_examples(dataset_id: str, paper_id: str, raw_data: QasperPaper) -> bool:
    questions = raw_data["qas"]["question"]
    answers = raw_data["qas"]["answers"]
    for question, answer in tqdm(zip(questions, answers), total=len(questions)):
        ground_truth_answer = None
        ground_truth_evidence = None
        answer_data = answer["answer"][0]
        answer_type = None
        # print("question: ", question)
        # print("unanswerable: ", answer_data["unanswerable"], type(answer_data["unanswerable"]))
        # print("yes_no: ", answer_data["yes_no"], type(answer_data["yes_no"]))
        # print("free_form_answer: ", answer_data["free_form_answer"], type(answer_data["free_form_answer"]))
        # print("evidence: ", answer_data["evidence"], type(answer_data["evidence"]))
        # print("extractive_spans: ", answer_data["extractive_spans"], type(answer_data["extractive_spans"]))
        # print("--------------------------------")
        if answer_data.get("unanswerable", False):
            ground_truth_answer = "Given the provided context, the question is unanswerable."
            answer_type = "unanswerable"
        elif answer_data.get("yes_no", None) is not None:
            ground_truth_answer = answer_data["yes_no"]
            answer_type = "yes_no"
        elif len(answer_data.get("extractive_spans", [])) > 0:
            ground_truth_answer = " ".join(answer_data["extractive_spans"])
            answer_type = "extractive_spans"
        elif answer_data.get("free_form_answer", None) is not None:
            ground_truth_answer = answer_data["free_form_answer"]
            answer_type = "free_form_answer"
        ground_truth_evidence = answer_data["evidence"]
        if answer_type is None:
            print(f"No answer type found for paper {paper_id} and question {question}")
            continue
        if ground_truth_answer and len(ground_truth_evidence) > 0:
            client.create_example(
                dataset_id=dataset_id,
                inputs={"paper_id": paper_id, "question": question},
                outputs={"ground_truth_answer": ground_truth_answer, "ground_truth_evidence": ground_truth_evidence, "answer_type": answer_type}
            )
        else:
            print(f"No ground truth answer or evidence found for paper {paper_id} and question {question}")


def load_qasper_to_langsmith(papers_df: pd.DataFrame, client: Client, limit: int = 15):
    if client.has_dataset(dataset_name=dataset_name):
        dataset = client.delete_dataset(dataset_name=dataset_name)
    dataset = client.create_dataset(dataset_name)
    for _, row in tqdm(papers_df.iterrows(), total=len(papers_df)):
        d = row.to_dict()
        create_examples(dataset.id, d["id"], d)
        limit -= 1
        if limit == 0:
            break
    print("Done loading QASPER to LangSmith")
    
def main():
    train_df = pd.read_parquet(Path(__file__).parent / "data" / "train.parquet")
    test_df = pd.read_parquet(Path(__file__).parent / "data" / "test.parquet")
    validation_df = pd.read_parquet(Path(__file__).parent / "data" / "validation.parquet")
    papers_df = pd.concat([train_df, test_df, validation_df])
    load_qasper_to_langsmith(papers_df, client)

if __name__ == "__main__":
    main()



