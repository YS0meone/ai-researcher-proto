from pprint import pprint
from huggingface_hub import login
from app.core.config import settings
from mlcroissant import Dataset
import pandas as pd
from pathlib import Path
from app.db.schema import ArxivPaper
from app.services.elasticsearch import ElasticsearchService
from app.services.qdrant import QdrantService
from app.core.config import settings


es_service = ElasticsearchService(settings.elasticsearch_config)
qdrant_service = QdrantService(settings.qdrant_config)

def convert_to_ArxivPaper(d: dict) -> ArxivPaper:
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
        es_service.add_paper(paper)
        print(f"Added paper {paper.id}")
    
def main():
    train_df = pd.read_parquet(Path(__file__).parent / "data" / "train.parquet")
    test_df = pd.read_parquet(Path(__file__).parent / "data" / "test.parquet")
    validation_df = pd.read_parquet(Path(__file__).parent / "data" / "validation.parquet")
    papers_df = pd.concat([train_df, test_df, validation_df])
    load_qasper_to_db(papers_df)

if __name__ == "__main__":
    login(token=settings.HF_TOKEN)
    main()



