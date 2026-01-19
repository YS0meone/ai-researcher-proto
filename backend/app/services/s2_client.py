from semanticscholar import SemanticScholar
from app.core.config import settings
from app.db.schema import S2Paper
from typing import List, Union

class S2Client:
    def __init__(self):
        self.client = SemanticScholar(api_key=settings.S2_API_KEY)
    

    def convert_to_s2_paper(self, result_item) -> S2Paper:
        """
        Converts a semanticscholar.Paper object into your S2Paper Pydantic model.
        """
        if result_item is None:
            return None
        return S2Paper(**result_item.raw_data)

    def search_papers(
                self,
                query: str,
                year: str = None,
                publication_types: list = None,
                open_access_pdf: bool = None,
                venue: list = None,
                fields_of_study: list = None,
                fields: list = [
                    'paperId', 
                    'title', 
                    'abstract', 
                    'year', 
                    'authors', 
                    'citationCount'
                ],
                publication_date_or_year: str = None,
                min_citation_count: int = None,
                limit: int = 10,
                bulk: bool = False,
                sort: str = None,
                match_title: bool = False
            ):

        results = self.client.search_paper(
            query=query,
            year=year,
            publication_types=publication_types,
            open_access_pdf=open_access_pdf,
            venue=venue,
            fields_of_study=fields_of_study,
            fields=fields,
            publication_date_or_year=publication_date_or_year,
            min_citation_count=min_citation_count,
            limit=limit,
            bulk=bulk,
            sort=sort,
            match_title=match_title
        )

        if match_title:
            return [self.convert_to_s2_paper(results)]

        papers = []
        for i in range(len(results)):
            papers.append(self.convert_to_s2_paper(results[i]))
        return papers
