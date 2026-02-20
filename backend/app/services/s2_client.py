import httpx
from app.core.config import settings
from app.core.schema import S2Paper
from typing import List, Optional

S2_BASE = "https://api.semanticscholar.org/graph/v1"


class S2Client:
    def __init__(self):
        self._headers = {"x-api-key": settings.S2_API_KEY} if settings.S2_API_KEY else {}
        self._timeout = 30.0

    def _to_paper(self, data: dict) -> Optional[S2Paper]:
        if not data or not data.get("paperId"):
            return None
        try:
            return S2Paper(**data)
        except Exception:
            return None

    async def search_papers(
        self,
        query: str,
        year: str = None,
        publication_types: list = None,
        open_access_pdf: bool = None,
        venue: list = None,
        fields_of_study: list = None,
        fields: list = None,
        publication_date_or_year: str = None,
        min_citation_count: int = None,
        limit: int = 10,
        bulk: bool = False,
        sort: str = None,
        match_title: bool = False,
    ) -> List[S2Paper]:
        if fields is None:
            fields = ["paperId", "title", "abstract", "year", "authors", "citationCount"]

        params: dict = {
            "query": query,
            "fields": ",".join(fields),
            "limit": 1 if match_title else min(limit, 100),
        }
        if year:
            params["year"] = year
        if publication_types:
            params["publicationTypes"] = ",".join(publication_types)
        if venue:
            params["venue"] = ",".join(venue)
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)
        if publication_date_or_year:
            params["publicationDateOrYear"] = publication_date_or_year
        if min_citation_count is not None:
            params["minCitationCount"] = min_citation_count
        if sort:
            params["sort"] = sort

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                f"{S2_BASE}/paper/search",
                params=params,
                headers=self._headers,
            )
            resp.raise_for_status()

        papers = [self._to_paper(item) for item in resp.json().get("data", [])]
        return [p for p in papers if p is not None]

    async def get_paper_citations(
        self,
        paper_id: str,
        fields: list = None,
        limit: int = 1000,
    ) -> List[dict]:
        if fields is None:
            fields = [
                "paperId", "corpusId", "title", "abstract", "authors",
                "year", "citationCount", "influentialCitationCount",
            ]
        params = {
            "fields": ",".join(f"citingPaper.{f}" for f in fields),
            "limit": min(limit, 1000),
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                f"{S2_BASE}/paper/{paper_id}/citations",
                params=params,
                headers=self._headers,
            )
            resp.raise_for_status()

        return [
            item["citingPaper"]
            for item in resp.json().get("data", [])
            if item.get("citingPaper")
        ]

    async def get_paper_references(
        self,
        paper_id: str,
        fields: list = None,
        limit: int = 1000,
    ) -> List[dict]:
        if fields is None:
            fields = [
                "paperId", "corpusId", "title", "abstract", "authors",
                "year", "citationCount", "influentialCitationCount",
            ]
        params = {
            "fields": ",".join(f"citedPaper.{f}" for f in fields),
            "limit": min(limit, 1000),
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                f"{S2_BASE}/paper/{paper_id}/references",
                params=params,
                headers=self._headers,
            )
            resp.raise_for_status()

        return [
            item["citedPaper"]
            for item in resp.json().get("data", [])
            if item.get("citedPaper")
        ]
