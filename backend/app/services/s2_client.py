from semanticscholar import SemanticScholar
from app.core.config import settings
from app.db.schema import S2Paper
from typing import List, Union
import time

class S2Client:
    def __init__(self):
        # Configure with longer timeout and rate limit handling
        self.client = SemanticScholar(
            api_key=settings.S2_API_KEY,
            timeout=30,  # Increase timeout to 30 seconds
            # Note: The library automatically retries arXiv queries,
            # but will gracefully fall back to S2-only data on HTTP 429
        )
    

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
    
    def get_paper_citations(
        self,
        paper_id: str,
        fields: list = None,
        limit: int = 1000
    ) -> List[dict]:
        """
        Get papers that cite the given paper (backward snowball).
        
        Args:
            paper_id: The paper ID (paperId or CorpusId)
            fields: List of fields to retrieve for each citing paper
            limit: Maximum number of citations to retrieve
            
        Returns:
            List of paper dictionaries (raw data from S2 API)
        """
        if fields is None:
            fields = [
                'paperId', 'corpusId', 'title', 'abstract', 'authors',
                'year', 'citationCount', 'influentialCitationCount',
                'isInfluential'
            ]
        try:
            # Get citations using the semanticscholar library
            # Note: limit parameter is page size, not total results
            citations = self.client.get_paper_citations(
                paper_id=paper_id,
                fields=fields,
                limit=1000  # Max 1000 per page
            )
            
            # Convert to list of raw data dicts, respecting the limit
            # Note: API returns structure with 'citingPaper' containing the actual paper data
            result = []
            for i, citation in enumerate(citations):
                if i >= limit:  # Stop after reaching the desired limit
                    break
                
                # Extract the citing paper data from the nested structure
                if citation and hasattr(citation, 'raw_data'):
                    raw = citation.raw_data
                    # The actual paper data is in 'citingPaper' field
                    if isinstance(raw, dict) and 'citingPaper' in raw:
                        result.append(raw['citingPaper'])
                    else:
                        result.append(raw)
                elif isinstance(citation, dict):
                    # Handle if it's already a dict
                    if 'citingPaper' in citation:
                        result.append(citation['citingPaper'])
                    else:
                        result.append(citation)
            
            return result
            
        except Exception as e:
            print(f"Error fetching citations for {paper_id}: {e}")
            return []
    
    def get_paper_references(
        self,
        paper_id: str,
        fields: list = None,
        limit: int = 1000
    ) -> List[dict]:
        """
        Get papers that the given paper cites/references (forward snowball).
        
        Args:
            paper_id: The paper ID (paperId or CorpusId)
            fields: List of fields to retrieve for each referenced paper
            limit: Maximum number of references to retrieve
            
        Returns:
            List of paper dictionaries (raw data from S2 API)
        """
        if fields is None:
            fields = [
                'paperId', 'corpusId', 'title', 'abstract', 'authors',
                'year', 'citationCount', 'influentialCitationCount',
                'isInfluential'
            ]
        
        try:
            # Get references using the semanticscholar library
            # Note: limit parameter is page size, not total results
            references = self.client.get_paper_references(
                paper_id=paper_id,
                fields=fields,
                limit=min(limit, 1000)  # Max 1000 per page
            )
            
            # Convert to list of raw data dicts, respecting the limit
            # Note: API returns structure with 'citedPaper' containing the actual paper data
            result = []
            for i, reference in enumerate(references):
                if i >= limit:  # Stop after reaching the desired limit
                    break
                
                # Extract the cited paper data from the nested structure
                if reference and hasattr(reference, 'raw_data'):
                    raw = reference.raw_data
                    # The actual paper data is in 'citedPaper' field
                    if isinstance(raw, dict) and 'citedPaper' in raw:
                        result.append(raw['citedPaper'])
                    else:
                        result.append(raw)
                elif isinstance(reference, dict):
                    # Handle if it's already a dict
                    if 'citedPaper' in reference:
                        result.append(reference['citedPaper'])
                    else:
                        result.append(reference)
            
            return result
            
        except Exception as e:
            print(f"Error fetching references for {paper_id}: {e}")
            return []
