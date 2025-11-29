from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class ArxivPaper(BaseModel):
    """Pydantic model for ArXiv paper data matching the exact schema."""
    
    # Required fields
    id: str = Field(..., description="ArXiv ID (e.g., '0704.0001')")
    title: str = Field(..., description="Paper title")
    abstract: str = Field(..., description="Paper abstract")
    
    # Optional fields
    submitter: Optional[str] = Field(None, description="Name of the submitter")
    authors: Optional[str] = Field(None, description="Authors as a string (e.g., 'C. Bal\\'azs, E. L. Berger')")
    comments: Optional[str] = Field(None, description="Additional comments")
    journal_ref: Optional[str] = Field(None, alias="journal-ref", description="Journal reference")
    doi: Optional[str] = Field(None, description="Digital Object Identifier")
    report_no: Optional[str] = Field(None, alias="report-no", description="Report number")
    categories: Optional[str] = Field(None, description="Space-separated categories (e.g., 'hep-ph')")
    license: Optional[str] = Field(None, description="License information")
    supporting_detail: Optional[str] = Field(None, description="Supporting detail")
    
    # Additional fields for our system
    submission_date: Optional[datetime] = Field(None, description="Date of submission")
    update_date: Optional[datetime] = Field(None, description="Date of last update")
    
    class Config:
        populate_by_name = True  # Allow both field names and aliases
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
    
    def to_elasticsearch_doc(self) -> dict:
        """Convert to dictionary format for Elasticsearch indexing."""
        doc = self.model_dump(by_alias=True, exclude_none=True)
        
        # Ensure proper field names for Elasticsearch
        if 'journal_ref' in doc:
            doc['journal-ref'] = doc.pop('journal_ref')
        if 'report_no' in doc:
            doc['report-no'] = doc.pop('report_no')
            
        return doc

class ArxivPaperBatch(BaseModel):
    """Model for batch operations on multiple papers."""
    papers: List[ArxivPaper] = Field(..., description="List of ArXiv papers")
    
    def to_elasticsearch_docs(self) -> List[dict]:
        """Convert all papers to Elasticsearch document format."""
        return [paper.to_elasticsearch_doc() for paper in self.papers]