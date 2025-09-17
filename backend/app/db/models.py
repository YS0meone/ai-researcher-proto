from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase, relationship
from sqlalchemy import String, Integer, BigInteger, Boolean, DateTime, Text, Table, Column, ForeignKey, Index, func
from sqlalchemy.dialects.postgresql import ARRAY, TSVECTOR
from pgvector.sqlalchemy import Vector
from pydantic import BaseModel, ConfigDict, field_serializer, field_validator
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
import numpy as np


class Base(DeclarativeBase):
    pass

class Paper(Base):
    __tablename__ = "papers"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    paper_id: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    abstract: Mapped[str] = mapped_column(Text, nullable=False)
    authors: Mapped[List[str]] = mapped_column(ARRAY(String), nullable=False)
    published_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    url: Mapped[str] = mapped_column(String, nullable=False)
    full_text: Mapped[str] = mapped_column(Text, nullable=False)
    search_vector: Mapped[Optional[str]] = mapped_column(TSVECTOR, nullable=True)
    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(384), nullable=True)

    @classmethod
    async def search(cls, session, query: str, limit: int = 10):
        """Full-text search papers using PostgreSQL tsvector."""
        from sqlalchemy import select
        
        search_query = func.plainto_tsquery('english', query)
        
        stmt = select(
            cls,
            func.ts_rank(cls.search_vector, search_query).label('relevance')
        ).filter(
            cls.search_vector.op('@@')(search_query)
        ).order_by(
            func.ts_rank(cls.search_vector, search_query).desc()
        ).limit(limit)
        
        result = await session.execute(stmt)
        return result.all()

    @classmethod
    async def semantic_search(cls, session, query_embedding: List[float], limit: int = 10):
        """Semantic search papers using vector similarity."""
        from sqlalchemy import select
        
        stmt = select(
            cls,
            cls.embedding.cosine_distance(query_embedding).label('distance')
        ).filter(
            cls.embedding.is_not(None)
        ).order_by(
            cls.embedding.cosine_distance(query_embedding)
        ).limit(limit)
        
        result = await session.execute(stmt)
        return result.all()

    @classmethod
    async def hybrid_search(cls, session, query: str, query_embedding: List[float], limit: int = 10, semantic_weight: float = 0.7):
        """Hybrid search combining full-text and semantic search."""
        from sqlalchemy import select, case, literal
        
        # Full-text search component
        search_query = func.plainto_tsquery('english', query)
        text_relevance = func.ts_rank(cls.search_vector, search_query)
        
        # Semantic search component (convert distance to similarity)
        semantic_similarity = 1 - cls.embedding.cosine_distance(query_embedding)
        
        # Combined score
        combined_score = (
            semantic_weight * semantic_similarity + 
            (1 - semantic_weight) * text_relevance
        )
        
        stmt = select(
            cls,
            combined_score.label('relevance'),
            text_relevance.label('text_relevance'),
            semantic_similarity.label('semantic_similarity')
        ).filter(
            cls.search_vector.op('@@')(search_query) | 
            cls.embedding.is_not(None)
        ).order_by(
            combined_score.desc()
        ).limit(limit)
        
        result = await session.execute(stmt)
        return result.all()

