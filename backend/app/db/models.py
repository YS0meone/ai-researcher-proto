from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase, relationship
from sqlalchemy import String, Integer, BigInteger, Boolean, DateTime, Text, Table, Column, ForeignKey, Index, func
from sqlalchemy.dialects.postgresql import ARRAY, TSVECTOR
from pydantic import BaseModel, ConfigDict, field_serializer, field_validator
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum


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

