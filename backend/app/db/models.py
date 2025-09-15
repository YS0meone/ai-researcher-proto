from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase, relationship
from sqlalchemy import String, Integer, BigInteger, Boolean, DateTime, Text, Table, Column, ForeignKey, Index
from sqlalchemy.dialects.postgresql import ARRAY
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

