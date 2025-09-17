"""Add embedding vector column for semantic search

Revision ID: add_embedding_vector
Revises: 375c3cdd70b8
Create Date: 2025-09-17 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = 'add_embedding_vector'
down_revision: Union[str, Sequence[str], None] = '375c3cdd70b8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    
    # Add embedding column
    op.add_column('papers', sa.Column('embedding', Vector(384), nullable=True))
    
    # Create index for vector similarity search
    op.execute("CREATE INDEX papers_embedding_idx ON papers USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)")


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('papers_embedding_idx', table_name='papers')
    op.drop_column('papers', 'embedding')
