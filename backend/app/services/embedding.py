"""Embedding service for generating vector representations of text."""
import asyncio
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from app.core.config import settings
import numpy as np


class EmbeddingService:
    """Service for generating embeddings using sentence-transformers."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._model: Optional[SentenceTransformer] = None
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def encode(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    async def encode_async(self, text: str) -> List[float]:
        """Generate embedding asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.encode, text)
    
    async def encode_batch_async(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.encode_batch, texts)
    
    def create_paper_embedding(self, title: str, abstract: str, full_text: str = "") -> List[float]:
        """Create embedding for a paper combining title, abstract, and optionally full text."""
        # Combine text with different weights - title and abstract are more important
        combined_text = f"{title}. {abstract}"
        if full_text:
            # Truncate full text to avoid memory issues and focus on most important parts
            full_text_truncated = full_text[:2000] if len(full_text) > 2000 else full_text
            combined_text += f" {full_text_truncated}"
        
        return self.encode(combined_text)
    
    async def create_paper_embedding_async(self, title: str, abstract: str, full_text: str = "") -> List[float]:
        """Create embedding for a paper asynchronously."""
        combined_text = f"{title}. {abstract}"
        if full_text:
            full_text_truncated = full_text[:2000] if len(full_text) > 2000 else full_text
            combined_text += f" {full_text_truncated}"
        
        return await self.encode_async(combined_text)


# Global instance
embedding_service = EmbeddingService()
