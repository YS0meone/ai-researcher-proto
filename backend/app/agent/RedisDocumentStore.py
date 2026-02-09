import pickle
from typing import List, Optional, Sequence, Tuple, Iterator
from langchain_core.documents import Document
from langchain_core.stores import BaseStore
from langchain_community.storage import RedisStore


class RedisDocumentStore(BaseStore[str, Document]):
    """RedisStore wrapper that automatically serializes/deserializes Documents.
    
    This makes RedisStore compatible with ParentDocumentRetriever.
    """
    
    def __init__(
        self,
        *,
        client = None,
        redis_url: Optional[str] = None,
        client_kwargs: Optional[dict] = None,
        ttl: Optional[int] = None,
        namespace: Optional[str] = None,
    ):
        """Initialize RedisDocumentStore.
        
        Args:
            client: A Redis connection instance
            redis_url: redis url (e.g., "redis://localhost:6379")
            client_kwargs: Keyword arguments to pass to the Redis client
            ttl: time to expire keys in seconds if provided
            namespace: if provided, all keys will be prefixed with this namespace
        """
        # Initialize underlying RedisStore
        self._redis_store = RedisStore(
            client=client,
            redis_url=redis_url,
            client_kwargs=client_kwargs,
            ttl=ttl,
            namespace=namespace or "parent_docs",  # Default namespace
        )
    
    def _serialize_document(self, doc: Document) -> bytes:
        """Serialize a Document to bytes using pickle."""
        return pickle.dumps(doc)
    
    def _deserialize_document(self, data: bytes) -> Document:
        """Deserialize bytes to a Document using pickle."""
        return pickle.loads(data)
    
    def mget(self, keys: Sequence[str]) -> List[Optional[Document]]:
        """Get Documents for the given keys.
        
        Args:
            keys: List of keys to retrieve
            
        Returns:
            List of Documents (or None for missing keys)
        """
        # Get raw bytes from Redis
        raw_values = self._redis_store.mget(keys)
        
        # Deserialize bytes to Documents
        documents = []
        for value in raw_values:
            if value is None:
                documents.append(None)
            else:
                try:
                    doc = self._deserialize_document(value)
                    documents.append(doc)
                except Exception as e:
                    print(f"Error deserializing document: {e}")
                    documents.append(None)
        
        return documents
    
    def mset(self, key_value_pairs: Sequence[Tuple[str, Document]]) -> None:
        """Set Documents for the given keys.
        
        Args:
            key_value_pairs: List of (key, Document) tuples
        """
        # Serialize Documents to bytes
        serialized_pairs = []
        for key, doc in key_value_pairs:
            if not isinstance(doc, Document):
                raise TypeError(
                    f"Expected Document, got {type(doc).__name__}. "
                    "RedisDocumentStore only accepts Document objects."
                )
            
            serialized_value = self._serialize_document(doc)
            serialized_pairs.append((key, serialized_value))
        
        # Store in Redis
        self._redis_store.mset(serialized_pairs)
    
    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete the given keys.
        
        Args:
            keys: List of keys to delete
        """
        self._redis_store.mdelete(keys)
    
    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        """Yield keys in the store.
        
        Args:
            prefix: Optional prefix to filter keys
            
        Yields:
            Keys matching the prefix
        """
        return self._redis_store.yield_keys(prefix=prefix)