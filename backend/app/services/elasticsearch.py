from elasticsearch import Elasticsearch
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import numpy as np
from sentence_transformers import SentenceTransformer
from app.core.config import settings, ElasticsearchConfig
from app.db.schema import ArxivPaper, ArxivPaperBatch
import time
import torch

# Global singleton for embedding model to avoid reloading
_embedding_model_singleton = None
_embedding_model_lock = None

def get_embedding_model():
    """Get or create the singleton embedding model instance (thread-safe)."""
    global _embedding_model_singleton, _embedding_model_lock
    import threading
    
    # Initialize lock if needed
    if _embedding_model_lock is None:
        _embedding_model_lock = threading.Lock()
    
    # Double-checked locking pattern
    if _embedding_model_singleton is None:
        with _embedding_model_lock:
            if _embedding_model_singleton is None:
                print("Loading embedding model: allenai-specter", flush=True)
                try:
                    import os
                    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                    
                    # Load model with explicit settings to avoid meta tensor issues
                    model = SentenceTransformer(
                        'allenai-specter',
                        device='cpu',
                    )
                    
                    # Force full materialization
                    model.eval()
                    for param in model.parameters():
                        param.data  # Force tensor materialization
                    
                    # Verify with test encode
                    test_embedding = model.encode("test", show_progress_bar=False)
                    print(f"Embedding model loaded on cpu (shape: {test_embedding.shape})", flush=True)
                    
                    _embedding_model_singleton = model
                    
                except Exception as e:
                    print(f"Error loading embedding model: {e}", flush=True)
                    raise
    
    return _embedding_model_singleton

class ElasticsearchService:
    def __init__(self, config: ElasticsearchConfig):
        auth = (config.username, config.password) if config.username and config.password else None
        print(config.url)
        self.client = Elasticsearch(
            [config.url], basic_auth=auth,
            verify_certs=False,
            ssl_show_warn=False,
        )
        self.index = config.index
        # Use singleton embedding model
        self.embedding_model = get_embedding_model()
        self._wait_for_connection()
        is_healthy = self.client.ping()
        print(is_healthy)   
        if not is_healthy:
            raise ConnectionError("Failed to connect to Elasticsearch")
        self.create_index()

    def _wait_for_connection(self, retries: int = 10, delay: float = 2.0) -> None:
        last_error: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                if self.client.ping():
                    return
            except Exception as exc:
                last_error = exc
            time.sleep(delay)
        message = "Failed to connect to Elasticsearch after multiple attempts"
        if last_error:
            message += f": {last_error}"
        raise ConnectionError(message)

    def get_paper(self, arxiv_id: str, index_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        target_index = index_name or self.index
        try:
            resp = self.client.get(index=target_index, id=arxiv_id)
            src = resp.get('_source', {}) or {}
            return {"id": arxiv_id, **src}
        except Exception as e:
            print(f"Error fetching paper {arxiv_id}: {e}")
            return None
            
    def create_index(self, index_name: Optional[str] = None) -> bool:
        """
        Create an index for storing academic papers with proper mapping for hybrid search.
        
        The index includes:
        - Text fields for full-text search (title, abstract, content)
        - Vector fields for semantic search (title_vector, abstract_vector)
        - Keyword fields for exact matching (authors, categories, doi)
        - Date and numeric fields for filtering
        
        Args:
            index_name: Optional index name, defaults to self.index
            
        Returns:
            bool: True if index was created or already exists, False otherwise
        """
        target_index = index_name or self.index
        
        # Check if index already exists
        if self.client.indices.exists(index=target_index):
            print(f"Index '{target_index}' already exists")
            return True
        
        # Define the mapping for the papers index (matching the Paper model schema)
        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "paper_analyzer": {
                            "type": "standard",
                            "stopwords": "_english_"
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    # ArXiv ID (the main identifier)
                    "id": {
                        "type": "keyword"
                    },
                    
                    # Submitter name
                    "submitter": {
                        "type": "text",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    
                    # Authors (string format as in ArXiv)
                    "authors": {
                        "type": "text",
                        "analyzer": "paper_analyzer",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 512
                            }
                        }
                    },
                    
                    # Title (text field for full-text search)
                    "title": {
                        "type": "text",
                        "analyzer": "paper_analyzer",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 512
                            }
                        }
                    },
                    
                    # Comments
                    "comments": {
                        "type": "text",
                        "analyzer": "paper_analyzer",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    
                    # Journal reference
                    "journal-ref": {
                        "type": "text",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    
                    # DOI
                    "doi": {
                        "type": "keyword"
                    },
                    
                    # Report number
                    "report-no": {
                        "type": "keyword"
                    },
                    
                    # Categories (space-separated string)
                    "categories": {
                        "type": "keyword"
                    },
                    
                    # License (can be null)
                    "license": {
                        "type": "keyword"
                    },
                    
                    # Abstract (main searchable content)
                    "abstract": {
                        "type": "text",
                        "analyzer": "paper_analyzer"
                    },
                                        
                    # Vector fields for semantic search (768 dimensions for SPECTER)
                    "title_vector": {
                        "type": "dense_vector",
                        "dims": 768,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "abstract_vector": {
                        "type": "dense_vector",
                        "dims": 768,
                        "index": True,
                        "similarity": "cosine"
                    },
                    
                    # Derived fields for easier querying
                    "categories_array": {
                        "type": "keyword"
                    },
                    "submission_date": {
                        "type": "date"
                    },
                    "update_date": {
                        "type": "date"
                    }
                }
            }
        }
        
        try:
            # Create the index
            response = self.client.indices.create(
                index=target_index,
                body=mapping
            )
            print(f"Successfully created index '{target_index}', {response}")
            return True
            
        except Exception as e:
            print(f"Error creating index '{target_index}': {str(e)}")
            return False
    # Add these methods to your ElasticsearchService class:

    def add_paper(self, paper: 'ArxivPaper', index_name: Optional[str] = None) -> bool:
        """
        Add a single paper to the Elasticsearch index with vector embeddings.
        
        Args:
            paper: ArxivPaper model instance
            index_name: Optional index name, defaults to self.index
            
        Returns:
            bool: True if paper was successfully indexed, False otherwise
        """
        target_index = index_name or self.index
        
        try:
            # Convert paper to dict format for Elasticsearch
            doc = paper.to_elasticsearch_doc()
            
            # Generate vector embeddings for semantic search using SPECTER
            if paper.title:
                doc['title_vector'] = self.embedding_model.encode(paper.title).tolist()
            
            if paper.abstract:
                doc['abstract_vector'] = self.embedding_model.encode(paper.abstract).tolist()
            
            # Parse categories into array for easier filtering
            if paper.categories:
                doc['categories_array'] = paper.categories.split()
            
            # Use ArXiv ID as document ID for upserts
            doc_id = paper.id
            
            # Index the document
            response = self.client.index(
                index=target_index,
                id=doc_id,
                body=doc
            )
            
            print(f"Successfully indexed paper: {doc_id} - {paper.title[:50]}...")
            return True
            
        except Exception as e:
            print(f"Error indexing paper {paper.id}: {str(e)}")
            return False

    def add_papers_batch(self, papers: List['ArxivPaper'], index_name: Optional[str] = None) -> Dict[str, int]:
        """
        Bulk add multiple papers to the Elasticsearch index.
        
        Args:
            papers: List of ArxivPaper model instances
            index_name: Optional index name, defaults to self.index
            
        Returns:
            Dict with success and error counts
        """
        from elasticsearch.helpers import bulk
        
        target_index = index_name or self.index
        
        def generate_docs():
            for paper in papers:
                try:
                    # Convert paper to dict format
                    doc = paper.to_elasticsearch_doc()
                    
                    # Generate embeddings using SPECTER
                    if paper.title:
                        doc['title_vector'] = self.embedding_model.encode(paper.title).tolist()
                    
                    if paper.abstract:
                        doc['abstract_vector'] = self.embedding_model.encode(paper.abstract).tolist()
                    
                    # Parse categories into array
                    if paper.categories:
                        doc['categories_array'] = paper.categories.split()
                    
                    yield {
                        "_index": target_index,
                        "_id": paper.id,
                        "_source": doc
                    }
                except Exception as e:
                    print(f"Error preparing paper {paper.id}: {str(e)}")
                    continue
        
        try:
            success_count, errors = bulk(
                self.client,
                generate_docs(),
                index=target_index,
                chunk_size=100,
                request_timeout=60,
                max_retries=3,
                initial_backoff=2,
                max_backoff=600
            )
            
            result = {
                'success': success_count,
                'errors': len(errors) if errors else 0
            }
            
            print(f"Bulk indexing completed: {result['success']} successful, {result['errors']} errors")
            
            # Print error details if any
            if errors:
                for error in errors[:5]:  # Show first 5 errors
                    print(f"Error detail: {error}")
            
            return result
            
        except Exception as e:
            print(f"Error in bulk indexing: {str(e)}")
            return {'success': 0, 'errors': len(papers)}
    
    def add_paper_batch(self, batch: 'ArxivPaperBatch', index_name: Optional[str] = None) -> Dict[str, int]:
        """
        Add papers using ArxivPaperBatch model.
        
        Args:
            batch: ArxivPaperBatch model instance
            index_name: Optional index name
            
        Returns:
            Dict with success and error counts
        """
        return self.add_papers_batch(batch.papers, index_name)
    
    def hybrid_search(
        self, 
        query: str, 
        limit: int = 10,
        categories_filter: List[str] = None,
        text_weight: float = 0.7,
        vector_weight: float = 0.3,
        index_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining text search and vector similarity.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            categories_filter: Optional list of categories to filter by
            text_weight: Weight for text-based search (0.0 to 1.0)
            vector_weight: Weight for vector similarity search (0.0 to 1.0)
            index_name: Optional index name, defaults to self.index
            
        Returns:
            List of search results with scores and paper data
        """
        target_index = index_name or self.index
        
        try:
            # Generate query embedding for vector search
            query_vector = self.embedding_model.encode(query).tolist()
            
            # Build the hybrid search query
            search_body = {
                "size": limit,
                "query": {
                    "bool": {
                        "should": [
                            # Text-based search component
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": [
                                        "title^3",      # Boost title matches
                                        "abstract^2",   # Boost abstract matches
                                        "authors^1.5",  # Boost author matches
                                        "comments",
                                        "journal-ref"
                                    ],
                                    "type": "best_fields",
                                    "fuzziness": "AUTO"
                                }
                            }
                        ],
                        "filter": []
                    }
                },
                "knn": {
                    "field": "title_vector",
                    "query_vector": query_vector,
                    "k": limit * 2,  # Get more candidates for reranking
                    "num_candidates": 100
                },
                "_source": {
                    "excludes": ["title_vector", "abstract_vector"]  # Exclude vectors from results
                },
                "highlight": {
                    "fields": {
                        "title": {},
                        "abstract": {"fragment_size": 150, "number_of_fragments": 2},
                        "authors": {}
                    }
                }
            }
            
            # Add category filter if specified
            if categories_filter:
                search_body["query"]["bool"]["filter"].append({
                    "terms": {
                        "categories_array": categories_filter
                    }
                })
            
            # Execute the search
            response = self.client.search(
                index=target_index,
                body=search_body
            )
            
            # Process and combine results
            results = []
            for hit in response['hits']['hits']:
                result = {
                    'id': hit['_id'],
                    'score': hit['_score'],
                    'source': hit['_source'],
                    'highlights': hit.get('highlight', {})
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error in hybrid search: {str(e)}")
            return []
    
    def semantic_search(
        self, 
        query: str, 
        limit: int = 10,
        field: str = "title_vector",
        categories_filter: List[str] = None,
        index_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform pure semantic search using vector similarity.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            field: Vector field to search against (title_vector, abstract_vector)
            categories_filter: Optional categories to filter by
            index_name: Optional index name
            
        Returns:
            List of semantically similar papers
        """
        target_index = index_name or self.index
        
        try:
            # Generate query embedding
            query_vector = self.embedding_model.encode(query).tolist()
            
            search_body = {
                "size": limit,
                "knn": {
                    "field": field,
                    "query_vector": query_vector,
                    "k": limit,
                    "num_candidates": 100
                },
                "_source": {
                    "excludes": ["title_vector", "abstract_vector"]
                }
            }
            
            # Add category filter if specified
            if categories_filter:
                search_body["query"] = {
                    "bool": {
                        "filter": [{
                            "terms": {
                                "categories_array": categories_filter
                            }
                        }]
                    }
                }
            
            response = self.client.search(
                index=target_index,
                body=search_body
            )
            
            results = []
            for hit in response['hits']['hits']:
                result = {
                    'id': hit['_id'],
                    'score': hit['_score'],
                    'source': hit['_source']
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error in semantic search: {str(e)}")
            return []
    
    def text_search(
        self, 
        query: str, 
        limit: int = 10,
        categories_filter: List[str] = None,
        index_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform traditional text-based search.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            categories_filter: Optional categories to filter by
            index_name: Optional index name
            
        Returns:
            List of text-matching papers
        """
        target_index = index_name or self.index
        
        try:
            search_body = {
                "size": limit,
                "query": {
                    "bool": {
                        "should": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": [
                                        "title^3",
                                        "abstract^2", 
                                        "authors^1.5",
                                        "comments",
                                        "journal-ref"
                                    ],
                                    "type": "best_fields",
                                    "fuzziness": "AUTO"
                                }
                            }
                        ],
                        "filter": []
                    }
                },
                "_source": {
                    "excludes": ["title_vector", "abstract_vector"]
                },
                "highlight": {
                    "fields": {
                        "title": {},
                        "abstract": {"fragment_size": 150, "number_of_fragments": 2},
                        "authors": {}
                    }
                }
            }
            
            # Add category filter if specified
            if categories_filter:
                search_body["query"]["bool"]["filter"].append({
                    "terms": {
                        "categories_array": categories_filter
                    }
                })
            
            response = self.client.search(
                index=target_index,
                body=search_body
            )
            
            results = []
            for hit in response['hits']['hits']:
                result = {
                    'id': hit['_id'],
                    'score': hit['_score'],
                    'source': hit['_source'],
                    'highlights': hit.get('highlight', {})
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error in text search: {str(e)}")
            return []