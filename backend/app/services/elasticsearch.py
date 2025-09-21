from elasticsearch import Elasticsearch
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from app.core.config import settings, ElasticsearchConfig

class ElasticsearchService:
    def __init__(self, config: ElasticsearchConfig):
        auth = (config.username, config.password) if config.username and config.password else None
        self.client = Elasticsearch([config.url], basic_auth=auth)
        self.index = config.index
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Verify connection
        if not self.client.ping():
            raise ConnectionError("Failed to connect to Elasticsearch")
    
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
                    # Database ID field
                    "id": {
                        "type": "integer"
                    },
                    
                    # Paper identifier
                    "paper_id": {
                        "type": "keyword"
                    },
                    
                    # Text fields for full-text search
                    "title": {
                        "type": "text",
                        "analyzer": "paper_analyzer",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    "abstract": {
                        "type": "text",
                        "analyzer": "paper_analyzer"
                    },
                    "full_text": {
                        "type": "text",
                        "analyzer": "paper_analyzer"
                    },
                    
                    # Vector fields for semantic search (384 dimensions for all-MiniLM-L6-v2)
                    "title_vector": {
                        "type": "dense_vector",
                        "dims": 384,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "abstract_vector": {
                        "type": "dense_vector",
                        "dims": 384,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "full_text_vector": {
                        "type": "dense_vector",
                        "dims": 384,
                        "index": True,
                        "similarity": "cosine"
                    },
                    
                    # Authors as keyword array for exact matching
                    "authors": {
                        "type": "keyword"
                    },
                    
                    # Date field
                    "published_date": {
                        "type": "date"
                    },
                    
                    # URL field
                    "url": {
                        "type": "keyword"
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
            print(f"Successfully created index '{target_index}'")
            return True
            
        except Exception as e:
            print(f"Error creating index '{target_index}': {str(e)}")
            return False
    