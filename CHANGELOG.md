# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### 🚀 Added - Semantic Search Implementation

#### New Search Capabilities
- **Semantic Search**: Vector-based search using sentence-transformers
  - Model: `all-MiniLM-L6-v2` (384 dimensions)
  - Powered by pgvector extension for PostgreSQL
  - Understands meaning and context, not just keywords
  
- **Hybrid Search**: Combines semantic and full-text search
  - Configurable weighting (default: 70% semantic + 30% text)
  - Provides comprehensive results with best of both approaches
  
- **Enhanced Full-Text Search**: Improved PostgreSQL tsvector implementation
  - Weighted ranking: Title (A) > Abstract (B) > Full text (C)
  - GIN indexing for fast performance

#### New Backend Components
- `app/services/embedding.py`: Embedding service for vector generation
- `app/tools/search.py`: Enhanced with semantic and hybrid search tools
- `app/db/models.py`: Added vector column for embeddings
- `alembic/versions/add_embedding_vector_column.py`: Database migration
- `test_semantic_search.py`: Comprehensive test suite

#### Agent Intelligence
- **Automatic Tool Selection**: Agent intelligently chooses best search method
  - Conceptual queries → Semantic search
  - Keyword searches → Full-text search
  - Complex queries → Hybrid search
- **Graceful Error Handling**: Provides fallback responses when database unavailable
- **Real-time Embedding Generation**: On-demand vector creation with progress tracking

#### Dependencies Added
- `sentence-transformers>=2.2.0`: For embedding generation
- `pgvector>=0.2.0`: PostgreSQL vector operations
- `numpy>=1.24.0`: Vector computations

#### Testing & Documentation
- Complete test suite for semantic search functionality
- Updated README files with detailed implementation guide
- Added testing instructions for both with/without database scenarios
- Comprehensive API examples and usage patterns

### 🔧 Technical Improvements
- **Async Support**: All search operations fully asynchronous
- **Error Handling**: Robust error handling with graceful degradation
- **Performance**: Efficient vector operations with MPS device support (Apple Silicon)
- **Scalability**: Configurable embedding models and dimensions

### 🎯 User Experience
- **Natural Language Queries**: Users can ask conceptual questions
- **Intelligent Responses**: Agent explains search strategy and provides context
- **Real-time Feedback**: Progress indicators for embedding generation
- **Fallback Mode**: Works without database for development/testing

## Usage Examples

### Semantic Search Queries
```
"Find papers about attention mechanisms in neural networks"
"Research on transformer architectures"
"Papers discussing self-supervised learning"
```

### Full-Text Search Queries  
```
"BERT transformer papers"
"Vaswani et al 2017"
"Papers by Geoffrey Hinton"
```

### Hybrid Search Queries
```
"Latest developments in large language models"
"Multimodal learning with vision and text"
"Efficient training methods for deep networks"
```

## Testing Status
- ✅ Embedding service functional
- ✅ All search tools implemented
- ✅ Agent integration complete
- ✅ Error handling verified
- ✅ Frontend compatibility confirmed
- ✅ LangGraph Studio integration working

## Migration Notes
- Database migration required for vector column
- New dependencies need installation via `uv sync`
- Environment variables added for embedding configuration
- Backward compatible with existing full-text search
