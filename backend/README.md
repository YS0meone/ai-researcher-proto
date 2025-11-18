# AI Researcher Backend

A LangGraph-powered research assistant that helps users discover and analyze academic papers through intelligent search and synthesis. Powered by OpenAI GPT-4o-mini with advanced RAG (Retrieval-Augmented Generation) capabilities.

## 🌟 Features

- **Multi-Strategy Search**: Hybrid, semantic, keyword, and category-based paper search
- **Intelligent Synthesis**: LangGraph agent orchestrates research tasks with tool calling
- **Vector Search**: Qdrant-powered semantic search on 768-dimensional embeddings
- **Full-Text Search**: Elasticsearch with metadata and abstract indexing
- **Parallel Processing**: Multi-worker data pipeline with 12 workers and batch processing
- **Fast & Slow Modes**: Metadata-only (1hr for 500k) or full PDF parsing (25hrs for 500k)
- **GROBID Integration**: Full-text extraction from PDFs for deep content search

## 📋 Prerequisites

- **Python 3.12+** with [uv](https://docs.astral.sh/uv/) for dependency management
- **Docker & Docker Compose** for services (Elasticsearch, Qdrant, Kibana, GROBID)
- **OpenAI API key** for LLM inference
- **16GB+ RAM** recommended for parallel processing
- **ArXiv metadata dataset** (see step 4 below)

## 🚀 Quick Start

### 1. Clone and Navigate

```bash
git clone <your-repo-url>
cd ai-researcher-proto/backend
```

### 2. Install Python Dependencies

```bash
uv sync
```

This installs all required dependencies including:
- **LangChain & LangGraph** - Agent framework
- **FastAPI** - API server
- **Elasticsearch 8.x** - Full-text search
- **Qdrant Client** - Vector database
- **Sentence Transformers** - allenai-specter embeddings (768-dim)
- **GROBID Parser** - PDF text extraction

### 3. Environment Setup

Copy the example environment file:

```bash
cp .env_example .env
```

Edit `.env` and configure:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Database URLs (default for Docker)
ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_INDEX=arxiv_papers
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=arxiv_papers
QDRANT_VECTOR_SIZE=768

# Paper Loader Configuration
LOADER_WORKERS=12              # Number of parallel workers
LOADER_BATCH_SIZE=100          # Papers per batch
LOADER_PROCESS_PDFS=false      # Fast mode (metadata only)
LOADER_ARXIV_METADATA_PATH=/path/to/arxiv-metadata-oai-snapshot.json
```

### 4. Download ArXiv Dataset

Download the ArXiv metadata snapshot from [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv/data):

1. Download `arxiv-metadata-oai-snapshot.json` (~3.5GB)
2. Place it in a known location
3. Update `LOADER_ARXIV_METADATA_PATH` in `.env`

**Note**: The pipeline filters for Computer Science papers (all 40 `cs.*` categories), extracting ~100k-150k papers from 500k lines.

### 5. Start Docker Services

Start Elasticsearch, Qdrant, Kibana, and GROBID:

```bash
cd .. # Go to project root
docker-compose up -d
```

Verify services are running:

```bash
curl http://localhost:9200     # Elasticsearch
curl http://localhost:6333     # Qdrant
curl http://localhost:5601     # Kibana (UI)
curl http://localhost:8070     # GROBID
```

### 6. Run Data Pipeline

**Fast Mode** (Recommended - Metadata + Embeddings Only):

```bash
cd backend
uv run python app/data_pipeline.py
```

This will:
- ✅ Process ~500k lines in **~1 hour**
- ✅ Index metadata (title, abstract, authors, etc.) to Elasticsearch
- ✅ Create title & abstract embeddings (768-dim) for semantic search
- ✅ Enable hybrid search (text + semantic)
- ❌ Skip PDF downloads and full-text extraction

**Slow Mode** (Full PDF Processing):

Set in `.env`:
```env
LOADER_PROCESS_PDFS=true
```

Then run the pipeline. This will:
- ✅ Download PDFs from ArXiv (parallel)
- ✅ Parse with GROBID for full text
- ✅ Index full document content to Qdrant
- ⏱️ Takes **~25 hours** for 500k lines

**Progress Monitoring**:
```
Overall Progress: 100%|████████████| 500000/500000 [58:42<00:00, 142lines/s, papers=98234, errors=145, rate=27.8/s]

Worker 0: Downloaded 95/100 PDFs in 42.1s (2.3 PDFs/sec)
Worker 0: GROBID parsed 1423 chunks in 218.7s (6.5 chunks/sec)
Worker 0: Indexed 1423 documents to Qdrant in 87.2s (16.3 docs/sec)
```

### 7. Start LangGraph Server

```bash
uv run langgraph dev
```

The server starts on `http://localhost:2024` with the `agent` graph available.

## 📁 Project Structure

```
backend/
├── app/
│   ├── agent/
│   │   ├── graph.py           # LangGraph agent with multi-node workflow
│   │   └── prompts.py         # Structured prompts with CoT and few-shot examples
│   ├── tools/
│   │   └── search.py          # Search tools (hybrid, semantic, keyword, category)
│   ├── services/
│   │   ├── paper_loader.py    # Parallel data pipeline with 12 workers
│   │   ├── elasticsearch.py   # ES service with hybrid search
│   │   └── qdrant.py          # Vector database service
│   ├── core/
│   │   └── config.py          # Pydantic settings management
│   └── data_pipeline.py       # Main script to load ArXiv data
├── .env                       # Environment variables (create from .env_example)
├── .env_example              # Example with all configuration options
├── langgraph.json            # LangGraph server configuration
├── pyproject.toml            # Python dependencies (managed by uv)
└── README.md                 # This file
```

## 🤖 How the Agent Works

The research agent uses a sophisticated multi-stage LangGraph workflow:

### 1. **Router Node**
Decides whether to search for papers or synthesize an answer from existing results.

### 2. **Search Agent Node**
- Analyzes user query to determine optimal search strategy
- Generates multiple search queries for comprehensive coverage
- Selects appropriate tools (hybrid, semantic, keyword, category search)
- Returns single AIMessage with multiple tool calls (proper LangGraph pattern)

### 3. **Tool Node**
Executes search tools in parallel:
- **`hybrid_search_papers`**: Combines text + semantic search (RRF scoring)
- **`semantic_search_papers`**: Vector similarity using allenai-specter
- **`keyword_search_papers`**: Full-text search on Elasticsearch
- **`search_papers_by_category`**: Filter by ArXiv categories
- **`get_paper_details`**: Fetch full paper metadata

### 4. **Reranking Node**
- Merges results from multiple searches
- Removes duplicates based on ArXiv ID
- Ranks by relevance score (text + semantic)
- Calculates coverage score for iteration control

### 5. **Synthesis Node**
- Determines if deep content search is needed
- Retrieves relevant paper sections if required
- Generates comprehensive answer with citations
- Includes paper metadata in response

### State Management
Uses Pydantic models with structured output:
- **RouterDecision**: Search vs. synthesize routing
- **SearchPlan**: Strategy + queries + tool calls
- **RerankingResult**: Deduplicated + scored papers
- **SynthesisDecision**: Deep search decision

### Graph Flow
```
START → router → search_agent → tools → merge_and_rerank
                     ↓                          ↓
                 decide_next ← increment_iter ←┘
                     ↓
                 synthesize → END
```

## 🔧 Environment Variables

### Required
- `OPENAI_API_KEY` - Your OpenAI API key

### Database Configuration
- `ELASTICSEARCH_URL` - Elasticsearch endpoint (default: `http://localhost:9200`)
- `ELASTICSEARCH_INDEX` - Index name (default: `arxiv_papers`)
- `ELASTICSEARCH_USERNAME` - ES username (if auth enabled)
- `ELASTICSEARCH_PASSWORD` - ES password (if auth enabled)
- `QDRANT_URL` - Qdrant endpoint (default: `http://localhost:6333`)
- `QDRANT_COLLECTION` - Collection name (default: `arxiv_papers`)
- `QDRANT_VECTOR_SIZE` - Embedding dimensions (default: `768`)

### Paper Loader Configuration
- `LOADER_WORKERS` - Parallel workers (default: `12`)
- `LOADER_BATCH_SIZE` - Papers per batch (default: `100`)
- `LOADER_PROCESS_PDFS` - Enable PDF processing (default: `false`)
- `LOADER_ARXIV_METADATA_PATH` - Path to ArXiv metadata JSON

## ✅ Verification & Testing

### Check Installation

```bash
# Check Python version
python --version  # Should be 3.12+

# Verify dependencies
uv pip list | grep -E "(langgraph|langchain|elasticsearch|qdrant)"

# Test graph compilation
uv run python -c "from app.agent.graph import graph; print('✅ Graph compiled')"

# Test Elasticsearch connection
uv run python -c "from app.services.elasticsearch import ElasticsearchService; from app.core.config import settings; es = ElasticsearchService(settings.elasticsearch_config); print('✅ Elasticsearch connected')"
```

### Test Search Tools

```bash
# Test hybrid search
uv run python -c "
from app.tools.search import hybrid_search_papers
results = hybrid_search_papers('transformer architecture', limit=5)
print(f'Found {len(results)} papers')
for r in results[:2]:
    print(f\"  - {r['title'][:80]}...\")
"
```

### Monitor Services

```bash
# Check Elasticsearch health
curl http://localhost:9200/_cluster/health?pretty

# Check Qdrant collection
curl http://localhost:6333/collections/arxiv_papers

# View Kibana dashboard
open http://localhost:5601
```

## 📊 Performance Metrics

### Data Pipeline (Fast Mode)
- **Throughput**: ~140-200 lines/sec
- **500k lines**: ~1 hour
- **Output**: ~100k-150k CS papers indexed

### Data Pipeline (Slow Mode with PDFs)
- **PDF Downloads**: ~2-3 PDFs/sec per worker (60 concurrent)
- **GROBID Parsing**: ~6-8 chunks/sec per worker
- **Qdrant Indexing**: ~15-20 docs/sec per worker
- **500k lines**: ~25 hours total

### Search Performance
- **Hybrid Search**: ~100-300ms for top-10 results
- **Semantic Search**: ~50-150ms for vector similarity
- **Keyword Search**: ~20-80ms for text matching

## 🐛 Troubleshooting

### "Cannot copy out of meta tensor" Error
**Issue**: Embedding model not properly initialized  
**Solution**: Restart LangGraph server - the singleton pattern will properly load the model

### Slow Data Pipeline
**Issue**: PDF processing is slow  
**Solution**: Use fast mode (`LOADER_PROCESS_PDFS=false`) for initial loading

### Elasticsearch Connection Failed
**Issue**: Docker container not running  
**Solution**: `docker-compose up -d elasticsearch` and verify with `curl http://localhost:9200`

### Qdrant Collection Not Found
**Issue**: Data pipeline hasn't run yet  
**Solution**: Run `uv run python app/data_pipeline.py` to create collections

## 📚 Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Elasticsearch Python Client](https://elasticsearch-py.readthedocs.io/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [GROBID Documentation](https://grobid.readthedocs.io/)
