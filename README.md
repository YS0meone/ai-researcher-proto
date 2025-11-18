# AI Researcher Proto

An AI-powered research assistant that helps users discover and analyze academic papers through intelligent search and synthesis. Built with LangGraph agents, React, and advanced RAG (Retrieval-Augmented Generation) capabilities.

## 🌟 Features

### 🔍 Intelligent Paper Search
- **Multi-Strategy Search**: Hybrid (text + semantic), pure semantic, keyword, and category-based
- **768-Dimensional Embeddings**: Using `allenai-specter` model for scientific papers
- **Vector Database**: Qdrant for semantic search on 100k+ papers
- **Full-Text Search**: Elasticsearch with metadata and abstract indexing
- **Parallel Search**: Multiple search strategies executed simultaneously

### 🤖 LangGraph Agent Orchestration
- **Multi-Node Workflow**: Router → Search Agent → Tools → Reranking → Synthesis
- **Structured Output**: Pydantic models with chain-of-thought reasoning
- **Tool Calling**: Proper LangGraph pattern with single AIMessage + multiple tool calls
- **Iteration Control**: Automatic search refinement based on coverage scores
- **Citation Generation**: Answers include paper references with metadata

### ⚡ High-Performance Data Pipeline
- **Parallel Processing**: 12 workers with batch processing (100 papers/batch)
- **Fast Mode**: Metadata + embeddings only (~1 hour for 500k papers)
- **Slow Mode**: Full PDF parsing with GROBID (~25 hours for 500k papers)
- **Progress Monitoring**: Real-time stats with tqdm progress bars
- **Deduplication**: Intelligent skip-existing logic for incremental loads

### 💬 Modern Chat Interface
- **React 19** with TypeScript and Vite
- **Real-time Streaming**: Live message updates from LangGraph
- **Thread Management**: Conversation history and context
- **shadcn/ui Components**: Beautiful, accessible UI primitives
- **Dark/Light Mode**: Tailwind CSS with theme support

## 📋 Prerequisites

- **Python 3.12+** with [uv](https://docs.astral.sh/uv/)
- **Node.js 20+** with pnpm
- **Docker & Docker Compose** for services
- **16GB+ RAM** recommended for parallel processing
- **OpenAI API Key** for GPT-4o-mini inference

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd ai-researcher-proto
```

### 2. Configure Environment

```bash
# Project root (for Docker services)
cp .env.example .env

# Backend (for Python app)
cp backend/.env_example backend/.env
```

Edit `backend/.env` with:
- Your `OPENAI_API_KEY`
- Path to ArXiv metadata JSON (`LOADER_ARXIV_METADATA_PATH`)
- Data pipeline settings (workers, batch size, PDF processing)

### 3. Start Docker Services

```bash
docker-compose up -d
```

This starts:
- **Elasticsearch** (9200) - Full-text search
- **Qdrant** (6333) - Vector database
- **Kibana** (5601) - Elasticsearch UI
- **GROBID** (8070) - PDF parsing service

### 4. Load Paper Data

Download ArXiv metadata from [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv/data), then:

```bash
cd backend
uv sync                           # Install Python dependencies
uv run python -m app.data_pipeline  # Load papers (fast mode ~1 hour)
```

### 5. Start Backend

```bash
cd backend
uv run langgraph dev  # Starts on http://localhost:2024
```

### 6. Start Frontend

```bash
cd web
pnpm install   # Install Node dependencies
pnpm dev       # Starts on http://localhost:5173
```

### 7. Start Researching!

1. Open `http://localhost:5173`
2. Enter connection details:
   - **Deployment URL**: `http://localhost:2024`
   - **Assistant ID**: `agent`
3. Ask questions like:
   - "What are transformer architectures?"
   - "Find papers about few-shot learning"
   - "Recent work on retrieval-augmented generation"

## 📁 Project Structure

```
ai-researcher-proto/
├── backend/                    # Python LangGraph backend
│   ├── app/
│   │   ├── agent/             # LangGraph agent (graph.py, prompts.py)
│   │   ├── tools/             # Search tools
│   │   ├── services/          # Elasticsearch, Qdrant, paper loader
│   │   └── data_pipeline.py  # Data loading script
│   ├── .env                   # Backend config (create from .env_example)
│   └── pyproject.toml        # Python dependencies (uv)
├── web/                       # React frontend
│   ├── src/
│   │   ├── components/       # UI components
│   │   └── providers/        # React contexts (Stream, Thread)
│   └── package.json          # Node dependencies (pnpm)
├── docker-compose.yml         # Services (ES, Qdrant, Kibana, GROBID)
└── .env                      # Docker config (create from .env_example)
```

## 🎯 Detailed Setup Guides

- **Backend**: See [`backend/README.md`](./backend/README.md) for:
  - Data pipeline configuration
  - Search tool details
  - Agent architecture
  - Performance tuning
  - Troubleshooting

- **Frontend**: See [`web/README.md`](./web/README.md) for:
  - UI components
  - Streaming integration
  - Thread management
  - Customization

## 🔧 Key Configuration

### Fast vs. Slow Mode

**Fast Mode** (Recommended for development):
```env
LOADER_PROCESS_PDFS=false  # Metadata + embeddings only
LOADER_WORKERS=12
LOADER_BATCH_SIZE=100
```
- ✅ ~1 hour for 500k papers
- ✅ Title & abstract semantic search
- ✅ Hybrid search enabled

**Slow Mode** (Full PDF processing):
```env
LOADER_PROCESS_PDFS=true   # Download + parse PDFs
```
- ⏱️ ~25 hours for 500k papers
- ✅ Full-text search in paper content
- ✅ Deep content retrieval

### Performance Tuning

Adjust based on your hardware:
```env
LOADER_WORKERS=12          # More workers = faster (but more RAM)
LOADER_BATCH_SIZE=100      # Larger batches = fewer operations
```

**Recommended Settings**:
- **8GB RAM**: 4 workers, batch size 50
- **16GB RAM**: 8 workers, batch size 100
- **32GB+ RAM**: 12 workers, batch size 100

## Fixing Setup Issues

### Platform Architecture Mismatch (ARM64 Mac)

**Error**: `The requested image's platform (linux/amd64) does not match the detected host platform (linux/arm64/v8)`

**Root Cause**: Elasticsearch and Kibana images are AMD64-only, but Apple Silicon Macs use ARM64 architecture.

**Solution**: Add platform specification to `docker-compose.yml`:

```yaml
elasticsearch:
  image: docker.elastic.co/elasticsearch/elasticsearch:${STACK_VERSION}
  platform: linux/amd64 # Force AMD64 architecture

kibana:
  image: docker.elastic.co/kibana/kibana:${STACK_VERSION}
  platform: linux/amd64 # Force AMD64 architecture
```

### Missing Environment Variables

**Error**: `indexCreatedVersionMajor is in the future: 10` or `CorruptIndexException`

**Root Cause**: Previous Elasticsearch runs left incompatible index data.

**Solution**: Clean up volumes and restart:

```bash
docker-compose down -v  # Remove all volumes
docker-compose up -d    # Start fresh
```

### Elasticsearch Client Version Mismatch

**Error**: `BadRequestError(400, 'media_type_header_exception', 'Accept version must be either version 8 or 7, but found 9')`

**Root Cause**: Python Elasticsearch client version 9.x is incompatible with Elasticsearch server 8.11.0.

**Solution**: Update `backend/pyproject.toml`:

```toml
# Change from:
"elasticsearch>=9.1.1",

# To:
"elasticsearch>=8.0.0,<9.0.0",
```

Then reinstall dependencies:

```bash
cd backend
uv sync
```

### Issue 5: HTTPS vs HTTP Connection

**Error**: `ConnectionError: Failed to connect to Elasticsearch after multiple attempts`

**Root Cause**: Backend `.env` file has HTTPS URL but Elasticsearch container runs on HTTP.

**Solution**: Update `backend/.env`:

```env
# Change from:
ELASTICSEARCH_URL=https://localhost:9200

# To:
ELASTICSEARCH_URL=http://localhost:9200
```
