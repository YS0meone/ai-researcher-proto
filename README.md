# AI Researcher Proto

An AI-powered research assistant that helps users **discover academic papers** and **answer deep questions** about them through intelligent multi-agent orchestration. Built with LangGraph, React, and advanced RAG (Retrieval-Augmented Generation) capabilities.

**Two core modes:**
1. 🔎 **Paper Finding** - Discover relevant papers on any research topic with iterative refinement
2. 💬 **Q&A** - Ask detailed questions about selected papers with evidence-based answers

Search across papers with hybrid text + semantic search, powered by Elasticsearch and Qdrant vector database.

**🎯 Current Focus:** Evaluation with the **QASPER dataset** - a benchmark for question answering on 1,585 NLP research papers with 5,049 annotated questions.

## 🌟 Features

### 🔍 Intelligent Paper Search & Q&A
- **Multi-Strategy Search**: Hybrid (text + semantic), pure semantic, keyword, and vector-based
- **768-Dimensional Embeddings**: Using `allenai-specter` model for scientific papers
- **Vector Database**: Qdrant for semantic search on 100k+ papers with segment-level retrieval
- **Full-Text Search**: Elasticsearch with metadata and abstract indexing
- **Parallel Search**: Multiple search strategies executed simultaneously
- **Deep Q&A**: Vector search within selected papers for answering specific questions
- **Retrieval Refinement**: Automatic query refinement if initial evidence is insufficient

### 🤖 LangGraph Agent Orchestration
- **Multi-Mode Architecture**: Separate subgraphs for paper finding and Q&A
- **Intelligent Routing**: Automatically decides between search, synthesis, or Q&A modes
- **Paper Finding**: Router → Search Agent → Tools → Reranking → Synthesis
- **Q&A System**: Retrieval → Quality Assessment → Refinement → Answer Generation
- **Structured Output**: Pydantic models with chain-of-thought reasoning
- **Tool Calling**: Proper LangGraph pattern with single AIMessage + multiple tool calls
- **Iteration Control**: Automatic search refinement based on coverage scores
- **Citation Generation**: Answers grounded in retrieved evidence with paper references

### ⚡ Data Loading Options
- **QASPER Dataset**: Pre-processed benchmark with 1,585 papers + 5,049 QA pairs (~10-15 min load)
- **ArXiv Pipeline**: General-purpose loader for broader research use
  - **Parallel Processing**: 12 workers with batch processing (100 papers/batch)
  - **Fast Mode**: Metadata + embeddings only (~1 hour for 500k papers)
  - **Slow Mode**: Full PDF parsing with GROBID (~25 hours for 500k papers)
  - **Progress Monitoring**: Real-time stats with tqdm progress bars

### 💬 Modern Chat Interface
- **React 19** with TypeScript and Vite
- **Real-time Streaming**: Live message updates from LangGraph
- **Thread Management**: Conversation history and context
- **shadcn/ui Components**: Beautiful, accessible UI primitives
- **Dark/Light Mode**: Tailwind CSS with theme support

## 📋 Prerequisites

- **Python 3.12+** with [uv](https://docs.astral.sh/uv/) - Fast Python package manager
- **Node.js 20+** with pnpm - Frontend dependencies
- **Docker & Docker Compose** - For Elasticsearch, Qdrant, Kibana, GROBID
- **16GB+ RAM** recommended for parallel processing
- **OpenAI API Key** - For GPT-4o-mini inference in agents

### Technology Stack

**Backend:**
- LangGraph 0.6+ - Multi-agent orchestration
- LangChain - LLM integration and tools
- FastAPI - Web framework
- Pydantic - Data validation and structured outputs
- Elasticsearch 8.11 - Full-text search
- Qdrant - Vector database
- sentence-transformers - allenai-specter embeddings
- GROBID - PDF parsing for full-text extraction

**Frontend:**
- React 19 - UI framework
- TypeScript - Type safety
- Vite - Build tool
- Tailwind CSS - Styling
- shadcn/ui - UI components
- LangGraph SDK - Real-time streaming from backend

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
- Your `OPENAI_API_KEY` (required)
- *(Optional)* Path to ArXiv metadata JSON (`LOADER_ARXIV_METADATA_PATH`) - only needed for ArXiv pipeline, not QASPER

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

**Option A: QASPER Dataset (Recommended for Evaluation)** ⭐

```bash
cd backend
uv sync                              # Install Python dependencies
uv run python -m eval.load_qasper   # Load QASPER dataset (~10-15 minutes)
```

This loads **1,585 papers** with **5,049 questions** from the QASPER benchmark for Q&A evaluation.

**Option B: ArXiv Dataset (General Use)**

Download ArXiv metadata from [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv/data), then:

```bash
cd backend
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
3. **Paper Finding Mode** - Ask research questions:
   - "What are transformer architectures?"
   - "Find papers about few-shot learning"
   - "Recent work on retrieval-augmented generation"
4. **Q&A Mode** - Ask questions about specific papers:
   - After finding papers, select some and ask detailed questions
   - "How does the attention mechanism work in this paper?"
   - "What were the main experimental results?"
   - "What datasets were used for evaluation?"

## 📁 Project Structure

```
ai-researcher-proto/
├── backend/                    # Python LangGraph backend
│   ├── app/
│   │   ├── agent/             # LangGraph agent modules
│   │   │   ├── graph.py       # Main router and orchestration
│   │   │   ├── paper_finder.py # Paper search subgraph
│   │   │   ├── qa.py          # Q&A subgraph for selected papers
│   │   │   ├── states.py      # Shared state definitions
│   │   │   ├── prompts.py     # System prompts for all agents
│   │   │   └── utils.py       # Helper functions
│   │   ├── tools/             # Search tools
│   │   │   └── search.py      # Hybrid, semantic, keyword, vector search
│   │   ├── services/          # External services
│   │   │   ├── elasticsearch.py # Full-text search
│   │   │   ├── qdrant.py      # Vector database
│   │   │   └── paper_loader.py # Data ingestion
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

## 🏗️ Agent Architecture

The system uses a **modular multi-agent architecture** with three main components:

### 1. Main Router (`graph.py`)
- Entry point that decides between paper finding, synthesis, or Q&A modes
- Routes based on conversation context and user intent
- Manages overall workflow orchestration

### 2. Paper Finder Subgraph (`paper_finder.py`)
- **Search Agent**: Plans and executes multi-strategy paper searches
- **Tools Node**: Executes parallel searches (hybrid, semantic, keyword, vector)
- **Merge & Rerank**: Deduplicates and ranks papers by relevance
- **Iteration Control**: Refines search if coverage is insufficient (max 3 iterations)
- **Output**: Top 50 ranked papers with coverage score

### 3. Q&A Subgraph (`qa.py`)
- **Retrieval Node**: Vector search within user-selected papers
- **Quality Assessment**: Evaluates if evidence is sufficient to answer
- **Refinement Node**: Generates alternative queries if needed
- **Answer Generation**: Produces grounded answers with citations
- **Use Case**: Deep dive into specific papers with detailed questions

### State Management (`states.py`)
Shared state between all agents:
- `messages`: Conversation history
- `papers`: Retrieved papers from search
- `search_queries`: Query history for iteration control
- `selected_ids`: Papers selected for Q&A
- `retrieved_segments`: Vector search results for Q&A
- `coverage_score`: How well papers cover the query

## 🎯 Detailed Setup Guides

- **Backend**: See [`backend/README.md`](./backend/README.md) for:
  - Data pipeline configuration
  - Search tool details
  - Agent architecture details
  - Performance tuning
  - Troubleshooting

- **Frontend**: See [`web/README.md`](./web/README.md) for:
  - UI components
  - Streaming integration
  - Thread management
  - Customization

## 🛠️ Usage Modes

### Paper Finding Mode
**Use when:** You want to discover papers on a topic

**Flow:**
1. User asks a research question
2. Router → Search Agent → Multiple parallel searches
3. Merge & Rerank → Coverage assessment
4. If coverage < 65%: Iterate with refined queries (max 3 iterations)
5. Synthesis → Generate comprehensive answer with top 10 paper citations

**Example queries:**
- "What are recent advances in retrieval-augmented generation?"
- "Find papers about efficient transformers"
- "How do vision transformers compare to CNNs?"

### Q&A Mode
**Use when:** You want to ask specific questions about selected papers

**Flow:**
1. User selects papers (from previous search or provides arXiv IDs)
2. Router → Q&A Retrieve → Vector search within selected papers
3. Quality Assessment → Check if evidence is sufficient
4. If insufficient: Refinement → Generate alternative queries
5. Answer Generation → Grounded answer with segment-level citations

**Example queries:**
- "How does the attention mechanism work in this paper?"
- "What were the experimental results on GLUE benchmark?"
- "What are the limitations mentioned by the authors?"
- "How does this model handle long sequences?"

## 📊 QASPER Evaluation Dataset

The project now focuses on the **QASPER dataset** for systematic Q&A evaluation:

### What is QASPER?

QASPER (Question Answering on Scientific Papers) is a benchmark dataset for evaluating question answering systems on research papers:

- **1,585 NLP research papers** with full text and section structure
- **5,049 questions** from researchers reading these papers
- **Human-annotated answers** with evidence spans and free-form responses
- **Multiple question types**: Extractive, abstractive, yes/no, unanswerable
- **Train/Val/Test splits**: 888/281/416 papers, 2,593/1,005/1,451 questions

### Why QASPER?

✅ **Full-text papers** - Tests segment-level retrieval, not just abstracts  
✅ **Complex questions** - Multi-hop reasoning across sections  
✅ **Evidence annotations** - Ground truth for retrieval quality  
✅ **Scientific domain** - Matches target use case (research papers)  
✅ **Fast loading** - Pre-processed parquet files, 10-15 min setup  

### Loading QASPER

```bash
cd backend
uv run python -m eval.load_qasper
```

Data is located in `backend/eval/data/`:
- `train.parquet` (888 papers)
- `validation.parquet` (281 papers)
- `test.parquet` (416 papers)

## 🔧 Key Configuration

### ArXiv Data Pipeline (General Use)

For broader research beyond QASPER evaluation:

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

## 🔍 Search Tools Overview

The system provides multiple search strategies optimized for different use cases:

### 1. Hybrid Search (`hybrid_search_papers`)
- **Combines:** Elasticsearch full-text + Qdrant vector similarity
- **Best for:** General queries where both keywords and semantic meaning matter
- **Example:** "transformer attention mechanisms for NLP"

### 2. Semantic Search (`semantic_search_papers`)
- **Uses:** Pure vector similarity with allenai-specter embeddings
- **Best for:** Conceptual queries, finding similar work
- **Example:** "papers similar to BERT in methodology"

### 3. Keyword Search (`keyword_search_papers`)
- **Uses:** Elasticsearch full-text search with BM25 ranking
- **Best for:** Specific terms, author names, exact phrases
- **Example:** "Yoshua Bengio attention mechanisms"

### 4. Vector Search (`vector_search_papers`)
- **Uses:** Qdrant vector search with segment-level retrieval
- **Best for:** Finding relevant paper sections for synthesis
- **Returns:** Paper segments with supporting details

### 5. Vector Search by IDs (`vector_search_papers_by_ids`)
- **Uses:** Scoped vector search within selected papers only
- **Best for:** Q&A mode - finding evidence in specific papers
- **Returns:** Relevant segments with similarity scores

The agent automatically selects and combines these tools based on your query!

## 💡 Key Features Explained

### Iterative Search Refinement
- System evaluates search quality with a **coverage score** (0.0-1.0)
- If coverage < 0.65, automatically refines search with new queries
- Max 3 iterations to find comprehensive results
- Prevents redundant queries by tracking search history

### Smart Reranking
- Takes top 100 papers from all searches
- Uses LLM to rerank top 30 based on relevance to query
- Considers both search scores and semantic relevance
- Returns top 50 papers sorted by true relevance

### Evidence-Based Synthesis
- **Decision phase:** Determines if deep content search is needed
- **Content retrieval:** Vector search for specific evidence
- **Answer generation:** Grounded in retrieved paper segments
- **Citations:** Every claim linked to specific papers

### Retrieval Quality Assessment
- Evaluates if evidence is sufficient to answer question
- Automatic query refinement if evidence is weak
- Confidence scoring for answer quality
- Transparent about limitations and missing information

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
