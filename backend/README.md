# AI Researcher Backend

A LangGraph-powered research assistant with **dual-mode architecture**: discover relevant papers on any topic and answer detailed questions about selected papers. Powered by OpenAI GPT-4o-mini with advanced RAG (Retrieval-Augmented Generation) capabilities.

## 🎯 Current Focus: Evaluation with QASPER Dataset

**This project now focuses on evaluation using the QASPER dataset** - a benchmark for question answering on scientific papers. The QASPER dataset provides:
- **Questions**: Real questions from researchers reading NLP papers
- **Full-text papers**: Complete paper content with section structure
- **Evidence annotations**: Human-annotated evidence spans for answers
- **Multiple answer types**: Extractive, abstractive, yes/no, and unanswerable questions

**Data Pipeline Options:**
1. 🔬 **QASPER Dataset (Recommended for Evaluation)**: Load pre-processed QASPER papers with QA annotations
2. 📚 **ArXiv Data Pipeline (General Use)**: Load any papers from ArXiv metadata for broader research use

## 🌟 Features

### Multi-Agent Architecture
- **Paper Finding Mode**: Discover papers through iterative multi-strategy search
- **Q&A Mode**: Ask specific questions about selected papers with evidence retrieval
- **Intelligent Routing**: Automatically switches between modes based on context
- **Modular Design**: Separate subgraphs for paper finding and Q&A workflows

### Search & Retrieval
- **Multi-Strategy Search**: Hybrid, semantic, keyword, and vector-based search
- **Vector Search**: Qdrant-powered semantic search on 768-dimensional embeddings
- **Full-Text Search**: Elasticsearch with metadata and abstract indexing
- **Scoped Retrieval**: Vector search within selected papers for Q&A
- **Smart Reranking**: LLM-based relevance ranking with coverage assessment

### Data Pipeline
- **Parallel Processing**: Multi-worker pipeline with 12 workers and batch processing
- **Fast & Slow Modes**: Metadata-only (1hr for 500k) or full PDF parsing (25hrs for 500k)
- **GROBID Integration**: Full-text extraction from PDFs for deep content search
- **Segment-Level Indexing**: Stores paper chunks for precise evidence retrieval

## 📋 Prerequisites

- **Python 3.12+** with [uv](https://docs.astral.sh/uv/) for dependency management
- **Docker & Docker Compose** for services (Elasticsearch, Qdrant, Kibana, GROBID)
- **DeepSeek API key** (or OpenAI API key) for LLM inference
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
- **langchain-deepseek** - DeepSeek LLM integration
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
# Required - LLM Configuration
OPENAI_API_KEY=sk-your-deepseek-api-key-here  # DeepSeek API key (uses same variable)
MODEL_NAME=deepseek-chat                      # Use deepseek-chat or gpt-4o-mini

# Database URLs (default for Docker)
ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_INDEX=papers
ELASTICSEARCH_USERNAME=elastic
ELASTICSEARCH_PASSWORD=elastic

QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=papers
QDRANT_VECTOR_SIZE=768
QDRANT_DISTANCE=COSINE

# Optional: ArXiv Paper Loader (for general use beyond QASPER)
LOADER_WORKERS=12              # Number of parallel workers
LOADER_BATCH_SIZE=100          # Papers per batch
LOADER_PROCESS_PDFS=false      # Fast mode (metadata only)
LOADER_ARXIV_METADATA_PATH=/path/to/arxiv-metadata-oai-snapshot.json  # Only needed for ArXiv pipeline
```

**Note**: For QASPER evaluation, you only need `OPENAI_API_KEY` and the database URLs. The loader settings are optional and only used for the general ArXiv data pipeline.

### 4. (Optional) Download ArXiv Dataset

**Skip this step if using QASPER** - The QASPER dataset is already included in `backend/eval/data/`.

For general research use beyond QASPER, download the ArXiv metadata snapshot from [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv/data):

1. Download `arxiv-metadata-oai-snapshot.json` (~3.5GB)
2. Place it in a known location
3. Update `LOADER_ARXIV_METADATA_PATH` in `.env`

**Note**: The ArXiv pipeline filters for Computer Science papers (all 40 `cs.*` categories), extracting ~100k-150k papers from 500k lines.

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

### 6. Load Dataset

**Option A: QASPER Dataset (Recommended for Evaluation)** ⭐

The QASPER dataset is the primary focus for evaluation and testing the Q&A capabilities:

```bash
cd backend
uv run python -m eval.load_qasper
```

**What this does:**
- ✅ Loads **1,585 papers** from QASPER benchmark (train + validation + test)
- ✅ Indexes full paper text with section structure to Elasticsearch
- ✅ Creates paragraph-level embeddings (768-dim) for semantic search
- ✅ Enables segment-level retrieval for Q&A evaluation
- ✅ Includes **5,049 questions** with human-annotated answers
- ⏱️ Takes **~10-15 minutes** for full dataset

**QASPER Dataset Structure:**
- **Papers**: NLP research papers with full text and section structure
- **Questions**: Multi-hop questions requiring reasoning across paper sections
- **Answers**: Include evidence spans, free-form answers, and extractive spans
- **Annotations**: Multiple annotators per question for answer validation

**Data Location:**
- Parquet files in `backend/eval/data/`:
  - `train.parquet` - 888 papers, 2,593 questions
  - `validation.parquet` - 281 papers, 1,005 questions  
  - `test.parquet` - 416 papers, 1,451 questions

---

**Option B: ArXiv Data Pipeline (General Use)**

For broader research use beyond QASPER evaluation:

```bash
cd backend
uv run python -m app.data_pipeline
```

**Fast Mode** (Metadata + Embeddings Only):
- ✅ Process ~500k lines in **~1 hour**
- ✅ Index metadata (title, abstract, authors) to Elasticsearch
- ✅ Create title & abstract embeddings for semantic search
- ❌ Skip PDF downloads and full-text extraction

**Slow Mode** (Full PDF Processing):
Set `LOADER_PROCESS_PDFS=true` in `.env`, then:
- ✅ Download PDFs from ArXiv (parallel)
- ✅ Parse with GROBID for full text
- ✅ Index full document content to Qdrant
- ⏱️ Takes **~25 hours** for 500k lines

**Note**: The ArXiv pipeline is for general paper discovery. For Q&A evaluation, use QASPER.

### 7. Verify Data Loading

**Check QASPER data:**

```bash
# Check Elasticsearch index
curl http://localhost:9200/arxiv_papers/_count

# Check Qdrant collection
curl http://localhost:6333/collections/arxiv_papers

# Should see ~1,585 papers indexed from QASPER
```

**Test Q&A retrieval:**

```bash
uv run python -c "
from app.tools.search import vector_search_papers_by_ids_impl
# Example: Search within a QASPER paper
results = vector_search_papers_by_ids_impl(
    query='attention mechanism',
    ids=['1706.03762'],  # Example paper ID
    limit=5
)
print(f'Found {len(results)} segments')
for r in results:
    print(f\"Score: {r['similarity_score']:.3f} - {r['supporting_detail'][:100]}...\")
"
```

### 8. Start LangGraph Server

```bash
uv run langgraph dev
```

The server starts on `http://localhost:2024` with the `agent` graph available.

## 📁 Project Structure

```
backend/
├── app/
│   ├── agent/
│   │   ├── graph.py           # Main router - orchestrates paper finder & QA
│   │   ├── paper_finder.py    # Paper search subgraph
│   │   ├── qa.py              # Q&A subgraph for selected papers
│   │   ├── states.py          # Shared state definitions
│   │   ├── prompts.py         # Structured prompts with CoT reasoning
│   │   └── utils.py           # Helper functions
│   ├── tools/
│   │   └── search.py          # Search tools (hybrid, semantic, keyword, vector)
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

The research agent uses a **modular multi-agent architecture** with three main components:

### Main Router (`graph.py`)

Entry point that orchestrates the entire workflow:
- Analyzes conversation context and user intent
- Routes to appropriate subgraph (paper finder or Q&A)
- Manages state transitions between modes
- Coordinates iterative search refinement

**Routing Logic:**
```python
if user_has_selected_papers and asking_specific_question:
    route_to → QA_SUBGRAPH
elif needs_paper_discovery:
    route_to → PAPER_FINDER_SUBGRAPH
else:
    route_to → SYNTHESIS
```

---

### Paper Finder Subgraph (`paper_finder.py`)

**Purpose:** Discover relevant papers through intelligent search

**Workflow:**
1. **Search Agent Node**
   - Analyzes user query to determine optimal search strategy
   - Generates multiple focused queries for comprehensive coverage
   - Selects appropriate tools (hybrid, semantic, keyword, vector)
   - Returns single AIMessage with multiple tool calls

2. **Tool Node** - Executes searches in parallel:
   - `hybrid_search_papers`: Combines text + semantic (RRF scoring)
   - `semantic_search_papers`: Vector similarity (allenai-specter)
   - `keyword_search_papers`: Full-text search (Elasticsearch BM25)
   - `vector_search_papers`: Segment-level vector search

3. **Merge & Rerank Node**
   - Deduplicates by arXiv ID across all search results
   - LLM reranks top 30 papers by true relevance
   - Calculates coverage score (0.0-1.0) for iteration control
   - Returns top 50 papers

4. **Iteration Control**
   - If coverage < 0.65: Refine search with new queries (max 3 iterations)
   - Tracks query history to avoid redundant searches
   - If coverage ≥ 0.65 or max iterations: Proceed to synthesis

**Graph Flow:**
```
START → search_agent → tools → merge_and_rerank
            ↓                          ↓
        (iterate) ← increment_iter ← decide_next
                                       ↓
                                  synthesize → END
```

---

### Q&A Subgraph (`qa.py`)

**Purpose:** Answer specific questions about user-selected papers

**Workflow:**
1. **Retrieval Node** (`qa_retrieve`)
   - Generates 1-3 focused search queries from user question
   - Executes vector search scoped to selected papers only
   - Returns top 10 unique segments with similarity scores

2. **Quality Assessment** (`qa_assess_quality`)
   - Evaluates if evidence is sufficient to answer
   - Checks segment quality (similarity scores, count)
   - LLM confidence scoring for answer quality
   - Decides: answer / refine / insufficient

3. **Refinement Node** (`qa_refine_retrieval`) - If needed
   - Generates alternative queries with different phrasings
   - Lower similarity threshold for broader recall
   - Merges with previous results

4. **Answer Generation** (`qa_answer`)
   - Synthesizes answer grounded in retrieved evidence
   - Cites sources with segment-level references
   - Acknowledges limitations and missing information
   - Formats as: [Paper: arxiv_id, Section: "quote"]

**Graph Flow:**
```
START → qa_retrieve → qa_assess → qa_answer → END
                          ↓
                      qa_refine → qa_answer → END
                      (if needed)
```

---

### State Management (`states.py`)

Shared `State` TypedDict across all agents:

**Paper Finding State:**
- `messages`: Conversation history (with `add_messages` annotation)
- `papers`: Retrieved papers from search
- `search_queries`: Query history for iteration control
- `iter`: Current search iteration (max 3)
- `coverage_score`: How well papers cover the query (0.0-1.0)
- `route`: Current routing decision (search/synthesize/qa)

**Q&A State:**
- `selected_ids`: Paper arXiv IDs selected for Q&A
- `retrieved_segments`: Vector search results from selected papers
- `qa_query`: The specific question for Q&A mode

**Structured Outputs:**
- `RouterDecision`: Search vs. synthesize vs. QA routing
- `SearchPlan`: Strategy + queries + tool calls (1-5 tools)
- `RerankingResult`: Deduplicated + scored papers + coverage
- `SynthesisDecision`: Deep search decision
- `RetrievalPlan`: Focused queries for evidence retrieval (1-3)
- `AnswerQuality`: Sufficiency assessment + confidence + refinement

---

### Search Tools (`tools/search.py`)

**Paper Discovery Tools:**
1. `hybrid_search_papers(query, limit)` - Text + semantic hybrid
2. `semantic_search_papers(query, limit)` - Pure vector similarity
3. `keyword_search_papers(query, limit)` - Elasticsearch full-text
4. `vector_search_papers(query, limit)` - Segment-level retrieval

**Q&A Tools:**
5. `vector_search_papers_by_ids(query, ids, limit)` - Scoped to selected papers only

**Design Pattern:**
- All tools return `List[Dict[str, Any]]` for consistent handling
- Include relevance scores (`search_score`, `similarity_score`, `text_score`)
- Metadata: title, abstract, authors, arxiv_id, categories
- Q&A results include `supporting_detail` field for evidence segments

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

### Paper Loader Configuration (Optional - ArXiv Pipeline Only)
- `LOADER_WORKERS` - Parallel workers (default: `12`)
- `LOADER_BATCH_SIZE` - Papers per batch (default: `100`)
- `LOADER_PROCESS_PDFS` - Enable PDF processing (default: `false`)
- `LOADER_ARXIV_METADATA_PATH` - Path to ArXiv metadata JSON (not needed for QASPER)

## 📊 Evaluation with QASPER

The QASPER dataset enables systematic evaluation of the Q&A system:

### Dataset Statistics

```
Total Papers: 1,585 NLP research papers
Total Questions: 5,049 questions

Split breakdown:
- Training:    888 papers, 2,593 questions
- Validation:  281 papers, 1,005 questions
- Test:        416 papers, 1,451 questions

Question types:
- Extractive: Answers are spans from the paper
- Abstractive: Answers require synthesis across sections
- Yes/No: Binary questions with evidence
- Unanswerable: Questions that cannot be answered from the paper
```

### Evaluation Workflow

```python
# 1. Load QASPER papers into database
uv run python -m eval.load_qasper

# 2. Test Q&A on a specific paper
from eval.load_qasper import load_qasper_to_db
import pandas as pd

# Load validation set
val_df = pd.read_parquet("eval/data/validation.parquet")

# Example paper and questions
paper = val_df.iloc[0]
paper_id = paper["id"]
questions = paper["qas"]["question"]
answers = paper["qas"]["answers"]

print(f"Paper: {paper['title']}")
print(f"Questions: {len(questions)}")

# 3. Run Q&A agent on each question
# (Implementation in evaluation script)

# 4. Compare agent answers with gold annotations
# Metrics: F1, Exact Match, Evidence Overlap
```

### Why QASPER for Evaluation?

1. **Full-text papers**: Tests segment-level retrieval, not just abstracts
2. **Complex questions**: Multi-hop reasoning across sections
3. **Evidence annotations**: Ground truth for retrieval quality
4. **Multiple annotators**: Robust answer validation
5. **Scientific domain**: Matches the target use case (research papers)

### QASPER vs ArXiv Pipeline

| Feature | QASPER | ArXiv Pipeline |
|---------|--------|----------------|
| **Purpose** | Q&A evaluation | General paper discovery |
| **Papers** | 1,585 NLP papers | 100k+ CS papers |
| **Full text** | ✅ Included | ⚠️ Requires PDF parsing |
| **Questions** | ✅ 5,049 annotated | ❌ None |
| **Load time** | ~10-15 minutes | ~1-25 hours |
| **Use case** | Evaluation & testing | Production research tool |

## 💡 Usage Examples

### Paper Finding Mode

**Use Case:** Discover papers on a research topic

```python
# User queries that trigger paper finding:
"What are recent advances in retrieval-augmented generation?"
"Find papers about efficient transformers"
"How do vision transformers compare to CNNs?"

# Agent workflow:
# 1. Router → Decides: "search" mode
# 2. Search Agent → Plans multi-strategy search
#    - hybrid_search("RAG advances")
#    - semantic_search("retrieval augmented generation")
#    - vector_search("combining retrieval with LLMs")
# 3. Tools → Execute parallel searches
# 4. Merge & Rerank → Dedupe + rank by relevance
# 5. Coverage check → If < 0.65, iterate with refined queries
# 6. Synthesis → Generate answer citing top 10 papers
```

**Output:**
```
Based on recent research, retrieval-augmented generation (RAG) combines...

Key papers:
1. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
   (Lewis et al., 2020) [arXiv:2005.11401]
   
2. "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"
   (Asai et al., 2023) [arXiv:2310.11511]
   
[... 8 more papers with summaries ...]
```

---

### Q&A Mode

**Use Case:** Ask specific questions about selected papers

```python
# Prerequisites:
# 1. User has selected papers (e.g., from previous search)
# 2. selected_ids = ["2005.11401", "2310.11511"]

# User queries that trigger Q&A:
"How does the attention mechanism work in this paper?"
"What were the experimental results on GLUE benchmark?"
"What are the limitations mentioned by the authors?"

# Agent workflow:
# 1. Router → Decides: "qa" mode (sees selected_ids in state)
# 2. QA Retrieve → Generates focused queries:
#    - "attention mechanism computation"
#    - "multi-head attention formula"
# 3. Vector search scoped to selected papers only
# 4. Quality assessment → Check if evidence sufficient
# 5. If insufficient → Refine with alternative queries
# 6. Answer generation → Grounded in retrieved segments
```

**Output:**
```
The attention mechanism in this paper uses scaled dot-product attention:

Attention(Q, K, V) = softmax(QK^T / √d_k)V

[Paper: 1706.03762, Section: "3.2.1 Scaled Dot-Product Attention"]

The authors compute attention scores by taking the dot product of queries 
and keys, scaling by the square root of the key dimension, then applying 
softmax to get weights. These weights are used to compute a weighted sum 
of the values.

[Paper: 1706.03762, Section: "The scaled dot-product attention is more 
computationally efficient..."]

Key advantages mentioned:
- Faster computation compared to additive attention
- Better parallelization on modern hardware
- Scales to longer sequences

Limitations acknowledged:
- Fixed attention span (limited by context window)
- Quadratic complexity in sequence length
```

---

### Switching Between Modes

The router automatically detects mode based on context:

```python
# State-based routing:
if state.get("selected_ids") and is_specific_question(query):
    # Has selected papers + asking detailed question
    → Route to Q&A mode
    
elif is_research_query(query):
    # Broad research question, no papers selected
    → Route to Paper Finding mode
    
else:
    # Already has papers from previous search
    → Route to Synthesis (summarize existing results)
```

**Example Conversation:**
```
User: "Find papers about transformers"
Agent: [Paper Finding Mode] → Returns 10 papers

User: "Tell me more about paper 3"
Agent: [Synthesis Mode] → Summarizes paper 3 from existing results

User: "How does the attention mechanism work?"
Agent: [Q&A Mode] → Vector search in paper 3 for attention details

User: "Find more papers about attention mechanisms"
Agent: [Paper Finding Mode] → New search iteration
```

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
**Issue**: Embedding model initialization with PyTorch meta device  
**Solution**: Fixed in latest version. Restart LangGraph server to reload the embedding model with proper device settings.

### Windows Multiprocessing Issues
**Issue**: `data_pipeline.py` fails with import errors on Windows  
**Solution**: Use `simple_load.py` instead - a single-process loader designed for Windows compatibility:
```bash
uv run python simple_load.py
```

### "Search results are 0" / Empty Results
**Issue**: No papers loaded in Elasticsearch  
**Solution**: 
1. Check paper count: `curl http://localhost:9200/papers/_count`
2. If count is 0, run the data loader: `uv run python simple_load.py`
3. Wait for papers to be indexed (check progress bar)

### DeepSeek API Validation Errors
**Issue**: `ValidationError for SearchPlan` - missing query fields  
**Solution**: Already fixed with fallback handling. The agent will use a simple hybrid search if structured output fails.

### Slow Data Pipeline
**Issue**: PDF processing is very slow  
**Solution**: Use fast mode (`LOADER_PROCESS_PDFS=false`) for initial loading - only indexes metadata and embeddings

### Elasticsearch Connection Failed
**Issue**: Docker container not running  
**Solution**: 
```bash
docker-compose up -d elasticsearch
curl http://localhost:9200  # Verify connection
```

### Qdrant Collection Not Found
**Issue**: Data pipeline hasn't run yet or failed  
**Solution**: Run the data loader to create collections and index papers

### Model Loading Takes Too Long
**Issue**: First request is slow due to model initialization  
**Solution**: The `allenai-specter` model (~500MB) downloads and loads on first use. Subsequent requests use the singleton instance and are fast.

## 📚 Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [DeepSeek API Documentation](https://platform.deepseek.com/api-docs/)
- [Elasticsearch Python Client](https://elasticsearch-py.readthedocs.io/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [GROBID Documentation](https://grobid.readthedocs.io/)

## 🤖 LLM Configuration

### DeepSeek (Default)
```env
MODEL_NAME=deepseek-chat
OPENAI_API_KEY=sk-your-deepseek-api-key
```

**Available Models**:
- `deepseek-chat` - General purpose chat (recommended)
- `deepseek-coder` - Optimized for code generation
- `deepseek-reasoner` - Enhanced reasoning capabilities

**Benefits**:
- 💰 Much cheaper than OpenAI (~1/10th the cost)
- 🌏 Better Chinese language support
- ⚡ Fast inference speed

### OpenAI (Alternative)
```env
MODEL_NAME=gpt-4o-mini
OPENAI_API_KEY=sk-your-openai-api-key
```

**Available Models**:
- `gpt-4o-mini` - Fast and affordable
- `gpt-4o` - Most capable
- `gpt-4-turbo` - Balanced performance

Both providers work identically with the same codebase thanks to LangChain's unified interface.
