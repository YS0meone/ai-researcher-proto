# AI Researcher Backend

A LangGraph-powered backend service that provides an AI chatbot agent using OpenAI's GPT-4o-mini model.

## Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) for dependency management
- OpenAI API key

## Quick Start

### 1. Clone and Navigate

```bash
git clone <your-repo-url>
cd ai-researcher-proto/backend
```

### 2. Install Dependencies

```bash
uv sync
```

This will create a virtual environment and install all required dependencies including:
- FastAPI
- LangChain with OpenAI support
- LangGraph and LangGraph CLI
- LangSmith
- Pydantic
- **sentence-transformers** for semantic search
- **pgvector** for PostgreSQL vector operations
- **numpy** for vector computations

### 3. Environment Setup

Copy the example environment file and add your OpenAI API key:

```bash
cp .env.example .env
```

Edit `.env` and add your actual OpenAI API key:

```env
OPENAI_API_KEY=sk-proj-your_actual_openai_api_key_here
```

**⚠️ Important**: Never commit the `.env` file to git! It contains sensitive API keys.

### 4. Run the Development Server

```bash
uv run langgraph dev --host localhost --port 2024
```

The LangGraph development server will start on `http://localhost:2024`.

**Note**: We specify `--host localhost --port 2024` explicitly to ensure:
- Proper browser compatibility (localhost vs 0.0.0.0)
- Consistent port for frontend connection
- Compatibility with LangGraph Studio

## Project Structure

```
backend/
├── app/
│   └── agent/
│       └── graph.py          # Main LangGraph agent definition
├── .env                      # Environment variables (create from .env_example)
├── .env_example             # Example environment file
├── langgraph.json           # LangGraph configuration
├── pyproject.toml           # Python dependencies and project config
└── README.md               # This file
```

## How It Works

The backend implements an intelligent research assistant using LangGraph with advanced search capabilities:

### 🧠 Core Architecture
1. **State Management**: Uses a TypedDict with a `messages` field to maintain conversation history
2. **Agent Node**: The `chatbot` function processes incoming messages and generates responses using GPT-4o-mini
3. **Graph Flow**: Messages flow from START → chatbot → tools → chatbot → END
4. **API Server**: LangGraph CLI serves the agent as a REST API with streaming support

### 🔍 Search System
The agent has access to three powerful search tools:

#### 1. **Semantic Search** (`semantic_search_papers`)
- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **Vector Database**: PostgreSQL with pgvector extension
- **Similarity**: Cosine distance for vector comparisons
- **Use Case**: Conceptual queries, finding related research

#### 2. **Full-Text Search** (`search_papers`)
- **Engine**: PostgreSQL tsvector with GIN indexing
- **Ranking**: `ts_rank` with weighted fields
- **Use Case**: Keyword searches, author names, specific terms

#### 3. **Hybrid Search** (`hybrid_search_papers`)
- **Combination**: Weighted semantic + text search
- **Default Weight**: 70% semantic + 30% text relevance
- **Use Case**: Comprehensive search with best results

### 🤖 Agent Intelligence
The agent automatically selects the best search strategy based on query type:
- Conceptual queries → Semantic search
- Specific keywords → Full-text search  
- Complex queries → Hybrid search

### Environment Variables

- `OPENAI_API_KEY` - Your OpenAI API key (required)
- `EMBEDDING_MODEL` - Sentence transformer model (default: "all-MiniLM-L6-v2")
- `EMBEDDING_DIMENSION` - Vector dimensions (default: 384)
- `DATABASE_URL` - PostgreSQL connection string (for production)
- `DATABASE_ASYNC_URL` - Async PostgreSQL connection string

### Checking Installation

Verify your setup:

```bash
# Check Python version
python --version

# Check if dependencies are installed
uv pip list

# Test the graph directly
uv run python -c "from app.agent.graph import graph; print('Graph loaded successfully')"

# Test semantic search implementation
uv run python test_semantic_search.py
```

## 🧪 Testing Semantic Search

### Without Database (Error Handling Test)
The system gracefully handles database connection failures:

```bash
# Start the server
uv run langgraph dev

# Test with curl
curl -X POST "http://localhost:2024/threads" \
  -H "Content-Type: application/json" \
  -d '{"assistant_id": "agent"}' | jq -r '.thread_id'

# Send a semantic search query (will show graceful error handling)
curl -X POST "http://localhost:2024/threads/YOUR_THREAD_ID/runs" \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "agent",
    "input": {"messages": [{"role": "user", "content": "Find papers about attention mechanisms"}]},
    "stream_mode": ["values"]
  }'
```

### Expected Behavior
- ✅ **Embedding generation works**: You'll see model loading and batch processing
- ✅ **Tool selection works**: Agent chooses appropriate search method
- ⚠️ **Database queries fail gracefully**: Provides fallback response with known papers
- ✅ **Error handling**: Clean error messages and helpful responses

### With Database (Full Functionality)
1. Set up PostgreSQL with pgvector extension
2. Run migrations: `uv run alembic upgrade head`
3. Populate papers with embeddings
4. Test all three search methods with real data

## 🔧 Troubleshooting

### Common Issues

#### "Extra inputs are not permitted" Error
If you see a Pydantic validation error about extra inputs:
```bash
ValidationError: 1 validation error for Settings
langsmith_api_key
  Extra inputs are not permitted
```

**Solution**: This means your `.env` file contains a variable not defined in `app/core/config.py`. All environment variables must be explicitly declared in the Settings class.

#### Configuration Loading Issues
Test your configuration:
```bash
# Test configuration loads
uv run python -c "from app.core.config import settings; print('Config OK')"

# Test graph loads
uv run python -c "from app.agent.graph import graph; print('Graph OK')"
```

#### Server Won't Start
1. Check your `.env` file exists and has `OPENAI_API_KEY`
2. Ensure all required environment variables have defaults in `config.py`
3. Use explicit host/port: `uv run langgraph dev --host localhost --port 2024`
