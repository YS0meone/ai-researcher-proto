# AI Researcher Architecture

## System Overview

An AI-powered research assistant that helps users discover and analyze academic papers through intelligent conversation. Built with LangGraph for agent orchestration and React for the chat interface.

## Core Architecture

### Backend (`/backend`) - AI Agent & Data Layer

- **LangGraph Agent**: Multi-step reasoning with search → synthesis workflow
- **Dual Database**: PostgreSQL (structured) + Elasticsearch (search/vectors)
- **Search Pipeline**: ArXiv paper ingestion with SPECTER embeddings
- **API**: LangGraph CLI serves REST endpoints with streaming

### Frontend (`/web`) - Chat Interface

- **React Chat**: Real-time streaming with LangGraph SDK
- **Thread Management**: Conversation history and state
- **UI Components**: shadcn/ui with Tailwind CSS styling

## Agent Workflow (LangGraph)

```
User Query → Router → Search Loop → Synthesis → Response
     ↓         ↓         ↓           ↓
  [Search]  [Generate] [Execute]  [Ground in]
  [Synthesize] [Queries] [Tools]  [Papers]
```

### State Management

- **Messages**: Conversation history with tool calls
- **Papers**: Search results with metadata and scores
- **Iteration Control**: Multi-round search with coverage scoring
- **Routing**: Search vs. synthesis decision logic

### Search Strategy

- **Hybrid Search**: Text + semantic similarity (primary)
- **Semantic Search**: Vector similarity for concepts
- **Keyword Search**: Exact text matching
- **Category Browse**: Domain-specific exploration
- **Paper Details**: Full metadata retrieval

## Data Flow

### Paper Ingestion

1. **ArXiv Metadata** → PostgreSQL (structured storage)
2. **Full Text** → Elasticsearch (search index)
3. **SPECTER Embeddings** → Vector fields (semantic search)
4. **Categories** → Filtering and browsing

### Search Execution

1. **Query Generation**: LLM creates diverse search queries
2. **Tool Selection**: Agent chooses appropriate search method
3. **Result Merging**: Deduplication and scoring
4. **Reranking**: LLM-based relevance assessment
5. **Coverage Check**: Determine if more search needed

## Key Components

### Backend Services

- **`app/agent/graph.py`**: Core LangGraph agent definition
- **`app/tools/search.py`**: Search tool implementations
- **`app/services/elasticsearch.py`**: Search service with hybrid capabilities
- **`app/db/models.py`**: PostgreSQL schema and full-text search
- **`app/data_pipeline.py`**: Paper ingestion and indexing

### Frontend Architecture

- **`src/components/thread/`**: Chat interface components
- **`src/providers/Stream.tsx`**: Real-time streaming with LangGraph SDK
- **`src/providers/Thread.tsx`**: Thread state management
- **Message Types**: Human, AI, tool-calls, interrupts

## Configuration

### Environment Variables

- **OpenAI API**: LLM access
- **Database URLs**: PostgreSQL connection strings
- **Elasticsearch**: Search cluster configuration
- **Paper Loader**: ArXiv dataset and processing settings

### Development Setup

- **Backend**: `uv run langgraph dev` (port 2024)
- **Frontend**: `pnpm dev` (port 5173)
- **Data Pipeline**: Docker Compose + Python ingestion

## Extension Points

### Adding New Search Methods

1. Implement tool in `app/tools/search.py`
2. Add to agent's tool list in `graph.py`
3. Update search strategy in agent logic

### UI Enhancements

1. Create components in `src/components/`
2. Follow shadcn/ui patterns
3. Update thread state management as needed

### Agent Logic Changes

1. Modify nodes/edges in `graph.py`
2. Update state schema if needed
3. Test with `uv run python -c "from app.agent.graph import graph"`

## Technology Stack

- **Agent**: LangGraph + LangChain + OpenAI GPT-4o-mini
- **Search**: Elasticsearch + SPECTER embeddings
- **Database**: PostgreSQL with full-text search
- **Frontend**: React 19 + TypeScript + Vite
- **Styling**: Tailwind CSS + shadcn/ui
- **Streaming**: LangGraph SDK with real-time updates
