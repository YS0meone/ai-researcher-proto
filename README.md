# AI Researcher Proto

An AI-powered research assistant built with LangGraph and React. This application helps users discover and analyze research papers through an intelligent conversational interface.

## Project Structure

This project is organized into two main components:

### 🐍 Backend (`/backend`)
- **LangGraph-powered AI agent** using OpenAI's GPT-4o-mini
- **Advanced search capabilities**: Semantic, full-text, and hybrid search
- **Vector embeddings** using sentence-transformers (all-MiniLM-L6-v2)
- **PostgreSQL + pgvector** for efficient vector similarity search
- **Python-based** with FastAPI integration via LangGraph CLI
- **Serves REST API** for the frontend to consume

### ⚛️ Frontend (`/web`)
- **React chat interface** built with Vite and TypeScript
- **Real-time messaging** with the AI agent
- **Modern UI** with Tailwind CSS and shadcn/ui components
- **Thread management** for conversation history

## Quick Start

### Prerequisites
- Python 3.12+ with [uv](https://docs.astral.sh/uv/) package manager
- Node.js 20+ with pnpm (`npm install -g pnpm`)
- OpenAI API key

### 🚀 30-Second Setup
```bash
# 1. Clone and setup backend
cd backend
uv sync
cp .env.example .env
# Edit .env and add your actual OpenAI API key

# 2. Setup frontend (in new terminal)
cd ../web
pnpm install

# 3. Start both servers
# Terminal 1: Backend
cd backend && uv run langgraph dev

# Terminal 2: Frontend  
cd web && pnpm dev

# 4. Open http://localhost:5173 and start chatting!
```

### Detailed Setup
1. **Backend setup**: See [`backend/README.md`](./backend/README.md) for complete Python environment setup
2. **Frontend setup**: See [`web/README.md`](./web/README.md) for React app configuration  
3. **Database setup**: Optional - for full search functionality with real papers

## Features

- 🤖 **Intelligent research paper discovery** with multiple search strategies
- 🧠 **Semantic search** using vector embeddings (sentence-transformers)
- 🔍 **Hybrid search** combining full-text and semantic search
- 💬 **Conversational interface** with natural language queries
- 📚 **Context-aware responses** with paper citations
- 🔄 **Real-time streaming** responses
- 📱 **Responsive design** for all devices
- 🌙 **Dark/light mode support**

## 🔍 Search Capabilities

The AI researcher offers three intelligent search strategies:

### 1. **Semantic Search** 🧠
- Uses vector embeddings to understand meaning and context
- Best for conceptual queries like "papers about attention mechanisms"
- Powered by `all-MiniLM-L6-v2` sentence transformer (384 dimensions)
- Finds papers with similar concepts even with different terminology

### 2. **Full-Text Search** 📝
- Traditional keyword-based search using PostgreSQL tsvector
- Best for specific terms, author names, or exact phrases
- Weighted search: Title (A) > Abstract (B) > Full text (C)
- Fast and precise for known terminology

### 3. **Hybrid Search** 🔄
- Combines semantic and full-text search for comprehensive results
- Configurable weighting between semantic similarity and text relevance
- Provides the best of both approaches
- Default: 70% semantic + 30% text relevance

## 🚀 Quick Testing

### Option 1: React Frontend (Recommended)
1. Start backend: `cd backend && uv run langgraph dev`
2. Start frontend: `cd web && pnpm dev`
3. Open `http://localhost:5173`
4. Connect to `http://localhost:2024` with agent ID `agent`

### Option 2: LangGraph Studio
1. Start backend: `cd backend && uv run langgraph dev`
2. Open: `https://smith.langchain.com/studio/?baseUrl=http://localhost:2024`

### Test Queries:
- **"Hello"** → Basic chat functionality
- **"Find papers about transformer architectures"** → Semantic search
- **"Search for BERT papers"** → Keyword search  
- **"Papers on attention mechanisms in NLP"** → Hybrid search

## Development Workflow

1. Start the backend (Python LangGraph server)
2. Start the frontend (React development server)
3. Open your browser and start chatting with the AI researcher!

For detailed setup instructions, check the README files in each directory.