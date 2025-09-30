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

### 3. Environment Setup

Copy the example environment file and add your OpenAI API key:

```bash
cp .env_example .env
```

Edit `.env` and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Run the Development Server

```bash
uv run langgraph dev
```

The LangGraph development server will start on `http://localhost:2024` by default.


### 5. Run the data pipeline

First make sure you have copy the `.env.example` file in both the backend and project root and create your own `.env` file. Especially, make sure you have the arxiv dataset downloaded and its path configured in the environment variable.

Then start the elastic search and grobid container using

```bash
docker-compose up -d
```

If it doesn't work, try

```bash
docker compose up -d
```

In the backend folder run

```bash
uv run python -m app.data_pipeline
```

Your elasticsearch index should be created. And the papers should be index and embedded into the data store of elastic search. (Notice, the paper loader for metadata loading is currently single threaded, the total amount of paper that is gonna be fetched from the arxiv dataset is based on: batch_size * worker. It would be migrated to concurrent version in the future.)

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

The backend implements a simple chatbot agent using LangGraph:

1. **State Management**: Uses a TypedDict with a `messages` field to maintain conversation history
2. **Agent Node**: The `chatbot` function processes incoming messages and generates responses using GPT-4o-mini
3. **Graph Flow**: Messages flow from START → chatbot → END
4. **API Server**: LangGraph CLI serves the agent as a REST API with streaming support

### Environment Variables

- `OPENAI_API_KEY` - Your OpenAI API key (required)

### Checking Installation

Verify your setup:

```bash
# Check Python version
python --version

# Check if dependencies are installed
uv pip list

# Test the graph directly
uv run python -c "from app.agent.graph import graph; print('Graph loaded successfully')"
```
