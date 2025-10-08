# AI Researcher Proto

An AI-powered research assistant built with LangGraph and React. This application helps users discover and analyze research papers through an intelligent conversational interface.

## Project Structure

This project is organized into two main components:

### 🐍 Backend (`/backend`)

- **LangGraph-powered AI agent** using OpenAI's GPT-4o-mini
- **Python-based** with FastAPI integration
- **Handles conversation logic** and research paper analysis
- **Serves REST API** for the frontend to consume

### ⚛️ Frontend (`/web`)

- **React chat interface** built with Vite and TypeScript
- **Real-time messaging** with the AI agent
- **Modern UI** with Tailwind CSS and shadcn/ui components
- **Thread management** for conversation history

## Quick Start

1. **Set up the env file for the docker compose**: copy the `.env.example` to create a `.env` in the project root and configure it based on your own need.
2. **Set up the all services**: run `docker-compose up -d` in project root.
3. **Set up the backend**: See [`backend/README.md`](./backend/README.md) for Python environment setup
4. **Set up the frontend**: See [`web/README.md`](./web/README.md) for React app setup
5. **Start developing**: The frontend automatically connects to the backend when both are running

If you have setup issues, please check the [Fixing Setup Issues](##fixing-setup-issues) section.

## Features

- 🤖 Intelligent research paper discovery
- 💬 Conversational interface
- 📚 Context-aware responses
- 🔄 Real-time streaming
- 📱 Responsive design
- 🌙 Dark/light mode support

## Development Workflow

1. Start the backend (Python LangGraph server)
2. Start the frontend (React development server)
3. Open your browser and start chatting with the AI researcher!

For detailed setup instructions, check the README files in each directory.

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
