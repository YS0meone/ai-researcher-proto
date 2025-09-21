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