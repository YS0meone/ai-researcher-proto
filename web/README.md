# AI Researcher Frontend

A React-based chat interface for the AI Researcher project. Built with Vite, TypeScript, and Tailwind CSS to provide a modern conversational interface with LangGraph agents.

**Two core modes:**
1. 🔎 **Paper Finding** - Discover relevant papers on any research topic
2. 💬 **Q&A** - Ask detailed questions about selected papers

The interface seamlessly switches between modes based on your conversation context.

## Prerequisites

- Node.js 20 or higher
- pnpm (recommended) or npm

## Quick Start

### 1. Install Dependencies

```bash
cd web
pnpm install
```

### 2. Start Development Server

```bash
pnpm dev
```

The app will be available at `http://localhost:5173`.

### 3. Connect to Backend

Make sure your backend is running on `http://localhost:2024`, then:

1. Open the web app at `http://localhost:5173`
2. Enter the following connection details:
   - **Deployment URL**: `http://localhost:2024`
   - **Assistant/Graph ID**: `agent`
   - **LangSmith API Key**: (leave empty for local development)
3. Click "Continue" to start chatting

### 4. Start Using the Interface

**Paper Finding Mode:**
- Ask broad research questions: "What are transformer architectures?"
- The agent will search, rank, and synthesize results from multiple papers
- View paper citations with metadata in the response

**Q&A Mode:**
- Select papers from search results (or provide arXiv IDs)
- Ask specific questions: "How does the attention mechanism work?"
- Get answers grounded in evidence with segment-level citations

## Available Scripts

- `pnpm dev` - Start development server
- `pnpm build` - Build for production
- `pnpm lint` - Run linting
- `pnpm format` - Format code
- `pnpm preview` - Preview production build

## Features

- 🤖 **Real-time chat** with multi-agent LangGraph backend
- 💬 **Message streaming** - See AI responses as they're generated
- 📚 **Thread management** - Conversation history and context preservation
- 🔍 **Dual-mode interface** - Seamlessly switch between paper finding and Q&A
- 📝 **Rich formatting** - Markdown rendering with syntax highlighting
- 🔗 **Citation support** - Paper references with metadata display
- 🌙 **Dark/light mode** - Theme support with system preference detection
- 📱 **Responsive design** - Works on desktop, tablet, and mobile
- 🎨 **Modern UI** - shadcn/ui components with Tailwind CSS
- ⚡ **Fast & lightweight** - Vite build tool for instant HMR
