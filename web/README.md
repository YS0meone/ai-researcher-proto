# AI Researcher Frontend

A React-based chat interface for the AI Researcher project. Built with Vite, TypeScript, and Tailwind CSS to provide a modern conversational interface with LangGraph agents.

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

## Available Scripts

- `pnpm dev` - Start development server
- `pnpm build` - Build for production
- `pnpm lint` - Run linting
- `pnpm format` - Format code
- `pnpm preview` - Preview production build

## Features

- 🤖 Real-time chat with AI agents
- 💬 Message streaming
- 📚 Thread management
- 🌙 Dark/light mode support
- 📱 Responsive design
- 🎨 Modern UI with shadcn/ui components
