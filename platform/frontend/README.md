# RAG Evaluation Platform - Frontend

React + Vite + TypeScript frontend for the RAG Evaluation Platform.

## Prerequisites

- [Node.js](https://nodejs.org/) (v18+)
- [npm](https://www.npmjs.com/)

## Quick Start

### Development

```bash
# Install dependencies
npm install

# Run the development server
npm run dev
```

The UI will be available at `http://localhost:3000`.

## Configuration

The frontend uses Vite's proxy to communicate with the backend. Configuration can be found in `vite.config.ts`.

- **Backend URL**: `http://localhost:8000` (proxied via `/api`)
- **Dev Port**: `3000`

## Features

- **Project Management**: Create and manage RAG evaluation projects.
- **Knowledge Bases**: Upload and index documents.
- **Test Sets**: Create and manage evaluation datasets.
- **RAG Configs**: Configure different RAG implementations.
- **Evaluations**: Run and view results of RAG evaluations.

## Built With

- [React](https://reactjs.org/)
- [Vite](https://vitejs.dev/)
- [Tailwind CSS](https://tailwindcss.com/)
- [shadcn/ui](https://ui.shadcn.com/)
- [TanStack Query](https://tanstack.com/query/latest)
- [Lucide React](https://lucide.dev/)
