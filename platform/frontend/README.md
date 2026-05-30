# RAG Evaluator Frontend

React, Vite, TypeScript, and Tailwind frontend for the RAG Evaluator Platform.

## Setup

```powershell
cd platform/frontend
npm install
```

## Run

Start the backend first:

```powershell
cd platform/backend
uv run python dev_server.py
```

Then start the frontend:

```powershell
cd platform/frontend
npm run dev
```

Open <http://localhost:3000>.

The Vite dev server proxies `/api` to `http://localhost:8000`.

## Scripts

| Command | Description |
| --- | --- |
| `npm run dev` | Start Vite development server on port 3000. |
| `npm run build` | Type-check and create a production build. |
| `npm run lint` | Run ESLint. |
| `npm run preview` | Preview the production build. |
| `npm run test` | Run Vitest. |

## Features

- Dashboard and project overview.
- Project workspaces with tabs for knowledge bases, test sets, RAG configs,
  evaluations, comparisons, and trends.
- Knowledge base document upload and index creation.
- Index progress and retry/archive actions.
- Test set creation, JSON import/export, AI generation, and generated-case review.
- Evaluation wizard, progress view, result detail, metric reasoning, and retrieval traces.
- Baseline evaluation selection.
- Evaluation comparison UI with aggregate and per-question deltas.
- Playground for ad hoc multi-index queries.

## API Configuration

The frontend API client uses:

```text
VITE_API_URL or http://localhost:8000
```

In development, requests are made to `/api/v1` through the Vite proxy. In a production
build, set `VITE_API_URL` to the backend origin if it differs from the frontend origin.

## Structure

```text
platform/frontend/src/
  api/             API client and TypeScript DTOs
  components/      Shared UI and feature components
  hooks/           React hooks
  lib/             Utilities
  pages/           Route-level pages
```
