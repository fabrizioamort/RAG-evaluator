# Troubleshooting Guide

> **Quick solutions for common issues with the RAG Evaluator Platform.**

This guide covers the most frequently encountered problems and their solutions. Issues are organized by category for quick navigation.

---

## Table of Contents

- [Installation Issues](#installation-issues)
- [Docker & Infrastructure](#docker--infrastructure)
- [API & Backend Issues](#api--backend-issues)
- [Frontend Issues](#frontend-issues)
- [Evaluation Issues](#evaluation-issues)
- [RAG-Specific Issues](#rag-specific-issues)
- [Performance Issues](#performance-issues)
- [Database Issues](#database-issues)
- [Getting Help](#getting-help)

---

## Quick Diagnostics

Run these commands first to identify common issues:

```bash
# Check Docker services
docker-compose ps

# Check backend logs
docker-compose logs backend --tail=50

# Check database connectivity
docker-compose exec backend python -c "from app.database import engine; print('DB OK')"

# Verify API is running
curl http://localhost:8000/api/v1/health
```

---

## Installation Issues

### Python Version Mismatch

**Error:**
```
ERROR: Package requires Python >=3.11 but you have Python 3.9
```

**Solution:**
```bash
# Check your Python version
python --version

# Install Python 3.11+ via pyenv (recommended)
pyenv install 3.11.7
pyenv local 3.11.7

# Or use uv's managed Python
uv python install 3.11
```

### uv Not Found

**Error:**
```
command not found: uv
```

**Solution:**
```bash
# Install uv (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install uv (Windows PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Restart your terminal after installation
```

### Dependency Conflicts

**Error:**
```
ERROR: Cannot install package-a==1.0 and package-b==2.0 because these requirements conflict
```

**Solution:**
```bash
# Clear uv cache and reinstall
uv cache clean
rm -rf .venv
uv sync --all-extras
```

---

## Docker & Infrastructure

### Containers Won't Start

**Error:**
```
ERROR: Service 'backend' failed to build
```

**Solutions:**

1. **Rebuild containers:**
```bash
docker-compose build --no-cache
docker-compose up -d
```

2. **Check disk space:**
```bash
docker system df
docker system prune -a  # Warning: removes all unused data
```

3. **Check port conflicts:**
```bash
# macOS/Linux
lsof -i :8000
lsof -i :3000

# Windows
netstat -ano | findstr :8000
```

### Database Connection Refused

**Error:**
```
sqlalchemy.exc.OperationalError: connection refused
```

**Solutions:**

1. **Ensure database is running:**
```bash
docker-compose up -d postgres
docker-compose logs postgres
```

2. **Check DATABASE_URL in .env:**
```env
# For Docker Compose
DATABASE_URL=postgresql+asyncpg://postgres:password@postgres:5432/rag_eval

# For local development
DATABASE_URL=sqlite+aiosqlite:///./data/rag_eval.db
```

3. **Wait for database to be ready:**
```bash
# Database might still be initializing
docker-compose logs -f postgres
# Wait until you see "database system is ready to accept connections"
```

### Qdrant Connection Failed

**Error:**
```
qdrant_client.http.exceptions.UnexpectedResponse: Connection refused
```

**Solutions:**

1. **Start Qdrant:**
```bash
docker-compose up -d qdrant
```

2. **Check QDRANT_URL:**
```env
# Docker Compose
QDRANT_URL=http://qdrant:6333

# Local development
QDRANT_URL=http://localhost:6333
```

3. **Verify Qdrant is healthy:**
```bash
curl http://localhost:6333/health
```

### Neo4j Connection Issues

**Error:**
```
neo4j.exceptions.ServiceUnavailable: Unable to retrieve routing information
```

**Solutions:**

1. **Start Neo4j:**
```bash
docker-compose up -d neo4j
# Wait 30-60 seconds for Neo4j to initialize
```

2. **Check credentials:**
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

3. **Access Neo4j browser:**
Open [http://localhost:7474](http://localhost:7474) and verify you can log in.

---

## API & Backend Issues

### OpenAI API Key Invalid

**Error:**
```
openai.AuthenticationError: Invalid API key
```

**Solutions:**

1. **Verify API key:**
```bash
# Check if key is set
echo $OPENAI_API_KEY

# Test the key
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

2. **Check .env file:**
```env
# Make sure there are no quotes or extra spaces
OPENAI_API_KEY=sk-your-key-here
```

3. **Restart backend:**
```bash
docker-compose restart backend
```

### API Rate Limits

**Error:**
```
openai.RateLimitError: Rate limit exceeded
```

**Solutions:**

1. **Enable sequential mode:**
```env
DEEPEVAL_ASYNC_MODE=False
DEEPEVAL_THROTTLE_VALUE=1.0
```

2. **Reduce concurrent requests:**
```env
DEEPEVAL_MAX_CONCURRENT=3
```

3. **Increase timeouts:**
```env
OPENAI_TIMEOUT=900
DEEPEVAL_PER_TASK_TIMEOUT=1200
```

### 500 Internal Server Error

**Solution:**
```bash
# Check backend logs for details
docker-compose logs backend --tail=100

# Common causes:
# - Missing environment variables
# - Database migration issues
# - Import errors
```

### CORS Errors

**Error:**
```
Access to fetch at 'http://localhost:8000/api/v1/...' has been blocked by CORS policy
```

**Solutions:**

1. **Check frontend is using correct API URL:**
```typescript
// In frontend/.env or environment
VITE_API_URL=http://localhost:8000/api/v1
```

2. **Verify CORS is configured in backend:**
The backend should allow requests from the frontend origin.

---

## Frontend Issues

### Blank Page on Load

**Solutions:**

1. **Check browser console for errors:**
Press F12 and look at the Console tab.

2. **Verify API connection:**
```bash
curl http://localhost:8000/api/v1/health
```

3. **Rebuild frontend:**
```bash
cd platform/frontend
npm run build
```

### npm Install Fails

**Error:**
```
npm ERR! ERESOLVE unable to resolve dependency tree
```

**Solution:**
```bash
# Remove node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Or force resolution
npm install --legacy-peer-deps
```

### Hot Reload Not Working

**Solution:**
```bash
# Restart dev server
cd platform/frontend
npm run dev

# Check Vite config if using Docker volumes
# Volume mounts can interfere with file watching
```

---

## Evaluation Issues

### Evaluation Stuck at 0%

**Solutions:**

1. **Check backend logs:**
```bash
docker-compose logs -f backend
```

2. **Verify RAG index exists:**
The knowledge base must be indexed before evaluation.

3. **Check SSE connection:**
Open browser DevTools → Network tab → look for `/stream` request.

### Evaluation Times Out

**Error:**
```
TimeoutError: Evaluation exceeded maximum time
```

**Solutions:**

1. **Increase timeouts:**
```env
DEEPEVAL_PER_TASK_TIMEOUT=1800    # 30 minutes
DEEPEVAL_PER_ATTEMPT_TIMEOUT=600  # 10 minutes
OPENAI_TIMEOUT=900                # 15 minutes
```

2. **Reduce test set size:**
Try running with fewer test cases first.

3. **Check OpenAI status:**
Visit [status.openai.com](https://status.openai.com) for outages.

### All Metrics Return 0

**Possible Causes:**

1. **Empty retrieval results:**
```bash
# Test retrieval directly
uv run python -c "
from rag_evaluator.rag_implementations.vector_semantic import ChromaSemanticRAG
rag = ChromaSemanticRAG()
result = rag.query('test question')
print(result)
"
```

2. **Index not built:**
Verify the index was created successfully.

3. **Wrong RAG config:**
Ensure the selected RAG config points to the correct index.

### Inconsistent Scores

Scores vary significantly between runs.

**Solutions:**

1. **Set temperature to 0:**
```env
OPENAI_TEMPERATURE=0
```

2. **Use consistent test cases:**
Don't modify test cases between evaluations.

3. **Check for randomness in RAG:**
Some RAG implementations have non-deterministic elements.

---

## RAG-Specific Issues

### ChromaDB: Collection Not Found

**Error:**
```
ValueError: Collection 'rag_documents' does not exist
```

**Solution:**
```bash
# Rebuild the index
uv run rag-eval prepare --rag-type vector_semantic --input-dir data/raw
```

### Hybrid RAG: SPLADE Model Download Fails

**Error:**
```
OSError: Can't load tokenizer for 'prithvida/Splade_pp_en_v1'
```

**Solutions:**

1. **Check internet connection**

2. **Pre-download model:**
```python
from fastembed import SparseTextEmbedding
model = SparseTextEmbedding(model_name="prithvida/Splade_pp_en_v1")
```

3. **Use alternative model:**
```env
SPARSE_MODEL_NAME=naver/splade-cocondenser-ensembledistil
```

### Graph RAG: Entity Extraction Fails

**Error:**
```
neo4j.exceptions.CypherSyntaxError: Invalid input
```

**Solutions:**

1. **Check Neo4j version:**
```bash
docker-compose exec neo4j neo4j --version
# Should be 5.x
```

2. **Clear and rebuild:**
```bash
docker-compose down neo4j
docker volume rm rag-evaluator_neo4j_data
docker-compose up -d neo4j
# Wait for startup, then re-index
```

### Filesystem RAG: Preparation Fails

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/prepared/filesystem_rag'
```

**Solution:**
```bash
# Create directory and prepare
mkdir -p data/prepared/filesystem_rag
uv run rag-eval prepare --rag-type filesystem_rag --input-dir data/raw
```

---

## Performance Issues

### Slow Indexing

**Solutions:**

1. **Reduce batch size for large documents:**
```env
HYBRID_INDEXING_BATCH_SIZE=8
```

2. **Use smaller chunks:**
```env
HYBRID_CHUNK_SIZE=500
```

3. **Index in batches:**
Upload documents in smaller groups.

### Slow Evaluations

**Solutions:**

1. **Enable async mode:**
```env
DEEPEVAL_ASYNC_MODE=True
```

2. **Reduce metrics:**
Only evaluate necessary metrics.

3. **Use faster model:**
```env
OPENAI_MODEL=gpt-4o-mini
```

### High Memory Usage

**Solutions:**

1. **Reduce concurrent evaluations:**
```env
DEEPEVAL_MAX_CONCURRENT=3
```

2. **Clear embeddings cache:**
```python
# ChromaDB
import chromadb
client = chromadb.PersistentClient(path="./data/chroma_db")
client.delete_collection("rag_documents")
```

3. **Increase Docker memory:**
```yaml
# docker-compose.yml
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 4G
```

---

## Database Issues

### Migration Errors

**Error:**
```
alembic.util.exc.CommandError: Target database is not up to date
```

**Solutions:**

1. **Run migrations:**
```bash
cd platform/backend
uv run alembic upgrade head
```

2. **Check migration history:**
```bash
uv run alembic history
uv run alembic current
```

3. **Reset database (dev only):**
```bash
docker-compose down -v
docker-compose up -d
```

### SQLite Locked

**Error:**
```
sqlite3.OperationalError: database is locked
```

**Solutions:**

1. **Use PostgreSQL for concurrent access:**
```env
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/rag_eval
```

2. **Increase timeout:**
```python
# In database.py
engine = create_engine(url, connect_args={"timeout": 30})
```

### Data Corruption

**Solutions:**

1. **Backup and restore:**
```bash
# PostgreSQL
docker-compose exec postgres pg_dump -U postgres rag_eval > backup.sql
docker-compose exec -i postgres psql -U postgres rag_eval < backup.sql
```

2. **Reset completely:**
```bash
docker-compose down -v
rm -rf storage/*
docker-compose up -d
```

---

## Getting Help

### Gathering Debug Information

Before asking for help, collect:

1. **System info:**
```bash
python --version
uv --version
docker --version
docker-compose --version
```

2. **Error logs:**
```bash
docker-compose logs backend > backend.log 2>&1
```

3. **Configuration (remove secrets!):**
```bash
grep -v "KEY\|PASSWORD\|SECRET" .env > config-sanitized.txt
```

### Where to Get Help

| Resource | Best For |
|----------|----------|
| [GitHub Issues](https://github.com/fabrizioamort/RAG-evaluator/issues) | Bug reports, feature requests |
| [Discussions](https://github.com/fabrizioamort/RAG-evaluator/discussions) | Questions, best practices |
| Documentation | How-to guides, reference |

### Reporting Bugs

Include in your bug report:
- Operating system and version
- Python version
- Docker version
- Full error message and stack trace
- Steps to reproduce
- Expected vs. actual behavior
- Relevant configuration (sanitized)

---

## Diagnostic Commands Reference

```bash
# === Docker ===
docker-compose ps                    # Check service status
docker-compose logs -f <service>     # Follow logs
docker-compose restart <service>     # Restart a service
docker-compose down -v              # Remove all containers and volumes

# === Backend ===
cd platform/backend
uv run python -c "from app.config import settings; print(settings)"  # Check config
uv run alembic current              # Check migration status
uv run pytest -x                    # Run tests

# === Frontend ===
cd platform/frontend
npm run lint                        # Check for issues
npm run build                       # Build for production

# === Core ===
uv run rag-eval --help              # CLI help
uv run pytest tests/ -v             # Run core tests

# === Database ===
docker-compose exec postgres psql -U postgres -d rag_eval -c "\dt"  # List tables
docker-compose exec postgres psql -U postgres -d rag_eval -c "SELECT count(*) FROM project"

# === Vector Stores ===
curl http://localhost:6333/collections  # Qdrant collections
curl http://localhost:7474             # Neo4j browser
```
