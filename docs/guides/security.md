# Security & Best Practices Guide

> **Guidelines for secure deployment and optimal usage of the RAG Evaluator Platform.**

This guide covers security considerations, best practices for production deployment, and recommendations for getting the most out of your RAG evaluation workflows.

---

## Table of Contents

- [Security Overview](#security-overview)
- [Authentication & Authorization](#authentication--authorization)
- [Secrets Management](#secrets-management)
- [Data Security](#data-security)
- [Network Security](#network-security)
- [Production Deployment](#production-deployment)
- [RAG Evaluation Best Practices](#rag-evaluation-best-practices)
- [Cost Optimization](#cost-optimization)
- [Monitoring & Observability](#monitoring--observability)

---

## Security Overview

### Threat Model

The RAG Evaluator Platform handles sensitive data:

| Data Type | Sensitivity | Protection |
|-----------|-------------|------------|
| API Keys | High | Environment variables, never logged |
| Documents | Variable | Access controls, encryption at rest |
| Test Sets | Medium | Project-level isolation |
| Evaluation Results | Medium | Database access controls |
| User Data | Variable | Depends on deployment |

### Security Principles

1. **Least Privilege**: Services only have necessary permissions
2. **Defense in Depth**: Multiple security layers
3. **Secure by Default**: Safe configuration out of the box
4. **Audit Trail**: Logging for security events

---

## Authentication & Authorization

### Default Configuration

The open-source edition runs **without authentication** by default for local development convenience.

```
WARNING: The default configuration is NOT suitable for production
or multi-user environments without additional security measures.
```

### Production Authentication Options

#### Option 1: Reverse Proxy Authentication

Use a reverse proxy (nginx, Traefik, AWS ALB) for authentication:

```nginx
# nginx.conf example
server {
    listen 443 ssl;
    server_name rag-eval.example.com;

    ssl_certificate /etc/ssl/certs/cert.pem;
    ssl_certificate_key /etc/ssl/private/key.pem;

    # Basic authentication
    auth_basic "RAG Evaluator";
    auth_basic_user_file /etc/nginx/.htpasswd;

    location / {
        proxy_pass http://frontend:3000;
    }

    location /api/ {
        proxy_pass http://backend:8000;
    }
}
```

#### Option 2: OAuth2/OIDC Proxy

Use oauth2-proxy for SSO integration:

```yaml
# docker-compose.yml addition
oauth2-proxy:
  image: quay.io/oauth2-proxy/oauth2-proxy
  environment:
    - OAUTH2_PROXY_PROVIDER=google
    - OAUTH2_PROXY_CLIENT_ID=${OAUTH_CLIENT_ID}
    - OAUTH2_PROXY_CLIENT_SECRET=${OAUTH_CLIENT_SECRET}
    - OAUTH2_PROXY_COOKIE_SECRET=${COOKIE_SECRET}
    - OAUTH2_PROXY_UPSTREAMS=http://frontend:3000
  ports:
    - "4180:4180"
```

#### Option 3: API Gateway

Use cloud-native API gateways:
- AWS API Gateway with Cognito
- Google Cloud Endpoints with IAP
- Azure API Management with AAD

### API Key Authentication (Custom)

For programmatic access, implement API key authentication:

```python
# Example middleware
from fastapi import Header, HTTPException

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key not in valid_api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
```

---

## Secrets Management

### Required Secrets

| Secret | Purpose | Rotation Frequency |
|--------|---------|-------------------|
| `OPENAI_API_KEY` | LLM access | Monthly |
| `DB_PASSWORD` | Database access | Quarterly |
| `NEO4J_PASSWORD` | Graph DB access | Quarterly |
| `QDRANT_API_KEY` | Vector DB access | Quarterly |

### Best Practices

#### Never Commit Secrets

```bash
# .gitignore
.env
.env.*
*.pem
*.key
credentials.json
```

#### Use Environment-Specific Keys

```bash
# Development (limited permissions)
OPENAI_API_KEY=sk-dev-...

# Production (full permissions)
OPENAI_API_KEY=sk-prod-...
```

#### Secret Rotation

1. Generate new secret
2. Update in secret manager
3. Deploy with new secret
4. Verify functionality
5. Revoke old secret

### Production Secret Storage

#### Docker Secrets

```yaml
# docker-compose.yml
secrets:
  openai_key:
    external: true

services:
  backend:
    secrets:
      - openai_key
    environment:
      - OPENAI_API_KEY_FILE=/run/secrets/openai_key
```

#### Kubernetes Secrets

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: rag-evaluator-secrets
type: Opaque
stringData:
  openai-api-key: sk-your-key
  db-password: your-password
```

#### Cloud Secret Managers

```python
# AWS Secrets Manager
import boto3

def get_secret(secret_name):
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return response['SecretString']

# Google Secret Manager
from google.cloud import secretmanager

def get_secret(project_id, secret_id):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")
```

---

## Data Security

### Document Security

#### Upload Validation

The platform validates uploaded files:

| Check | Purpose |
|-------|---------|
| File extension | Only allow PDF, DOCX, TXT, MD |
| MIME type | Verify content matches extension |
| File size | Prevent DoS via large files |
| Virus scan | Optional integration with ClamAV |

#### Sensitive Document Handling

```
CAUTION: Documents are stored on disk and in vector databases.
Consider data classification before uploading sensitive content.
```

**Recommendations:**
- Remove PII before uploading
- Use data anonymization for test sets
- Implement document-level access controls if needed
- Consider on-premise deployment for sensitive data

### Database Security

#### PostgreSQL Hardening

```sql
-- Create dedicated user with minimal permissions
CREATE USER rag_eval_app WITH PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE rag_eval TO rag_eval_app;
GRANT USAGE ON SCHEMA public TO rag_eval_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO rag_eval_app;

-- Enable SSL
ALTER SYSTEM SET ssl = on;
ALTER SYSTEM SET ssl_cert_file = '/path/to/server.crt';
ALTER SYSTEM SET ssl_key_file = '/path/to/server.key';
```

#### Connection Security

```env
# Force SSL connection
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db?ssl=require
```

### Encryption

#### At Rest

| Component | Encryption Option |
|-----------|-------------------|
| PostgreSQL | TDE (Transparent Data Encryption) |
| Vector stores | Filesystem encryption (LUKS, EBS encryption) |
| Backups | GPG encryption |

#### In Transit

All communications should use TLS:

```yaml
# docker-compose.yml with TLS
services:
  nginx:
    image: nginx:alpine
    volumes:
      - ./certs:/etc/nginx/certs:ro
    ports:
      - "443:443"
```

---

## Network Security

### Deployment Architecture

![Network Security Architecture](../images/security-network-architecture.png)
<!-- PLACEHOLDER: security-network-architecture.png - Network diagram showing DMZ, internal network, databases -->

```
┌─────────────────────────────────────────────────────────────────┐
│                         INTERNET                                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                     ┌──────┴──────┐
                     │   WAF/CDN   │
                     └──────┬──────┘
                            │
┌───────────────────────────┼─────────────────────────────────────┐
│  DMZ                      │                                     │
│                    ┌──────┴──────┐                              │
│                    │    nginx    │                              │
│                    │   (TLS)     │                              │
│                    └──────┬──────┘                              │
└───────────────────────────┼─────────────────────────────────────┘
                            │
┌───────────────────────────┼─────────────────────────────────────┐
│  Application Tier         │                                     │
│         ┌─────────────────┼─────────────────┐                  │
│         │                 │                 │                   │
│   ┌─────┴─────┐    ┌──────┴──────┐  ┌──────┴──────┐           │
│   │  Frontend │    │   Backend   │  │   Worker    │           │
│   │   :3000   │    │    :8000    │  │   (Jobs)    │           │
│   └───────────┘    └──────┬──────┘  └──────┬──────┘           │
└───────────────────────────┼─────────────────┼───────────────────┘
                            │                 │
┌───────────────────────────┼─────────────────┼───────────────────┐
│  Data Tier                │                 │                   │
│         ┌─────────────────┼─────────────────┤                  │
│         │                 │                 │                   │
│   ┌─────┴─────┐    ┌──────┴──────┐  ┌──────┴──────┐           │
│   │ PostgreSQL│    │   Qdrant    │  │    Neo4j    │           │
│   │   :5432   │    │    :6333    │  │    :7687    │           │
│   └───────────┘    └─────────────┘  └─────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

### Firewall Rules

| Source | Destination | Port | Protocol | Allow |
|--------|-------------|------|----------|-------|
| Internet | nginx | 443 | HTTPS | Yes |
| nginx | Frontend | 3000 | HTTP | Yes |
| nginx | Backend | 8000 | HTTP | Yes |
| Backend | PostgreSQL | 5432 | TCP | Yes |
| Backend | Qdrant | 6333 | HTTP | Yes |
| Backend | Neo4j | 7687 | Bolt | Yes |
| Backend | OpenAI API | 443 | HTTPS | Yes |

### Docker Network Isolation

```yaml
# docker-compose.yml
networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true  # No external access

services:
  nginx:
    networks:
      - frontend
      - backend

  frontend:
    networks:
      - frontend

  backend:
    networks:
      - backend

  postgres:
    networks:
      - backend
```

---

## Production Deployment

### Pre-Deployment Checklist

- [ ] All secrets in secure storage (not in code)
- [ ] TLS certificates installed
- [ ] Database backups configured
- [ ] Monitoring and alerting set up
- [ ] Authentication configured
- [ ] Firewall rules in place
- [ ] Rate limiting enabled
- [ ] Log aggregation configured

### Health Checks

```yaml
services:
  backend:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### Resource Limits

```yaml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

### High Availability

For production:
- Use managed databases (RDS, Cloud SQL)
- Deploy multiple backend replicas
- Use load balancer with health checks
- Implement database replication

---

## RAG Evaluation Best Practices

### Test Set Design

1. **Diverse Questions**: Cover different difficulty levels
2. **Representative Sample**: Match real-world usage patterns
3. **Regular Updates**: Keep test sets current with content changes
4. **Version Control**: Track test set changes over time

### Evaluation Strategy

| Stage | Frequency | Test Size | Metrics |
|-------|-----------|-----------|---------|
| Development | Per change | 10-20 | Faithfulness |
| Pre-commit | Daily | 50-100 | Faith + Correct |
| Release | Weekly | 100+ | All metrics |

### Result Interpretation

```
GUIDELINE: Focus on trends, not absolute numbers.
A 5% improvement over baseline is more meaningful
than hitting an arbitrary 0.85 threshold.
```

### A/B Testing RAG Configurations

1. Create baseline evaluation
2. Make one configuration change
3. Run evaluation on same test set
4. Compare results statistically
5. Document findings

---

## Cost Optimization

### Token Usage

| Optimization | Impact | Trade-off |
|--------------|--------|-----------|
| Smaller model (gpt-4o-mini) | 10-20x cheaper | Slightly lower quality |
| Reduce top_k | Fewer tokens per query | May miss context |
| Shorter chunks | Less context per chunk | May fragment meaning |
| Fewer metrics | Fewer LLM calls | Less comprehensive |

### Evaluation Cost Estimation

```
Cost per evaluation ≈ (test_cases × metrics × avg_tokens) × token_price

Example:
- 100 test cases
- 3 metrics
- ~2000 tokens per metric call
- $0.01 per 1K tokens (gpt-4o-mini)

Cost ≈ 100 × 3 × 2 × $0.01 = $6 per evaluation
```

### Cost-Saving Strategies

1. **Cache Retrieval Results**: Reuse context across evaluations
2. **Batch Evaluations**: Run during off-peak hours
3. **Progressive Evaluation**: Start with quick metrics, add more if needed
4. **Sampling**: Evaluate subset for development, full set for releases

---

## Monitoring & Observability

### Key Metrics

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| API Response Time | < 500ms | > 2s |
| Evaluation Success Rate | > 95% | < 90% |
| Database Connections | < 80% pool | > 90% pool |
| Error Rate | < 1% | > 5% |

### Logging

```python
# Recommended log format
import logging

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Log security events
logger.info(f"Evaluation started: project={project_id}, user={user_id}")
logger.warning(f"Rate limit approached: user={user_id}, requests={count}")
logger.error(f"Authentication failed: ip={client_ip}")
```

### Alerting

Set up alerts for:
- Evaluation failures
- High error rates
- API timeout increases
- Database connection issues
- Disk space warnings

### Dashboard Recommendations

Include panels for:
- Evaluation throughput
- Metric score trends
- Token usage over time
- Error rate by type
- Response time percentiles

---

## Security Incident Response

### If API Keys Are Compromised

1. **Immediately** revoke the compromised key
2. Generate new key
3. Update all deployments
4. Check logs for unauthorized usage
5. Review and enhance storage practices

### If Data Breach Suspected

1. Isolate affected systems
2. Preserve logs for investigation
3. Assess scope of exposure
4. Notify stakeholders as required
5. Implement remediation measures

---

## Related Documentation

- [Configuration Reference](configuration.md) - Environment variables
- [Deployment Guide](../deployment.md) - Production deployment
- [Troubleshooting](troubleshooting.md) - Common issues
- [Architecture Overview](../ARCHITECTURE.md) - System design
