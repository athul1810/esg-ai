# Backend Service Blueprint

## 1. Goals
- Decouple Streamlit UI from data layer with a REST API that can be secured, scaled, and versioned independently.
- Provide a consistent contract for portfolios, alerts, annotations, and analytics derived from the warehouse.
- Establish a foundation that can later run under FastAPI + async workers, with JWT-based authentication.

## 2. Service Overview
| Service | Responsibilities | Notes |
| --- | --- | --- |
| `api-gateway` | Terminates TLS, validates JWTs, rate limiting | Deployed via API Gateway/Kong/NGINX |
| `portfolio-service` | CRUD portfolios, thresholds, annotations | Reads/writes Postgres, caches in Redis |
| `analytics-service` | Aggregations (tone metrics, catalysts) via warehouse | Async jobs for heavy calcs |
| `event-service` | Audit logging, structured events -> ELK/Datadog | Exposes `/events` API |
| `auth-service` | Integrates with IdP (Auth0/Azure AD) for token issuance | Optional if leveraging managed IdP |

For the current iteration we will prototype a combined FastAPI app exposing `/portfolios`, `/alerts`, `/notes`, and `/events` endpoints, leaving room to split services later.

## 3. API Contract (v1)

### Authentication
- JWT bearer tokens issued by IdP, containing `sub` (user id), `role`, `tenant` claims.
- Local development can use static tokens (mirrors Streamlit prototype).

### Endpoints
- `GET /v1/portfolios` – list portfolios for tenant.
- `POST /v1/portfolios` – create/update portfolio `{name, companies, start_date, end_date, alerts}`.
- `DELETE /v1/portfolios/{name}` – remove portfolio.
- `POST /v1/portfolios/{name}/thresholds` – update alert thresholds.
- `GET /v1/portfolios/{name}/notes` – list notes.
- `POST /v1/portfolios/{name}/notes` – add note `{note, author}`.
- `GET /v1/analytics/portfolio?name=...` – return computed risk snapshot, charts, catalysts.
- `GET /v1/events` – paginated audit log filtered by portfolio/user/action.

### Response Models (simplified)
```json
{
  "name": "Climate Watch",
  "companies": ["tesla", "bp"],
  "start_date": "2025-01-01",
  "end_date": "2025-03-01",
  "alerts": {
    "tone_floor": -0.5,
    "positive_share_min": 0.45
  },
  "updated_at": "2025-09-19T10:30:00Z"
}
```

## 4. Data Persistence
- **Database:** Postgres (tables: `portfolios`, `portfolio_thresholds`, `portfolio_notes`, `portfolio_events`).
- **Warehouse:** Snowflake/BigQuery – analytics service queries aggregated tables or materialized views.
- **Cache:** Redis for recent analytics snapshots.
- **Migration:** Alembic for schema migrations.

## 5. Security
- JWT validation middleware + role/permission enforcement per endpoint.
- Row-level access using `tenant_id` claim.
- Audit events written asynchronously (Celery worker) to event-store; surfaced via `/events`.

## 6. Deployment
- Containerised (FastAPI + Uvicorn)
- Helm chart deploying API behind ALB/API Gateway.
- CI pipeline: linting (ruff), tests (pytest), security scans (bandit, trivy), auto deploy to staging.

## 7. Milestones
1. **MVP service (local)** – FastAPI app, SQLite/postgres (dev), `/portfolios` CRUD and `analytics/portfolio` stub (using existing utils).
2. **Auth Integration** – JWT verification, role enforcement, swap Streamlit to call API.
3. **Analytics pipeline** – connect to warehouse queries; integrate caching.
4. **Event API** – central logging, Streamlit audit view.

---
_Maintained: 2025-09-19_
