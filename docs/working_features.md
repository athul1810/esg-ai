# ESG AI Platform – Working Features

_Updated: 2025-09-19_

## 1. Authentication & Roles
- **Roles:** `admin`, `analyst`, `viewer` with sign-in through the sidebar.
- **Capabilities**
  - Admin: full control (save/delete portfolios, update thresholds, add notes).
  - Analyst: save/update portfolios, modify thresholds, add notes; cannot delete.
  - Viewer: read-only experience (view portfolios, analytics, and notes).
- Role awareness enforced both in the UI (buttons/sliders disabled) and in backend requests via headers.

## 2. Portfolio Command Center
- Load data window (e.g., `dec30_to_jan12`) and define reporting period.
- Select and persist up to 10 companies per portfolio.
- Configure risk thresholds (tone floor, positive share minimum, tone vs industry delta, tone momentum).
- Save, load, and delete named portfolios; directory view shows existing configs and timestamps.
- Engagement notes: log analyst commentary per portfolio with author attribution.
- Risk snapshot metrics (articles, average tone, positive share) and CSV download.
- Catalyst digest, tone/volume charts, and ESG pillar exposure tables.
- Backend-connected analytics via `/v1/analytics/portfolio`, with automatic fallback to local computation if API unavailable.

## 3. Analytics & Scoring
- Shared helpers (`utils/analytics.py`, `services/portfolio.py`) compute:
  - Company context (tone trend, positive share, industry comparison).
  - Catalyst timeline (high-impact dates, pillars, sources).
  - Risk scores driven by configurable thresholds, categorised as Stable/Watch/Critical.
  - ESG pillar coverage, source-level sentiment tables.

## 4. Persistence Layer
- SQLite-backed storage for portfolios, thresholds, and notes via `services/storage.py`.
- Snapshot saved copies under `snapshots/change-1` and synced to `~/Desktop/esg-ai-backup` for rollback.

## 5. Backend API (FastAPI Prototype)
- Endpoints (header-based auth for now):
  - `GET /v1/portfolios`, `POST /v1/portfolios`, `DELETE /v1/portfolios/{name}`.
  - `POST /v1/portfolios/{name}/thresholds`.
  - `GET/POST /v1/portfolios/{name}/notes`.
  - `GET /v1/analytics/portfolio` (by name) or `POST /v1/analytics/portfolio` (ad hoc companies).
  - `POST /v1/auth/login`, `GET /v1/auth/me` (dev/testing only).
- Shares analytics logic with the Streamlit UI via `services/portfolio.py`.

## 6. Observability
- Structured JSON logging initialised app-wide (`utils/logging.py`), capturing key events: dataset load, portfolio synthesis, saves/deletes, notes.
- Logs printed to console currently; ready to be shipped to Datadog/ELK.

## 7. Documentation & Planning
- `docs/enterprise_blueprint.md`, `docs/architecture_overview.md`, `docs/implementation_roadmap.md`, and `docs/backend_service_design.md` lay out the enterprise roadmap and architecture.

## 8. Known Limitations / Next Steps
- Authentication still uses local tokens; needs JWT/IdP integration.
- Backend uses SQLite & sample data; replace with Postgres + warehouse pipeline.
- No audit/event API yet exposed to UI (logs stored only in console).
- Streamlit `API_BASE_URL` points to localhost; parameterise via environment for deployment.
- Tests limited to compile checks; unit/integration tests for backend pending.
