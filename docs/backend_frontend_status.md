# Backend & Frontend Configuration Overview

_Updated: 2025-09-19_

## Backend (FastAPI)
- **App**: `backend/main.py`
- **Key endpoints**
  - Authentication stubs: `POST /v1/auth/login`, `GET /v1/auth/me`
  - Portfolios: `GET/POST /v1/portfolios`, `DELETE /v1/portfolios/{name}`
  - Thresholds: `POST /v1/portfolios/{name}/thresholds`
  - Notes: `GET/POST /v1/portfolios/{name}/notes`
  - Analytics: `GET /v1/analytics/portfolio`, `POST /v1/analytics/portfolio`
  - Advisory: `POST /v1/advisory/generate`
  - Events: `POST /v1/events`, `GET /v1/events`
- **Auth placeholder**: header-based using `X-User` and `X-Role`. Next step is JWT verification.
- **Persistence**: SQLite via `services/storage.py` tables for portfolios, annotations, events.
- **Shared analytics**: `services/portfolio.py` consumed by both API and Streamlit.
- **Run**: `uvicorn backend.main:app --reload`

## Frontend (Streamlit)
- **Main app**: `app.py`
  - Issuer overview with tabs: Overview, Insights, Source Library, Connections, Insight Report, Advisory.
  - Advisory tab hits `/v1/advisory/generate`, offers PDF export, logs events.
  - Uses env `ESG_AI_API_BASE` (default `http://127.0.0.1:8000/v1`).
- **Portfolio Command Center**: `pages/01_Portfolio_Command.py`
  - Portfolio saving/loading, threshold tuning, notes.
  - Calls backend `/v1/analytics/portfolio` and `/v1/events` with fallbacks.
  - Admin audit view with filters (portfolio/user/action).
- **Auth**: Streamlit login sidebar (`utils/auth.py`) using same static tokens as backend stub.
- **Shared utilities**: `utils/analytics.py` for filtering/context; `services/portfolio.py` for risk scoring.

## Event Flow
- UI actions call `_record_event` → `/v1/events` → stored in `portfolio_events`.
- Admin audit tab fetches `/v1/events` with filtering to display recent activity.

## Next Steps
1. Introduce JWT auth: add environment-configured JWKS URL, verification helper, update FastAPI dependencies + Streamlit API callers.
2. Replace SQLite with Postgres + Alembic migrations.
3. Surface `/v1/events` in React terminal once built.
