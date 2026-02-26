# Implementation Milestones – Advisory & Terminal Evolution

_Updated: 2025-09-19_

## 1. Immediate (0-4 Weeks)
- **Advisory Workspace 2.0**
  - Build `/advisory/generate` backend stub returning structured sections.
  - Update Streamlit advisory tab with new UI layout (Executive Summary, Talking Points, Risk Radar, Actions, Evidence).
  - Store client profile metadata per portfolio and surface in advisory prompts.
- **Event Logging**
  - Finish wiring UI actions (save, update, delete, note, advisory export) to `/v1/events`.
  - Add simple audit view in Streamlit (admin-only) reading `/v1/events`.
- **Developer Ready**
  - Finalise API docs (OpenAPI) and publish quickstart for backend endpoints.
  - Stabilise `.env`/settings for API base URL, tokens.

## 2. Near Term (1-3 Months)
- **Backend Hardening**
  - Swap header auth for JWT validation against chosen IdP.
  - Move persistence from SQLite to Postgres; add Alembic migrations.
  - Deploy API to staging with CI/CD and observability (logs/metrics).
- **Data Pipeline**
  - Ingest real news feeds daily (batch) into warehouse; integrate with analytics service.
  - Expand context (financial KPIs, filings) consumed by advisory responses.
- **UI Enhancements**
  - Introduce React/Next.js shell for terminal navigation; embed Streamlit modules where needed.
  - Build audit/event dashboard + export.
  - Implement advisory template library & PDF export.

## 3. Mid Term (3-6 Months)
- **Portfolio Terminal**
  - Real-time alerting engine, watchlists, comparative charts.
  - Scenario analysis (stress tests) and regulatory compliance checks.
- **Integration Layer**
  - Push notifications (Teams/Slack) for events.
  - CRM/ticketing integration (Salesforce/ServiceNow) for task handoffs.
- **Developer Portal**
  - API key management, usage analytics, reference notebooks.
- **Data Quality & Lineage**
  - Implement data catalog, lineage tracking, automated QA alerts.

## 4. Long Term (6-12 Months)
- **Market Overview Module** – live indices, ESG sector heatmaps, news widgets.
- **Global Coverage Expansion** – multiple languages, region-specific compliance.
- **Advisory Studio** – full document builder, collaboration, approvals, presentation exports.
- **AI Governance** – feedback loop, content moderation, tone-of-voice customisation, model retraining pipeline.

## 5. Dependencies & Owners (TBD)
- Assign leads for data engineering, backend, front-end, AI/ML, product/design.
- Establish steering cadence (monthly) to review progress vs milestones.
