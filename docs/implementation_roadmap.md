# ESG AI Implementation Roadmap

## 1. Programme Structure
- **Horizon 1 – Foundations (0-3 months):** Establish secure platform core, automated data ingestion, and enterprise-ready dashboards.
- **Horizon 2 – Intelligence (3-6 months):** Expand analytics, collaboration workflows, and reporting automation.
- **Horizon 3 – Differentiation (6-12 months):** Deliver advanced AI copilot capabilities, scenario modelling, and ecosystem integrations.

## 2. Epics & Key Deliverables
| Epic | Focus | Horizon |
| --- | --- | --- |
| Data Ingestion & Quality | Automated pipelines, QA, catalog | H1 |
| Access & Security | SSO, RBAC, audit logs, tenant isolation | H1 |
| Portfolio Intelligence | Persistent configs, alerting, benchmarking APIs | H1 |
| Issuer Copilot | Deep-dive workspace, catalyst explainability | H2 |
| Workflow & Collaboration | Comments, tasks, approval flows, notifications | H2 |
| Reporting Suite | Template designer, PDF/PPT/XLSX exports, scheduling | H2 |
| AI Governance | Guardrails, monitoring, feedback loop, explainability | H3 |
| Ecosystem Integrations | CRM, ticketing, BI connectors | H3 |

## 3. Sprint Backlog (Next 6-8 Weeks)

### Sprint 1: Platform Hardening
- Integrate Auth0/Azure AD for SSO + RBAC (admin, analyst, viewer roles).
- Implement persistent storage for portfolio configurations and alert rules (PostgreSQL/Cloud SQL).
- Instrument structured logging & error tracking (Sentry/Datadog).
- Add environment configuration management (dotenv/secrets manager).

### Sprint 2: Data Ops
- Stand up ingestion pipeline for daily media feeds (GDELT APIs) with QA checks (Great Expectations).
- Load ingested data into warehouse and expose via analytics API.
- Automate data refresh scheduling and alert on failures.
- Update Streamlit app to read from new warehouse endpoints.

### Sprint 3: Portfolio Enhancements
- Persist risk scores, alerts, and analyst annotations.
- Add configurable alert thresholds and notification preferences.
- Introduce benchmarking charts (compare vs sector averages over time).
- Export Portfolio Command Center views to PDF with branding and append evidence.

### Sprint 4: Issuer Workspace Upgrade
- Build backend service for issuer narratives (tone trajectory, ESG pillars, catalysts).
- Launch React-based issuer summary page (shared components) while keeping Streamlit for internal use.
- Implement evidence bookmarking and shareable insight links.
- Integrate metadata provenance (source IDs, retrieval timestamps).

### Sprint 5: Workflow Foundations
- Create engagement tracker data model (issues, owners, deadlines).
- Surface tasks within Streamlit and send notifications to Teams/Slack.
- Add audit logging for critical actions (alert acknowledgements, report generation).
- Draft compliance documentation (SOPs, runbooks).

## 4. Dependencies & Risks
- **Data Licensing:** Confirm rights for third-party ESG feeds.
- **Security Review:** Early penetration test to validate auth and network boundaries.
- **Change Management:** Analyst onboarding, training materials, adoption metrics.
- **Resource Planning:** Dedicated data engineer, backend engineer, front-end engineer, ML engineer, product/design support.

## 5. Success Metrics
- Time to surface high-risk issuer (goal < 10 minutes from data ingestion).
- Number of manual analyst hours saved per engagement (target 30% reduction).
- Report generation turnaround time (<1 hour end-to-end).
- Platform uptime (≥99.5%) and incident mean time to resolve (<2 hours).
- User adoption and satisfaction (NPS ≥ 45 for analysts).

---
_Maintained: 2025-09-19_
