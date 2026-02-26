# ESG AI Platform Architecture

## 1. Platform Principles
- **Modular services** with clear domain boundaries (ingest, analytics, workflow, reporting).
- **Configurable multi-tenancy** to support multiple client workspaces and environment isolation.
- **Observability-first** design for trust, compliance, and rapid incident response.
- **Security & compliance by design** (least privilege, data residency, auditability).

## 2. Logical Architecture
```
┌──────────────┐      ┌──────────────┐      ┌───────────────┐      ┌──────────────┐
│ Data Sources │ ───▶ │ Ingestion    │ ───▶ │ Processing &   │ ───▶ │ Serving Layer │
│  • Media     │      │  Pipelines   │      │  Enrichment    │      │  (APIs/UI)    │
│  • Filings   │      │  (stream/bat)│      │  (ETL/ML)      │      │              │
│  • Ratings   │      └──────────────┘      └───────────────┘      └──────────────┘
│  • ESG KPIs  │             │                         │                       │
└──────────────┘             ▼                         ▼                       ▼
                        Object Storage           Analytics Warehouse     Apps (Streamlit,
                        (Raw Landing)            Feature Store           React Portal,
                                                 Vector DB               APIs)
```

## 3. Component Breakdown

### 3.1 Data Ingestion Layer
- **Collectors:** Python services / Airbyte connectors / managed ingestion (e.g., AWS AppFlow) for each source.
- **Message Bus:** Kafka/Kinesis for streaming media feeds; SQS/Cloud Pub/Sub for batch orchestration.
- **Landing Zone:** Object storage (S3/GCS) partitioned by source/date, versioned for traceability.
- **Metadata Capture:** Data catalog (Amundsen/OpenMetadata) with lineage and quality metrics.

### 3.2 Processing & Enrichment
- **Batch Transformation:** DBT/Spark pipelines to cleanse, normalise, deduplicate, and score.
- **Feature Engineering:** Generate sentiment aggregates, ESG pillars, network embeddings, controversy tags.
- **Machine Learning:** Managed ML (SageMaker/Vertex) for model training (sentiment, classification, risk scoring).
- **Vector Indexing:** Store embeddings in Pinecone/Weaviate/pgvector for semantic search and clustering.
- **Quality Gates:** Automated checks (Great Expectations) with alerting to Slack/Teams.

### 3.3 Serving Layer
- **Data Warehouse:** Snowflake/BigQuery/Athena for analytics queries, powering dashboards and exports.
- **API Gateway:** FastAPI/GraphQL endpoints behind Kong/API Gateway with JWT-based access.
- **Workflow Service:** Task/approval microservice (e.g., Temporal) to orchestrate engagement lifecycle.
- **Report Service:** Headless rendering (WeasyPrint/LaTeX/DeckTape) for PDF/PPT/XLSX exports.
- **LLM Service:** Guardrailed prompt orchestration (LangChain/LlamaIndex) calling managed LLM endpoints with safety filters.

### 3.4 Experience Layer
- **Internal UI:** Streamlit for analyst consoles (Portfolio Command, Issuer Workspace).
- **Client Portal:** React/Next.js app for external users with branded visuals and limited feature set.
- **Notifications:** Integration service for Teams/Slack/email with templated alerts.

## 4. Cross-Cutting Concerns
- **Authentication:** OAuth2/OIDC with SCIM provisioning (Okta, Azure AD). Support MFA, conditional access.
- **Authorisation:** RBAC + attribute checks (client, region, practice) enforced at API and query level.
- **Audit & Logging:** Centralised log aggregation (CloudWatch/Loki), immutable audit trails stored in cold archive.
- **Observability:** Metrics (Prometheus), traces (OpenTelemetry), alerts via PagerDuty.
- **Configuration:** Environment configs via secrets manager and parameter store. Feature flags for gradual rollout.

## 5. Deployment Topology
- **Kubernetes Cluster** per environment with autoscaling and network policies.
- **CI/CD:** GitHub Actions triggering Terraform (infra) and ArgoCD (app deploy). Automated tests, security scans, and policy checks.
- **Data Backups:** Point-in-time recovery for warehouse, daily snapshots for object storage, disaster recovery region.
- **Data Residency:** Region-specific clusters + storage; routing rules ensuring client data stays within agreed geography.

## 6. Security & Compliance
- **Encryption:** TLS 1.2+, KMS-managed keys for storage, secrets rotation.
- **Compliance Controls:** SOC 2/ISO policies, GDPR legal basis tracking, DLP scanning, retention policies.
- **Access Reviews:** Quarterly RBAC reviews, automated detection of privilege escalations.
- **Vulnerability Management:** Regular container scans, dependency monitoring (Dependabot), penetration tests.

## 7. Scalability Considerations
- Horizontal scaling for ingestion workers and API pods.
- Caching layers (Redis/CloudFront) for frequently accessed datasets.
- Partitioned data models for large portfolios (client/region/time).
- Event-driven architecture to process new data incrementally instead of full reloads.

## 8. Data Model Highlights
- **Entities:** Company, Source, Article, ESG Metric, Alert, Engagement, Report, User, Tenant.
- **Relationships:** Articles link to companies & sources; alerts reference trigger rules and engagements; reports reference portfolios.
- **Versioning:** Slowly changing dimensions for company metadata, event-driven updates for scores.

## 9. Open Questions
- Preferred cloud/platform stack? (AWS vs GCP vs Azure).
- Required data residency locales? (US, EU, APAC).
- Existing identity provider integration specifics?
- Target SLAs/OLAs and support model expectations?
- Third-party data licensing constraints?

---
_Maintained: 2025-09-19_
