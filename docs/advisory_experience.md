# Advisory Workspace 2.0

_Updated: 2025-09-19_

## 1. Vision
- Transform the current advisory tab into a co-pilot that curates investment talking points, risk narratives, and client-ready action plans in real time.
- Blend AI-driven synthesis with measurable evidence, compliance guardrails, and collaboration workflows.

## 2. User Journeys
| Persona | Use Case | Advisory Outcome |
| --- | --- | --- |
| Sector Analyst | Prep for client briefing | Tailored summary (key risks, opportunities, catalysts, recommended messaging) |
| Relationship Manager | Daily check-in | “What changed since yesterday?”, watchlist, alerts in plain language |
| Partner / Director | Executive briefing | One-page narrative + heat map of client exposures |

## 3. Advisory Components
1. **Client Context Layer**
   - Input client portfolio, mandate, risk appetite, ESG priorities.
   - Link to stored engagement notes and open actions.
2. **Insight Generator**
   - Prompt templates (e.g., *“Draft a briefing for high-risk issuers”*, *“Summarise controversies for board-level update”*).
   - Response includes: executive summary, top 3 talking points, risk watchlist, recommended next steps, supporting data.
3. **Evidence Attachments**
   - Cite tone metrics, catalyst articles, ESG pillar deltas.
   - Provide one-click export (Markdown → PDF/PPT).
4. **Collaboration Hooks**
   - Add advisory note to portfolio (logged in events).
   - Share via email/Teams directly from the interface.
5. **Compliance Guardrails**
   - Model disclaimers (“AI-generated, review before client use”).
   - Keyword filters to avoid prohibited phrasing; highlight missing data.

## 4. Enhancements Roadmap
| Phase | Feature | Notes |
| --- | --- | --- |
| Phase 1 | Guided prompts, richer output sections, improved UI copy | Deliver within current Streamlit tab |
| Phase 2 | Client profiles, template library per persona | Store/reuse templates, auto-fill client details |
| Phase 3 | Multi-modal exports (PDF/PPT), collaborative approvals | Integrate with reporting service & notification layer |
| Phase 4 | Continuous learning feedback loop, tone-of-voice controls | Capture analyst edits, improve AI responses |

## 5. Technical Requirements
- **Data Inputs:** Portfolio snapshot, ESG context, event log, client profile metadata.
- **AI Stack:** Prompt orchestration layer, safety filters, summarisation + recommendation models, retrieval augmented content.
- **UI:** Rich markdown renderer, “accept/edit” workflow before finalising note, export buttons.
- **Backend:** `/advisory/generate` endpoint (POST) taking prompt, client context, returning structured advisory object.
- **Observability:** Log prompt, response length, confidence scores; feed into `/events` for audit.

## 6. Next Actions
1. Prototype `/advisory/generate` API stub returning structured sections.
2. Update advisor tab to display sections: Executive Summary, Talking Points, Risk Radar, Recommended Actions, Evidence.
3. Introduce client profile storage (per portfolio) to personalize outputs.
4. Add “Save advisory note” to portfolio events + export to Markdown/PDF.
