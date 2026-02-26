"""FastAPI backend for ESG AI."""

from datetime import date
from typing import List, Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, status

from download_data import Data
from services.storage import (
    initialise_storage,
    list_portfolio_configs,
    save_portfolio_config,
    get_portfolio_config,
    delete_portfolio_config,
    update_portfolio_alerts,
    add_portfolio_annotation,
    list_portfolio_annotations,
    log_portfolio_event,
    list_portfolio_events,
)
from services.portfolio import DEFAULT_ALERT_RULES, build_portfolio_snapshot
from utils.analytics import filter_on_date, build_company_context
from utils.logging import get_logger

from backend import schemas
from backend.dependencies import get_current_user


LOGGER = get_logger(__name__)
app = FastAPI(title="ESG AI API", version="0.1.0")


@app.on_event("startup")
def startup_event() -> None:
    initialise_storage()
    LOGGER.info("api_startup")


@app.post("/v1/auth/login")
def login(payload: dict):
    """Temporary endpoint for local credentials (mirrors Streamlit prototype)."""
    username = payload.get("username")
    token = payload.get("token")
    mapping = {
        "admin": "admin123",
        "analyst": "analyst123",
        "viewer": "viewer123",
        "superuser": "super-secret",
    }
    if not username or not token or mapping.get(username) != token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    role = "admin" if username in ("admin", "superuser") else username
    return {"username": username, "role": role}


@app.get("/v1/auth/me")
def auth_me(user=Depends(get_current_user)):
    return user


@app.get("/v1/portfolios", response_model=schemas.PortfolioListResponse)
def list_portfolios(user=Depends(get_current_user)):
    configs = list_portfolio_configs()
    LOGGER.info(
        "list_portfolios",
        extra={"user": user["username"], "count": len(configs)},
    )
    items = [schemas.PortfolioResponse(**cfg) for cfg in configs]
    return schemas.PortfolioListResponse(items=items)


@app.post("/v1/portfolios", response_model=schemas.PortfolioResponse, status_code=status.HTTP_201_CREATED)
def create_portfolio(payload: schemas.PortfolioBase, user=Depends(get_current_user)):
    save_portfolio_config(
        payload.name,
        payload.companies,
        payload.start_date.isoformat(),
        payload.end_date.isoformat(),
        payload.alerts or DEFAULT_ALERT_RULES,
    )
    config = get_portfolio_config(payload.name)
    if not config:
        raise HTTPException(status_code=500, detail="Portfolio not persisted")
    LOGGER.info("create_portfolio", extra={"user": user["username"], "portfolio": payload.name})
    log_portfolio_event(payload.name, "create_portfolio", actor=user["username"], payload=config)
    return schemas.PortfolioResponse(**config)


@app.delete("/v1/portfolios/{name}", status_code=status.HTTP_204_NO_CONTENT)
def remove_portfolio(name: str, user=Depends(get_current_user)):
    delete_portfolio_config(name)
    LOGGER.info("delete_portfolio", extra={"user": user["username"], "portfolio": name})
    log_portfolio_event(name, "delete_portfolio", actor=user["username"])
    return None


@app.post("/v1/portfolios/{name}/thresholds", response_model=schemas.PortfolioResponse)
def update_thresholds(name: str, payload: schemas.ThresholdUpdate, user=Depends(get_current_user)):
    if not get_portfolio_config(name):
        raise HTTPException(status_code=404, detail="Portfolio not found")
    update_portfolio_alerts(name, payload.alerts)
    LOGGER.info("update_thresholds", extra={"user": user["username"], "portfolio": name})
    config = get_portfolio_config(name)
    log_portfolio_event(name, "update_thresholds", actor=user["username"], payload=payload.alerts)
    return schemas.PortfolioResponse(**config)


@app.get("/v1/portfolios/{name}/notes", response_model=List[schemas.NoteResponse])
def list_notes(name: str, user=Depends(get_current_user)):
    if not get_portfolio_config(name):
        raise HTTPException(status_code=404, detail="Portfolio not found")
    notes = list_portfolio_annotations(name)
    return [schemas.NoteResponse(**note) for note in notes]


@app.post("/v1/portfolios/{name}/notes", response_model=schemas.NoteResponse, status_code=status.HTTP_201_CREATED)
def add_note(name: str, payload: schemas.NoteCreate, user=Depends(get_current_user)):
    if not get_portfolio_config(name):
        raise HTTPException(status_code=404, detail="Portfolio not found")
    add_portfolio_annotation(name, payload.note, payload.author or user["username"])
    LOGGER.info("add_note", extra={"user": user["username"], "portfolio": name})
    log_portfolio_event(
        name,
        "add_note",
        actor=user["username"],
        payload={"note": payload.note, "author": payload.author or user["username"]},
    )
    notes = list_portfolio_annotations(name)
    return (
        schemas.NoteResponse(**notes[0])
        if notes
        else schemas.NoteResponse(note=payload.note, author=payload.author, created_at=pd.Timestamp.utcnow())
    )


def _load_dataset(start_label: str, end_label: str):
    data = Data().read(start_label, end_label)
    df = data["data"].copy()
    df["DATE"] = pd.to_datetime(df["DATE"])
    return data, df


@app.get("/v1/analytics/portfolio", response_model=schemas.PortfolioAnalyticsResponse)
def portfolio_analytics(
    name: Optional[str] = Query(default=None),
    companies: Optional[List[str]] = Query(default=None),
    start_date: Optional[date] = Query(default=None),
    end_date: Optional[date] = Query(default=None),
    start_window: str = Query(default="dec30"),
    end_window: str = Query(default="jan12"),
    user=Depends(get_current_user),
):
    if name:
        config = get_portfolio_config(name)
        if not config:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        companies = config["companies"]
        start_date = date.fromisoformat(config["start_date"])
        end_date = date.fromisoformat(config["end_date"])
        alerts = config.get("alerts") or DEFAULT_ALERT_RULES
    else:
        if not companies:
            raise HTTPException(status_code=400, detail="Companies are required when portfolio name is not provided")
        if not start_date or not end_date:
            raise HTTPException(status_code=400, detail="Start and end dates are required")
        alerts = DEFAULT_ALERT_RULES

    data, df_raw = _load_dataset(start_window, end_window)
    snapshot = build_portfolio_snapshot(
        df_raw,
        data,
        companies,
        start_date,
        end_date,
        alerts,
    )
    LOGGER.info(
        "portfolio_analytics",
        extra={"user": user["username"], "companies": len(companies)},
    )
    records = [schemas.PortfolioAnalyticsRecord(**record) for record in snapshot["records"]]
    alert_models = [schemas.PortfolioAlert(**alert) for alert in snapshot["alerts"]]
    return schemas.PortfolioAnalyticsResponse(records=records, alerts=alert_models)


def _generate_advisory_summary(context: Dict, profile: Optional[Dict], prompt: Optional[str]) -> schemas.AdvisoryResponse:
    company_label = context.get("company_label", "The company")
    article_count = context.get("article_count", 0)
    range_label = context.get("range_label", "the selected period")
    avg_tone = context.get("avg_tone")
    positive_share = context.get("positive_share")
    tone_vs_industry = context.get("tone_vs_industry")
    busiest_day = context.get("busiest_day")
    busiest_day_count = context.get("busiest_day_count")

    executive_summary = (
        f"Between {range_label}, ESG AI monitored {article_count:,d} articles for {company_label}. "
        f"Average tone registered at {avg_tone:.2f} with positive share "
        f"{positive_share*100:.1f}%" if (avg_tone is not None and positive_share is not None) else
        f"Between {range_label}, ESG AI monitored {article_count:,d} articles for {company_label}."
    )
    if tone_vs_industry is not None:
        executive_summary += f" Tone is {tone_vs_industry:+.2f} versus industry peers."
    if busiest_day and busiest_day_count:
        executive_summary += f" Coverage peaked on {pd.to_datetime(busiest_day).strftime('%d %b %Y')} with {busiest_day_count} mentions."

    profile_notes = []
    if profile:
        mandate = profile.get("mandate")
        focus = profile.get("focus")
        region = profile.get("region")
        if mandate:
            profile_notes.append(f"Client mandate: {mandate}.")
        if focus:
            profile_notes.append(f"Priority focus: {focus}.")
        if region:
            profile_notes.append(f"Regional scope: {region}.")
    if prompt:
        profile_notes.append(f"Prompt: {prompt}.")

    talking_points = [
        executive_summary,
    ]
    if tone_vs_industry:
        talking_points.append(
            f"Tone differential vs peers: {tone_vs_industry:+.2f} (positive indicates stronger sentiment)."
        )
    if context.get("positive_vs_industry") is not None:
        talking_points.append(
            f"Positive coverage delta: {context['positive_vs_industry']*100:+.1f}pp versus industry."
        )
    if profile_notes:
        talking_points.extend(profile_notes)

    risk_radar = []
    if context.get("top_negative_articles") is not None and not context["top_negative_articles"].empty:
        top_neg = context["top_negative_articles"].head(2)
        for _, row in top_neg.iterrows():
            risk_radar.append(
                f"{pd.to_datetime(row['DATE']).strftime('%d %b %Y')} – {row['SourceCommonName']} (Tone {row['Tone']:.2f})"
            )
    negative_share = context.get("negative_share")
    if negative_share is not None:
        risk_radar.append(f"Negative/neutral coverage share: {negative_share*100:.1f}%.")

    recommended_actions = []
    if context.get("positive_share") is not None and context["positive_share"] < 0.45:
        recommended_actions.append(
            "Escalate positive messaging—less than half of coverage carries supportive tone."
        )
    if context.get("tone_change") is not None and context["tone_change"] < -0.3:
        recommended_actions.append(
            "Investigate drivers of worsening tone and prepare mitigation plan."
        )
    if not recommended_actions:
        recommended_actions.append("Maintain cadence; continue monitoring for fresh catalysts.")

    evidence = []
    recent = context.get("recent_articles")
    if recent is not None and not recent.empty:
        for _, row in recent.head(3).iterrows():
            evidence.append(
                f"{pd.to_datetime(row['DATE']).strftime('%d %b %Y')} – {row['SourceCommonName']} (Tone {row['Tone']:.2f})"
            )

    disclaimer = "AI-generated advisory draft. Review and customise before distribution."

    return schemas.AdvisoryResponse(
        executive_summary=executive_summary,
        talking_points=talking_points,
        risk_radar=risk_radar,
        recommended_actions=recommended_actions,
        evidence=evidence,
        disclaimer=disclaimer,
    )


@app.post("/v1/advisory/generate", response_model=schemas.AdvisoryResponse)
def advisory_generate(payload: schemas.AdvisoryRequest, user=Depends(get_current_user)):
    data, df_raw = _load_dataset(payload.start_window or "dec30", payload.end_window or "jan12")
    df_company = df_raw[df_raw.Organization == payload.company]
    if df_company.empty:
        raise HTTPException(status_code=404, detail="Company not found in dataset")

    start_date = payload.start_date or df_company["DATE"].min().date()
    end_date = payload.end_date or df_company["DATE"].max().date()

    context = build_company_context(
        payload.company,
        df_company.assign(DATE=df_company["DATE"].dt.date),
        df_raw.assign(DATE=df_raw["DATE"].dt.date),
        data,
        start_date,
        end_date,
    )

    response = _generate_advisory_summary(context, payload.profile, payload.prompt)
    log_portfolio_event(
        payload.profile.get("portfolio") if payload.profile else None,
        "generate_advisory",
        actor=user["username"],
        payload={"company": payload.company, "prompt": payload.prompt},
    )
    LOGGER.info("generate_advisory", extra={"user": user["username"], "company": payload.company})
    return response


@app.post("/v1/events", status_code=status.HTTP_202_ACCEPTED)
def ingest_event(event: dict, user=Depends(get_current_user)):
    portfolio_name = event.get("portfolio")
    action = event.get("action")
    payload = event.get("payload")
    if not action:
        raise HTTPException(status_code=400, detail="Action is required")
    log_portfolio_event(portfolio_name, action, actor=user["username"], payload=payload)
    return {"status": "accepted"}


@app.get("/v1/events")
def get_events(
    portfolio: Optional[str] = Query(default=None),
    actor: Optional[str] = Query(default=None),
    action: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    user=Depends(get_current_user),
):
    events = list_portfolio_events(portfolio, actor, action, limit)
    LOGGER.info(
        "list_events",
        extra={
            "user": user["username"],
            "portfolio": portfolio,
            "action": action,
            "actor": actor,
            "count": len(events),
        },
    )
    return events


__all__ = ["app"]
