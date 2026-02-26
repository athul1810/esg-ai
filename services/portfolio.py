"""Portfolio analytics helpers shared across UI and backend."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from utils.analytics import (
    build_company_context,
    filter_on_date,
)


DEFAULT_ALERT_RULES = {
    "fusion_score_floor": -0.3,  # New threshold based on fused score
    "reliability_min": 0.3,
    "tone_floor": -0.5,
    "positive_share_min": 0.45,
    "tone_vs_industry_min": -0.4,
    "tone_change_min": -0.4,
}


def compute_risk_signal(context: Dict, alert_rules: Optional[Dict] = None) -> Tuple[str, str, int]:
    rules = alert_rules or DEFAULT_ALERT_RULES

    article_count = context.get("article_count") or 0
    avg_tone = context.get("avg_tone")
    positive_share = context.get("positive_share")
    tone_vs_industry = context.get("tone_vs_industry")
    tone_change = context.get("tone_change")

    score = 0
    notes: List[str] = []

    # --- INVENTIVE FEATURE: Scoring from Adaptive Fusion ---
    final_esg_score = context.get("final_esg_score")
    reliability = context.get("reliability_score")
    fusion_score_floor = rules.get("fusion_score_floor", -0.3)
    reliability_min = rules.get("reliability_min", 0.3)

    if final_esg_score is not None:
        if final_esg_score <= fusion_score_floor - 0.2:
            notes.append("Severe fused ESG risk")
            score += 5
        elif final_esg_score <= fusion_score_floor:
            notes.append("Elevated adaptive risk")
            score += 3
            
    if reliability is not None and reliability < reliability_min:
        notes.append("Low signal reliability")
        score += 1
    # ------------------------------------------------------

    tone_floor = rules.get("tone_floor", -0.5)
    tone_vs_industry_min = rules.get("tone_vs_industry_min", -0.4)
    tone_change_min = rules.get("tone_change_min", -0.4)
    positive_share_min = rules.get("positive_share_min", 0.45)

    if avg_tone is not None:
        if avg_tone <= tone_floor - 0.2:
            notes.append("Net negative tone")
            score += 2  # Reduced weight since fusion handles it now
        elif avg_tone <= tone_floor:
            notes.append("Tone below neutral")
            score += 1

    if positive_share is not None and positive_share < positive_share_min:
        notes.append("Positive share under threshold")
        score += 2

    if tone_vs_industry is not None and tone_vs_industry < tone_vs_industry_min:
        notes.append("Lagging peer tone")
        score += 2

    if tone_change is not None and tone_change < tone_change_min:
        notes.append("Momentum deterioration")
        score += 2

    if score >= 6:
        rating = "Critical"
    elif score >= 3:
        rating = "Watch"
    else:
        rating = "Stable"

    headline = ", ".join(notes) if notes else "No major alerts"
    return rating, headline, score


def build_portfolio_snapshot(
    df: pd.DataFrame,
    data: Dict,
    companies: Iterable[str],
    start_date,
    end_date,
    alert_rules: Optional[Dict] = None,
) -> Dict[str, List[Dict]]:
    records: List[Dict] = []
    alerts: List[Dict] = []

    date_filtered = filter_on_date(df, start_date, end_date, "DATE")
    market_scope = date_filtered.copy()

    for company in companies:
        company_slice = date_filtered[date_filtered.Organization == company]
        if company_slice.empty:
            records.append(
                {
                    "company": company,
                    "article_count": 0,
                    "avg_tone": None,
                    "positive_share": None,
                    "tone_vs_industry": None,
                    "risk_rating": "No data",
                    "risk_headline": "No filtered coverage for this window.",
                    "risk_score": 0,
                }
            )
            continue

        context = build_company_context(
            company,
            company_slice.assign(DATE=company_slice["DATE"].dt.date),
            market_scope.assign(DATE=market_scope["DATE"].dt.date),
            data,
            start_date,
            end_date,
        )

        rating, headline, score = compute_risk_signal(context, alert_rules)

        record = {
            "company": company,
            "article_count": int(context.get("article_count", 0)),
            "avg_tone": context.get("avg_tone"),
            "final_esg_score": context.get("final_esg_score"),
            "reliability_score": context.get("reliability_score"),
            "positive_share": context.get("positive_share"),
            "tone_vs_industry": context.get("tone_vs_industry"),
            "risk_rating": rating,
            "risk_headline": headline,
            "risk_score": score,
        }
        records.append(record)

        if rating != "Stable":
            alerts.append(
                {
                    "company": company,
                    "rating": rating,
                    "score": score,
                    "headline": headline,
                }
            )

    alerts.sort(key=lambda item: item["score"], reverse=True)
    return {"records": records, "alerts": alerts}


__all__ = [
    "DEFAULT_ALERT_RULES",
    "compute_risk_signal",
    "build_portfolio_snapshot",
]

