"""Shared analytics helpers used across ESG AI workspaces."""

from __future__ import annotations

import numpy as np
import pandas as pd
from services.adaptive_fusion import AdaptiveFusionEngine


def filter_on_date(df: pd.DataFrame, start, end, date_col: str = "DATE") -> pd.DataFrame:
    """Return dataframe filtered between start and end dates."""

    if df.empty:
        return df
    series = pd.to_datetime(df[date_col])
    mask = (series >= pd.to_datetime(start)) & (series <= pd.to_datetime(end))
    return df.loc[mask]


def format_number(value, decimals: int = 2, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    return f"{value:.{decimals}f}{suffix}"


def format_percentage(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    return f"{value * 100:.1f}%"


def build_pillar_breakdown(df_company: pd.DataFrame) -> pd.DataFrame:
    columns = ["Pillar", "Articles", "Share", "Average Tone", "Positive Share"]
    if df_company.empty:
        return pd.DataFrame(columns=columns)
    total = len(df_company)
    mapping = {"E": "Environment", "S": "Social", "G": "Governance"}
    rows = []
    for col, label in mapping.items():
        subset = df_company[df_company[col]]
        count = len(subset)
        share = count / total if total else 0.0
        avg_tone = subset["Tone"].mean() if count else np.nan
        positive_share = (
            (subset["PositiveTone"] > subset["NegativeTone"]).mean() if count else np.nan
        )
        rows.append(
            {
                "Pillar": label,
                "Articles": int(count),
                "Share": share,
                "Average Tone": avg_tone,
                "Positive Share": positive_share,
            }
        )
    return pd.DataFrame(rows)


def build_company_context(company, df_company, df_market, data, start, end) -> dict:
    context = {
        "company": company,
        "company_label": company.title(),
        "start": start,
        "end": end,
        "range_label": f"{pd.to_datetime(start).strftime('%b %d, %Y')} — {pd.to_datetime(end).strftime('%b %d, %Y')}"
    }

    context["article_count"] = len(df_company)
    if context["article_count"] == 0:
        return context

    context["avg_tone"] = df_company["Tone"].mean()
    context["median_tone"] = df_company["Tone"].median()
    context["tone_std"] = df_company["Tone"].std()
    tone_daily = df_company.groupby("DATE")["Tone"].mean().sort_index()
    context["tone_daily"] = tone_daily
    if not tone_daily.empty:
        window = min(3, len(tone_daily))
        context["tone_early"] = tone_daily.head(window).mean()
        context["tone_recent"] = tone_daily.tail(window).mean()
        context["tone_change"] = tone_daily.iloc[-1] - tone_daily.iloc[0] if len(tone_daily) > 1 else 0.0
        context["tone_best_date"] = tone_daily.idxmax()
        context["tone_best_value"] = tone_daily.max()
        context["tone_worst_date"] = tone_daily.idxmin()
        context["tone_worst_value"] = tone_daily.min()
    else:
        context["tone_early"] = context["tone_recent"] = context["tone_change"] = None
        context["tone_best_date"] = context["tone_worst_date"] = None
        context["tone_best_value"] = context["tone_worst_value"] = None

    positive_mask = df_company["PositiveTone"] > df_company["NegativeTone"]
    context["positive_share"] = positive_mask.mean() if len(df_company) else None
    context["negative_share"] = (1 - context["positive_share"]) if context["positive_share"] is not None else None
    context["avg_polarity"] = df_company["Polarity"].mean()
    context["avg_activity"] = df_company["ActivityDensity"].mean()
    context["avg_wordcount"] = df_company["WordCount"].mean()

    context["publisher_counts"] = df_company["SourceCommonName"].value_counts().head(5)
    context["recent_articles"] = df_company.sort_values("DATE", ascending=False).head(5)
    context["top_positive_articles"] = df_company.sort_values("Tone", ascending=False).head(3)
    context["top_negative_articles"] = df_company.sort_values("Tone", ascending=True).head(3)

    source_summary = (
        df_company.assign(_positive=positive_mask)
        .groupby("SourceCommonName")
        .agg(
            Articles=("SourceCommonName", "size"),
            AvgTone=("Tone", "mean"),
            PositiveShare=("_positive", "mean"),
            LastMention=("DATE", "max"),
        )
        .sort_values("Articles", ascending=False)
    )
    context["source_summary"] = source_summary

    context["daily_volume"] = df_company.groupby("DATE").size().sort_index()
    if not context["daily_volume"].empty:
        context["busiest_day"] = context["daily_volume"].idxmax()
        context["busiest_day_count"] = int(context["daily_volume"].max())
    else:
        context["busiest_day"] = None
        context["busiest_day_count"] = None

    context["industry_tone_avg"] = df_market["Tone"].mean() if not df_market.empty else None
    industry_pos_mask = df_market["PositiveTone"] > df_market["NegativeTone"] if not df_market.empty else pd.Series(dtype=float)
    context["industry_positive_share"] = industry_pos_mask.mean() if len(industry_pos_mask) else None
    context["industry_polarity"] = df_market["Polarity"].mean() if not df_market.empty else None
    if context["industry_tone_avg"] is not None:
        context["tone_vs_industry"] = context["avg_tone"] - context["industry_tone_avg"]
    else:
        context["tone_vs_industry"] = None
    if context.get("positive_share") is not None and context.get("industry_positive_share") is not None:
        context["positive_vs_industry"] = context["positive_share"] - context["industry_positive_share"]
    else:
        context["positive_vs_industry"] = None

    esg_matrix = data.get("ESG")
    context["esg_scores"] = {}
    context["esg_industry"] = {}
    if esg_matrix is not None:
        esg_df = esg_matrix.set_index("Unnamed: 0")
        if company in esg_df.columns:
            comp_scores = esg_df[company]
            industry_scores = esg_df.select_dtypes(include=[np.number]).mean(axis=1)
            context["esg_scores"] = comp_scores.to_dict()
            context["esg_industry"] = industry_scores.to_dict()

    # --- INVENTIVE FEATURE: Adaptive Fusion Engine Integration ---
    # Compute Structured Composite
    if context.get("esg_scores"):
        structured_score = np.nanmean(list(context["esg_scores"].values()))
    else:
        structured_score = 0.0

    # Compute Signal Momentum
    momentum = context.get("tone_change", 0.0) or 0.0

    # Execute Adaptive Fusion
    fusion = AdaptiveFusionEngine()
    fusion_result = fusion.fuse(
        avg_tone=context.get("avg_tone", 0.0) or 0.0,
        structured_score=float(structured_score),
        momentum=float(momentum),
        tone_std=context.get("tone_std", 0.0) or 0.0,
        article_count=context.get("article_count", 0)
    )

    context["final_esg_score"] = fusion_result["final_esg_score"]
    context["fusion_weights"] = fusion_result["weights"]
    context["reliability_score"] = fusion_result["reliability_metric"]
    # -----------------------------------------------------------

    return context


def identify_catalyst_timeline(df_company: pd.DataFrame, limit: int = 4):
    if df_company.empty:
        return []

    df = df_company.copy()
    df["_positive"] = df["PositiveTone"] > df["NegativeTone"]
    daily = (
        df.groupby("DATE")
        .agg(
            volume=("Tone", "size"),
            avg_tone=("Tone", "mean"),
            positive_share=("_positive", "mean"),
        )
        .sort_index()
    )

    source_counts = (
        df.groupby(["DATE", "SourceCommonName"])
        .size()
        .reset_index(name="count")
    )
    if source_counts.empty:
        top_sources = pd.DataFrame(columns=["DATE", "SourceCommonName", "count"]).set_index("DATE")
    else:
        top_source_idx = source_counts.groupby("DATE")["count"].idxmax()
        top_sources = source_counts.loc[top_source_idx].set_index("DATE")

    overall_tone = df["Tone"].mean()
    events = []
    for date, row in daily.iterrows():
        day_df = df[df["DATE"] == date]
        if day_df.empty:
            continue
        polarity_series = day_df["Polarity"].abs().dropna()
        if polarity_series.empty:
            highlight_idx = day_df.index[0]
        else:
            highlight_idx = polarity_series.idxmax()
        try:
            highlight_row = day_df.loc[[highlight_idx]].iloc[0]
        except KeyError:
            highlight_row = day_df.iloc[0]
        pillars = []
        for col, label in [("E", "E"), ("S", "S"), ("G", "G")]:
            if day_df[col].any():
                pillars.append(label)
        top_source = top_sources.loc[date] if date in top_sources.index else None
        impact = row.volume * (abs(row.avg_tone - overall_tone) + 0.1)
        events.append(
            {
                "date": date,
                "volume": int(row.volume),
                "avg_tone": row.avg_tone,
                "positive_share": row.positive_share,
                "pillars": pillars,
                "top_source": top_source["SourceCommonName"] if top_source is not None else None,
                "top_source_articles": int(top_source["count"]) if top_source is not None else None,
                "highlight_url": highlight_row.get("URL"),
                "highlight_source": highlight_row.get("SourceCommonName"),
                "highlight_tone": highlight_row.get("Tone"),
                "impact_score": impact,
            }
        )

    events.sort(key=lambda item: item["impact_score"], reverse=True)
    return events[:limit]


__all__ = [
    "filter_on_date",
    "format_number",
    "format_percentage",
    "build_pillar_breakdown",
    "build_company_context",
    "identify_catalyst_timeline",
]
