import os
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime
from typing import Optional

from download_data import Data
from services.storage import (
    initialise_storage,
    save_portfolio_config,
    list_portfolio_configs,
    get_portfolio_config,
    delete_portfolio_config,
    update_portfolio_alerts,
    add_portfolio_annotation,
    list_portfolio_annotations,
)
from utils.logging import get_logger
from utils.auth import ensure_authenticated, render_user_controls
from utils.analytics import (
    filter_on_date,
    format_number,
    format_percentage,
    build_company_context,
    build_pillar_breakdown,
    identify_catalyst_timeline,
)


from services.portfolio import (
    DEFAULT_ALERT_RULES,
    build_portfolio_snapshot,
)

LOGGER = get_logger(__name__)
initialise_storage()
SESSION_USER = ensure_authenticated(allowed_roles=["admin", "analyst", "viewer"])
render_user_controls(SESSION_USER)
st.sidebar.markdown("---")


ROLE_CAPABILITIES = {
    "admin": {
        "can_save": True,
        "can_delete": True,
        "can_update_thresholds": True,
        "can_annotate": True,
        "can_adjust_threshold_sliders": True,
    },
    "analyst": {
        "can_save": True,
        "can_delete": False,
        "can_update_thresholds": True,
        "can_annotate": True,
        "can_adjust_threshold_sliders": True,
    },
    "viewer": {
        "can_save": False,
        "can_delete": False,
        "can_update_thresholds": False,
        "can_annotate": False,
        "can_adjust_threshold_sliders": False,
    },
}

USER_ROLE = SESSION_USER.get("role", "viewer")
CAP = ROLE_CAPABILITIES.get(USER_ROLE, ROLE_CAPABILITIES["viewer"])
API_BASE_URL = os.getenv('ESG_AI_API_BASE', 'http://127.0.0.1:8000/v1')

SESSION_COMPANIES = "portfolio_selected_companies"
SESSION_DATES = "portfolio_selected_dates"
SESSION_NAME = "portfolio_active_name"
SESSION_ALERTS = "portfolio_alert_rules"

AUDIT_ACTIONS = [
    "All",
    "save_portfolio",
    "delete_portfolio",
    "update_thresholds",
    "add_note",
    "generate_advisory",
    "export_advisory",
]



@st.cache_data(show_spinner=False)
def load_dataset(start_label: str, end_label: str):
    data = Data().read(start_label, end_label)
    df = data["data"].copy()
    df["DATE"] = pd.to_datetime(df["DATE"])
    LOGGER.info(
        "dataset_loaded",
        extra={"start": start_label, "end": end_label, "records": len(df)},
    )
    return data, df

def build_portfolio_charts(df, selected):
    plot_df = df[df.Organization.isin(selected)].copy()
    if plot_df.empty:
        return None, None
    trend_df = (
        plot_df.groupby(["DATE", "Organization"])["Tone"].mean().reset_index()
    )
    volume_df = (
        plot_df.groupby(["DATE", "Organization"]).size().reset_index(name="Articles")
    )

    tone_chart = (
        alt.Chart(trend_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("DATE:T", title="Date"),
            y=alt.Y("Tone:Q", title="Average tone"),
            color="Organization:N",
            tooltip=["Organization", alt.Tooltip("Tone", format=".2f"), "DATE"],
        )
        .properties(height=320)
        .interactive()
    )

    volume_chart = (
        alt.Chart(volume_df)
        .mark_area(opacity=0.6)
        .encode(
            x=alt.X("DATE:T", title="Date"),
            y=alt.Y("Articles:Q", title="Article volume"),
            color="Organization:N",
            tooltip=["Organization", "Articles", "DATE"],
        )
        .properties(height=260)
        .interactive()
    )

    return tone_chart, volume_chart


def build_catalyst_digest(df, selected, window=3):
    rows = []
    for company in selected:
        slice_df = df[df.Organization == company]
        if slice_df.empty:
            continue
        catalysts = identify_catalyst_timeline(slice_df)
        for event in catalysts[:window]:
            rows.append(
                {
                    "Company": company,
                    "Date": pd.to_datetime(event["date"]).strftime("%d %b %Y"),
                    "Tone": format_number(event.get("avg_tone")),
                    "Lead source": event.get("top_source") or event.get("highlight_source") or "—",
                    "Pillars": ", ".join(event.get("pillars", [])) or "—",
                }
            )
    return pd.DataFrame(rows)

with st.sidebar:
    st.header("Portfolio Controls")
    available_dir = set()
    try:
        available_dir = set(os.listdir("Data"))
    except FileNotFoundError:
        pass

    # Filter to only actual directories (exclude symlinks and files)
    available_datasets = [d for d in available_dir if "_to_" in d]
    
    # Prefer synthetic_2021_to_now if available, otherwise use dec30_to_jan12
    default_dataset = "synthetic_2021_to_now" if "synthetic_2021_to_now" in available_datasets else "dec30_to_jan12"
    
    if available_datasets:
        # Show full dataset name, but store it properly for loading
        dataset_choice = st.selectbox(
            "Data window",
            options=sorted(available_datasets),
            index=sorted(available_datasets).index(default_dataset) if default_dataset in available_datasets else 0,
            format_func=lambda x: x.replace("synthetic_", "").replace("_to_", " to ").replace("_", " ").title()
        )
        # Split the dataset name for the data loader
        start_label, end_label = dataset_choice.split("_to_")
    else:
        start_label, end_label = "dec30", "jan12"

    data, df_raw = load_dataset(start_label, end_label)

    min_date = df_raw["DATE"].min()
    max_date = df_raw["DATE"].max()
    default_dates = st.session_state.get(
        SESSION_DATES,
        (min_date.date(), max_date.date()),
    )
    start_date, end_date = st.date_input(
        "Reporting period",
        value=default_dates,
        min_value=min_date.date(),
        max_value=max_date.date(),
    )
    st.session_state[SESSION_DATES] = (start_date, end_date)

    all_companies = sorted(df_raw.Organization.unique())
    saved_selection = st.session_state.get(SESSION_COMPANIES, all_companies[:5])
    default_selection = [c for c in saved_selection if c in all_companies]
    if not default_selection:
        default_selection = all_companies[:5]
    selected_companies = st.multiselect(
        "Portfolio companies",
        all_companies,
        default=default_selection,
        help="Select up to 10 companies to monitor",
    )
    st.session_state[SESSION_COMPANIES] = selected_companies

    if len(selected_companies) > 10:
        st.warning("Only the first 10 selected companies will be analysed.")
        selected_companies = selected_companies[:10]
        st.session_state[SESSION_COMPANIES] = selected_companies

    stored_alerts = st.session_state.get(SESSION_ALERTS, DEFAULT_ALERT_RULES.copy())
    if not isinstance(stored_alerts, dict):
        stored_alerts = DEFAULT_ALERT_RULES.copy()

    with st.expander("Risk thresholds", expanded=False):
        if not CAP["can_adjust_threshold_sliders"]:
            st.caption("Read-only view. Thresholds controlled by administrators.")
        tone_floor = st.slider(
            "Tone floor (critical)",
            min_value=-5.0,
            max_value=2.0,
            value=float(stored_alerts.get("tone_floor", DEFAULT_ALERT_RULES["tone_floor"])),
            step=0.05,
            help="If average tone falls below this value the issuer is flagged.",
            disabled=not CAP["can_adjust_threshold_sliders"],
        )
        positive_share_min = st.slider(
            "Minimum positive share",
            min_value=0.0,
            max_value=1.0,
            value=float(stored_alerts.get("positive_share_min", DEFAULT_ALERT_RULES["positive_share_min"])),
            step=0.01,
            disabled=not CAP["can_adjust_threshold_sliders"],
        )
        tone_vs_industry_min = st.slider(
            "Tone vs industry floor",
            min_value=-2.0,
            max_value=1.0,
            value=float(stored_alerts.get("tone_vs_industry_min", DEFAULT_ALERT_RULES["tone_vs_industry_min"])),
            step=0.05,
            help="Trigger if tone lags industry average by this amount.",
            disabled=not CAP["can_adjust_threshold_sliders"],
        )
        tone_change_min = st.slider(
            "Tone momentum floor",
            min_value=-2.0,
            max_value=1.0,
            value=float(stored_alerts.get("tone_change_min", DEFAULT_ALERT_RULES["tone_change_min"])),
            step=0.05,
            help="Trigger if tone deteriorates by this amount across the window.",
            disabled=not CAP["can_adjust_threshold_sliders"],
        )

    current_alerts = {
        "tone_floor": tone_floor,
        "positive_share_min": positive_share_min,
        "tone_vs_industry_min": tone_vs_industry_min,
        "tone_change_min": tone_change_min,
    }
    st.session_state[SESSION_ALERTS] = current_alerts

    portfolio_name = st.text_input(
        "Portfolio label",
        value=st.session_state.get(SESSION_NAME, ""),
        help="Save the current configuration for later recall.",
    )

    configs = list_portfolio_configs()
    saved_names = [cfg["name"] for cfg in configs]

    save_col, manage_col = st.columns(2)
    with save_col:
        if not CAP["can_save"]:
            st.caption("Sign in as an analyst or admin to save portfolios.")
        if st.button("Save portfolio", use_container_width=True, disabled=not CAP["can_save"]):
            if not portfolio_name:
                st.warning("Provide a portfolio label before saving.")
            elif not selected_companies:
                st.warning("Select at least one company before saving.")
            else:
                save_portfolio_config(
                    portfolio_name,
                    selected_companies,
                    start_date.isoformat(),
                    end_date.isoformat(),
                    current_alerts,
                )
                st.session_state[SESSION_NAME] = portfolio_name
                LOGGER.info(
                    "portfolio_saved",
                    extra={
                        "portfolio": portfolio_name,
                        "company_count": len(selected_companies),
                    },
                )
                _record_event(
                    "save_portfolio",
                    portfolio_name,
                    {
                        "companies": selected_companies,
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "alerts": current_alerts,
                    },
                )
                st.success(f"Portfolio '{portfolio_name}' saved.")

    with manage_col:
        load_options = ["—"] + saved_names
        current_index = 0
        if st.session_state.get(SESSION_NAME) in saved_names:
            current_index = load_options.index(st.session_state[SESSION_NAME])
        load_choice = st.selectbox(
            "Saved portfolios",
            options=load_options,
            index=current_index,
        )
        load_pressed = st.button("Load", use_container_width=True)
        delete_pressed = st.button(
            "Delete",
            use_container_width=True,
            key="delete_portfolio",
            disabled=not CAP["can_delete"],
        )
        update_pressed = st.button(
            "Update thresholds",
            use_container_width=True,
            key="update_alerts",
            disabled=not CAP["can_update_thresholds"],
        )

    if load_pressed and load_choice != "—":
        config = get_portfolio_config(load_choice)
        if config:
            st.session_state[SESSION_COMPANIES] = [
                company for company in config["companies"] if company in all_companies
            ]
            st.session_state[SESSION_DATES] = (
                pd.to_datetime(config["start_date"]).date(),
                pd.to_datetime(config["end_date"]).date(),
            )
            st.session_state[SESSION_NAME] = config["name"]
            if config.get("alerts"):
                st.session_state[SESSION_ALERTS] = {
                    **DEFAULT_ALERT_RULES,
                    **config["alerts"],
                }
            st.experimental_rerun()
        else:
            st.error("Selected portfolio could not be loaded.")

    if delete_pressed and CAP["can_delete"] and load_choice != "—":
        delete_portfolio_config(load_choice)
        if st.session_state.get(SESSION_NAME) == load_choice:
            st.session_state.pop(SESSION_NAME, None)
        LOGGER.info("portfolio_deleted", extra={"portfolio": load_choice})
        _record_event("delete_portfolio", load_choice)
        st.experimental_rerun()

    if update_pressed and CAP["can_update_thresholds"]:
        target = portfolio_name or (load_choice if load_choice != "—" else None)
        if not target:
            st.warning("Select or name a portfolio before updating thresholds.")
        else:
            update_portfolio_alerts(target, current_alerts)
            LOGGER.info("portfolio_alerts_updated", extra={"portfolio": target})
            _record_event("update_thresholds", target, current_alerts)
            st.success("Alert thresholds updated.")

    if not selected_companies:
        st.warning("Select at least one company to build the portfolio view.")
        st.stop()

    st.markdown("---")
    st.caption("Download portfolio tables from the main view below.")

    with st.expander("Saved portfolios directory", expanded=False):
        if configs:
            catalog_df = pd.DataFrame(
                [
                    {
                        "Name": cfg["name"],
                        "Companies": ", ".join(cfg["companies"][:5]) + ("…" if len(cfg["companies"]) > 5 else ""),
                        "Updated": cfg["updated_at"],
                    }
                    for cfg in configs
                ]
            )
            st.table(catalog_df)
        else:
            st.caption("No saved portfolios yet.")


st.title("Portfolio Command Center")
st.caption("Enterprise-grade ESG sentiment surveillance for multi-company coverage.")


import requests
from requests import RequestException


def _api_get(path: str):
    url = f"{API_BASE_URL}{path}"
    response = requests.get(
        url,
        headers={"X-User": SESSION_USER["username"], "X-Role": SESSION_USER["role"]},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def _api_post(path: str, payload: dict):
    response = requests.post(
        f"{API_BASE_URL}{path}",
        json=payload,
        headers={"X-User": SESSION_USER["username"], "X-Role": SESSION_USER["role"]},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def _record_event(action: str, portfolio_name: Optional[str] = None, payload: Optional[dict] = None):
    try:
        _api_post(
            "/events",
            {
                "action": action,
                "portfolio": portfolio_name,
                "payload": payload or {},
            },
        )
    except RequestException:
        LOGGER.warning("event_record_failed", extra={"action": action, "portfolio": portfolio_name})


def _api_delete(path: str):
    response = requests.delete(
        f"{API_BASE_URL}{path}",
        headers={"X-User": SESSION_USER["username"], "X-Role": SESSION_USER["role"]},
        timeout=10,
    )
    if response.status_code not in {200, 204}:
        response.raise_for_status()


def call_events_api(portfolio_name=None, actor=None, action=None, limit=100):
    params = {"limit": limit}
    if portfolio_name:
        params["portfolio"] = portfolio_name
    if actor:
        params["actor"] = actor
    if action and action != "All":
        params["action"] = action
    try:
        response = requests.get(
            f"{API_BASE_URL}/events",
            params=params,
            headers={"X-User": SESSION_USER["username"], "X-Role": SESSION_USER["role"]},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except RequestException:
        LOGGER.warning("events_api_unavailable")
        return []


def _compute_local_snapshot(df_raw, data, companies, start_date, end_date, alerts):
    snapshot = build_portfolio_snapshot(
        df_raw,
        data,
        companies,
        start_date,
        end_date,
        alerts,
    )
    return snapshot


with st.spinner("Synthesising portfolio insights…"):
    try:
        if st.session_state.get(SESSION_NAME):
            payload = _api_get(f"/analytics/portfolio?name={st.session_state[SESSION_NAME]}")
        else:
            payload = _api_post(
                "/analytics/portfolio",
                {
                    "companies": selected_companies,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                },
            )
        records = payload["records"]
        alerts = payload["alerts"]
    except RequestException:
        LOGGER.warning("api_unavailable_falling_back", extra={"endpoint": "/analytics/portfolio"})
        local_snapshot = _compute_local_snapshot(
            df_raw,
            data,
            selected_companies,
            start_date,
            end_date,
            current_alerts,
        )
        records = local_snapshot["records"]
        alerts = local_snapshot["alerts"]
    LOGGER.info(
        "portfolio_synthesised",
        extra={
            "portfolio": st.session_state.get(SESSION_NAME),
            "companies": len(selected_companies),
            "alerts": len(alerts),
        },
    )

raw_df = pd.DataFrame(records)
portfolio_df = pd.DataFrame(
    [
        {
            "Company": record["company"].title(),
            "Articles": record["article_count"],
            "Average tone": format_number(record.get("avg_tone")),
            "Positive share": format_percentage(record.get("positive_share")),
            "Tone vs industry": format_number(record.get("tone_vs_industry")),
            "Risk": record.get("risk_rating"),
            "Insight": record.get("risk_headline"),
        }
        for record in records
    ]
)

# === Portfolio Snapshot ===
st.subheader("Portfolio snapshot")
metrics_row = st.columns(4)
portfolio_articles = raw_df["article_count"].sum()
positive_values = [val for val in raw_df["positive_share"] if val is not None]
portfolio_positive_mean = sum(positive_values) / len(positive_values) if positive_values else float("nan")
avg_tones = [val for val in raw_df["avg_tone"] if val is not None]
portfolio_tone_mean = sum(avg_tones) / len(avg_tones) if avg_tones else float("nan")

with metrics_row[0]:
    st.metric("Companies tracked", len(selected_companies))
with metrics_row[1]:
    st.metric("Articles in window", f"{portfolio_articles:,}")
with metrics_row[2]:
    st.metric("Avg tone", "n/a" if pd.isna(portfolio_tone_mean) else f"{portfolio_tone_mean:.2f}")
with metrics_row[3]:
    st.metric(
        "Avg positive share",
        "n/a" if pd.isna(portfolio_positive_mean) else f"{portfolio_positive_mean*100:.1f}%",
    )

st.dataframe(portfolio_df, use_container_width=True)

csv_download = portfolio_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download portfolio table (CSV)",
    csv_download,
    file_name=f"esg_portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv",
)

with st.expander("Workflow guidance", expanded=False):
    st.markdown(
        """
        - Share the CSV directly with client teams or ingest it into internal dashboards.
        - Use the risk alerts below to prioritise deeper dives and populate issue logs.
        - Pair with the Catalyst digest for meeting prep and narrative walkthroughs.
        """
    )


# === Alerts ===
st.subheader("Risk alerts")
if alerts:
    for alert in alerts:
        color = "⚠️" if alert["rating"] == "Watch" else "🚨"
        st.markdown(f"{color} **{alert['company']}** — {alert['headline']}")
else:
    st.info("No elevated risk alerts for the selected companies.")


active_portfolio = st.session_state.get(SESSION_NAME) or (portfolio_name if portfolio_name else None)
st.subheader("Engagement notes")
if active_portfolio:
    existing_notes = list_portfolio_annotations(active_portfolio)
    if existing_notes:
        for note in existing_notes:
            author = note.get("author") or "Analyst"
            st.markdown(
                f"- {note['created_at']} · {author}: {note['note']}"
            )
    else:
        st.caption("No notes logged yet for this portfolio.")

    if CAP["can_annotate"]:
        note_col, author_col = st.columns([3, 1])
        with note_col:
            new_note = st.text_input(
                "Log a note",
                key="portfolio_note_text",
                placeholder="Capture follow-ups, client actions, or risk rationale.",
            )
        with author_col:
            new_author = st.text_input(
                "Author",
                key="portfolio_note_author",
                value=SESSION_USER["username"],
            )
        if st.button("Add note", key="portfolio_add_note"):
            if not new_note.strip():
                st.warning("Enter a note before saving.")
            else:
                add_portfolio_annotation(active_portfolio, new_note.strip(), new_author.strip() or None)
                LOGGER.info(
                    "portfolio_note_added",
                    extra={
                        "portfolio": active_portfolio,
                        "author": new_author.strip() or SESSION_USER["username"],
                    },
                )
                _record_event(
                    "add_note",
                    active_portfolio,
                    {"note": new_note.strip(), "author": new_author.strip() or SESSION_USER["username"]},
                )
                st.success("Note captured.")
                st.experimental_rerun()
    else:
        st.caption("View-only mode. Analysts or admins can log engagement notes.")
else:
    st.caption("Name and save a portfolio to start logging engagement notes.")

if USER_ROLE == "admin":
    st.subheader("Audit log")
    filter_cols = st.columns([2, 2, 2, 1])
    with filter_cols[0]:
        audit_portfolio = st.text_input(
            "Portfolio filter",
            value=st.session_state.get(SESSION_NAME, ""),
            key="audit_portfolio_filter",
        )
    with filter_cols[1]:
        audit_user = st.text_input(
            "User filter",
            value=st.session_state.get("audit_user_filter", ""),
        )
    with filter_cols[2]:
        audit_action = st.selectbox(
            "Action",
            AUDIT_ACTIONS,
            index=AUDIT_ACTIONS.index(st.session_state.get("audit_action_filter", AUDIT_ACTIONS[0]))
            if st.session_state.get("audit_action_filter") in AUDIT_ACTIONS
            else 0,
        )
    with filter_cols[3]:
        audit_limit = st.number_input(
            "Rows",
            min_value=10,
            max_value=200,
            step=10,
            value=st.session_state.get("audit_limit", 50),
        )

    refresh_clicked = st.button("Refresh audit log")

    current_filters = {
        "portfolio": audit_portfolio.strip() if audit_portfolio else None,
        "actor": audit_user.strip() if audit_user else None,
        "action": audit_action,
        "limit": int(audit_limit),
    }

    if (
        refresh_clicked
        or "audit_events_cache" not in st.session_state
        or st.session_state.get("audit_filters") != current_filters
    ):
        events = call_events_api(
            current_filters["portfolio"],
            current_filters["actor"],
            current_filters["action"],
            current_filters["limit"],
        )
        st.session_state["audit_events_cache"] = events
        st.session_state["audit_filters"] = current_filters
    else:
        events = st.session_state.get("audit_events_cache", [])

    if events:
        events_df = pd.DataFrame(events)
        events_df["created_at"] = pd.to_datetime(events_df["created_at"], errors="coerce")
        events_df.sort_values("created_at", ascending=False, inplace=True)
        st.dataframe(events_df, use_container_width=True)
        st.caption(f"Showing {len(events_df)} events")
    else:
        st.caption("No events recorded yet for this selection.")


# === Charts ===
tone_chart, volume_chart = build_portfolio_charts(
    filter_on_date(df_raw, start_date, end_date, "DATE"), selected_companies
)

chart_cols = st.columns(2)
with chart_cols[0]:
    if tone_chart is not None:
        st.altair_chart(tone_chart, use_container_width=True)
    else:
        st.caption("Tone data unavailable for the selection.")
with chart_cols[1]:
    if volume_chart is not None:
        st.altair_chart(volume_chart, use_container_width=True)
    else:
        st.caption("Volume data unavailable for the selection.")


# === Catalyst digest ===
st.subheader("Catalyst digest")
catalyst_df = build_catalyst_digest(filter_on_date(df_raw, start_date, end_date, "DATE"), selected_companies)
if not catalyst_df.empty:
    st.table(catalyst_df)
else:
    st.caption("No significant catalysts identified across the portfolio.")


# === ESG Pillar exposure ===
st.subheader("ESG pillar exposure")
pillar_rows = []
date_filtered = filter_on_date(df_raw, start_date, end_date, "DATE")
for company in selected_companies:
    slice_df = date_filtered[date_filtered.Organization == company]
    pillar_table = build_pillar_breakdown(slice_df)
    if pillar_table.empty:
        continue
    for record in pillar_table.to_dict("records"):
        pillar_rows.append(
            {
                "Company": company,
                "Pillar": record["Pillar"],
                "Articles": record["Articles"],
                "Share": f"{record['Share']*100:.1f}%",
                "Average tone": "n/a" if pd.isna(record["Average Tone"]) else f"{record['Average Tone']:.2f}",
            }
        )

if pillar_rows:
    st.table(pd.DataFrame(pillar_rows))
else:
    st.caption("No ESG pillar tagging available for the selected portfolio.")


st.markdown("---")
st.caption("Portfolio Command Center · ESG AI")
