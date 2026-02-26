"""Lightweight persistence utilities using SQLite."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Optional

from config.settings import get_settings
from utils.logging import get_logger


LOGGER = get_logger(__name__)


@contextmanager
def get_connection():
    settings = get_settings()
    database_url = settings["DATABASE_URL"]
    if not database_url.startswith("sqlite:///"):
        raise ValueError("Only sqlite databases are supported in the current implementation.")

    path = database_url.replace("sqlite:///", "")
    conn = sqlite3.connect(path)
    try:
        yield conn
    finally:
        conn.close()


def initialise_storage() -> None:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio_configs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                companies TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                alerts TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio_annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_name TEXT NOT NULL,
                author TEXT,
                note TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(portfolio_name) REFERENCES portfolio_configs(name) ON DELETE CASCADE
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_name TEXT,
                action TEXT NOT NULL,
                actor TEXT,
                payload TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_portfolio_config(name: str, companies: List[str], start_date: str, end_date: str, alerts: Optional[Dict] = None) -> None:
    payload = json.dumps(alerts or {})
    now = datetime.utcnow().isoformat()

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO portfolio_configs (
                name, companies, start_date, end_date, alerts, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                companies=excluded.companies,
                start_date=excluded.start_date,
                end_date=excluded.end_date,
                alerts=excluded.alerts,
                updated_at=excluded.updated_at
            """,
            (
                name,
                json.dumps(companies),
                start_date,
                end_date,
                payload,
                now,
                now,
            ),
        )
        conn.commit()
    LOGGER.info(
        "saved_portfolio_configuration",
        extra={"portfolio": name, "companies": companies},
    )


def update_portfolio_alerts(name: str, alerts: Dict) -> None:
    payload = json.dumps(alerts or {})
    now = datetime.utcnow().isoformat()

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE portfolio_configs
            SET alerts = ?, updated_at = ?
            WHERE name = ?
            """,
            (payload, now, name),
        )
        conn.commit()
    LOGGER.info("updated_portfolio_alerts", extra={"portfolio": name, "alerts": alerts})


def list_portfolio_configs() -> List[Dict]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT name, companies, start_date, end_date, alerts, updated_at
            FROM portfolio_configs
            ORDER BY updated_at DESC
            """
        )
        rows = cursor.fetchall()

    configs = []
    for name, companies_json, start_date, end_date, alerts_json, updated_at in rows:
        alerts_payload = json.loads(alerts_json or "{}")
        if isinstance(alerts_payload, dict) and "alerts" in alerts_payload:
            alerts_payload = alerts_payload.get("alerts", {})
        configs.append(
            {
                "name": name,
                "companies": json.loads(companies_json),
                "start_date": start_date,
                "end_date": end_date,
                "alerts": alerts_payload,
                "updated_at": updated_at,
            }
        )
    return configs


def get_portfolio_config(name: str) -> Optional[Dict]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT name, companies, start_date, end_date, alerts, updated_at
            FROM portfolio_configs
            WHERE name = ?
            """,
            (name,),
        )
        row = cursor.fetchone()

    if row is None:
        return None

    name, companies_json, start_date, end_date, alerts_json, updated_at = row
    alerts_payload = json.loads(alerts_json or "{}")
    if "alerts" in alerts_payload:
        alerts_payload = alerts_payload.get("alerts", {})
    return {
        "name": name,
        "companies": json.loads(companies_json),
        "start_date": start_date,
        "end_date": end_date,
        "alerts": alerts_payload,
        "updated_at": updated_at,
    }


def delete_portfolio_config(name: str) -> None:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM portfolio_configs WHERE name = ?", (name,))
        conn.commit()
    LOGGER.info("deleted_portfolio_configuration", extra={"portfolio": name})


def add_portfolio_annotation(name: str, note: str, author: Optional[str] = None) -> None:
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO portfolio_annotations (portfolio_name, author, note, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (name, author, note, now),
        )
        conn.commit()
    LOGGER.info("added_portfolio_annotation", extra={"portfolio": name, "author": author})


def list_portfolio_annotations(name: str) -> List[Dict]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT author, note, created_at
            FROM portfolio_annotations
            WHERE portfolio_name = ?
            ORDER BY created_at DESC
            """,
            (name,),
        )
        rows = cursor.fetchall()

    return [
        {"author": author, "note": note, "created_at": created_at}
        for author, note, created_at in rows
    ]


def log_portfolio_event(name: Optional[str], action: str, actor: Optional[str] = None, payload: Optional[Dict] = None) -> None:
    now = datetime.utcnow().isoformat()
    payload_json = json.dumps(payload or {})
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO portfolio_events (portfolio_name, action, actor, payload, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (name, action, actor, payload_json, now),
        )
        conn.commit()
    LOGGER.info("portfolio_event_logged", extra={"portfolio": name, "action": action, "actor": actor})


def list_portfolio_events(
    name: Optional[str] = None,
    actor: Optional[str] = None,
    action: Optional[str] = None,
    limit: int = 100,
) -> List[Dict]:
    with get_connection() as conn:
        cursor = conn.cursor()
        query = """SELECT portfolio_name, action, actor, payload, created_at FROM portfolio_events WHERE 1=1"""
        params: List = []
        if name:
            query += " AND portfolio_name = ?"
            params.append(name)
        if actor:
            query += " AND (actor IS NOT NULL AND actor LIKE ?)"
            params.append(f"%{actor}%")
        if action:
            query += " AND action = ?"
            params.append(action)
        query += " ORDER BY datetime(created_at) DESC LIMIT ?"
        params.append(limit)
        cursor.execute(query, params)
        rows = cursor.fetchall()

    events = []
    for portfolio_name, action, actor, payload_json, created_at in rows:
        events.append(
            {
                "portfolio_name": portfolio_name,
                "action": action,
                "actor": actor,
                "payload": json.loads(payload_json or "{}"),
                "created_at": created_at,
            }
        )
    return events


__all__ = [
    "initialise_storage",
    "save_portfolio_config",
    "list_portfolio_configs",
    "get_portfolio_config",
    "update_portfolio_alerts",
    "delete_portfolio_config",
    "add_portfolio_annotation",
    "list_portfolio_annotations",
    "log_portfolio_event",
    "list_portfolio_events",
]
