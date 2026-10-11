import os
import sqlite3
from datetime import datetime, timedelta, timezone

from flask import current_app, g

ISO = "%Y-%m-%dT%H:%M:%SZ"
SCHEMA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sql", "schema.sql")


def now() -> str:
    return datetime.now(timezone.utc).strftime(ISO)


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = connect(current_app.config["DB_PATH"])
    return g.db


def close_db(_exc=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db(app, classifier):
    conn = connect(app.config["DB_PATH"])
    try:
        conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.DatabaseError:
        pass
    with open(SCHEMA, encoding="utf-8") as f:
        conn.executescript(f.read())
    for name in classifier.labels:
        conn.execute(
            "INSERT OR IGNORE INTO waste_classes (class_name, recyclable) VALUES (?, ?)",
            (name, 1 if classifier.recyclable.get(name, True) else 0),
        )
    conn.commit()
    conn.close()


def get_stats(db, hours: int, line: str | None, threshold: float) -> dict:
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime(ISO)
    where, params = "p.created_at >= ?", [since]
    if line:
        where += " AND p.line_id = ?"
        params.append(line)
    final = "COALESCE(p.reviewed_class, p.predicted_class)"
    frm = f"FROM predictions p JOIN waste_classes c ON c.class_name = {final} WHERE {where}"

    k = db.execute(
        f"""SELECT COUNT(*) AS total,
                   COALESCE(SUM(p.route = 'auto'), 0)            AS auto_sorted,
                   COALESCE(SUM(p.route = 'manual'), 0)          AS manual,
                   COALESCE(SUM(p.review_state = 'pending'), 0)  AS pending,
                   COALESCE(SUM(p.review_state = 'done'), 0)     AS reviewed,
                   COALESCE(SUM(p.review_state = 'done' AND p.reviewed_class = p.predicted_class), 0) AS reviewed_correct,
                   COALESCE(AVG(p.confidence), 0)                AS avg_conf,
                   COALESCE(SUM(c.recyclable), 0)                AS recyclable
            {frm}""",
        params,
    ).fetchone()
    total = k["total"]

    by_class = [dict(r) for r in db.execute(
        f"SELECT {final} AS label, COUNT(*) AS n {frm} GROUP BY label ORDER BY n DESC", params)]

    bucket = "%Y-%m-%d %H:00" if hours <= 48 else "%Y-%m-%d"
    timeline = [dict(r) for r in db.execute(
        f"""SELECT strftime('{bucket}', p.created_at) AS t, COUNT(*) AS n,
                   SUM(p.route = 'manual') AS manual {frm} GROUP BY t ORDER BY t""", params)]

    hist = [0] * 10
    for r in db.execute(
        f"SELECT MIN(CAST(p.confidence * 10 AS INTEGER), 9) AS b, COUNT(*) AS n {frm} GROUP BY b", params):
        hist[r["b"]] = r["n"]

    bands = {r["band"]: dict(r) for r in db.execute(
        f"""SELECT CASE WHEN p.confidence >= ? THEN 'high' ELSE 'low' END AS band,
                   COUNT(*) AS reviewed,
                   AVG(p.reviewed_class = p.predicted_class) AS accuracy
            {frm} AND p.review_state = 'done' GROUP BY band""", [threshold] + params)}

    confusions = [dict(r) for r in db.execute(
        f"""SELECT p.predicted_class AS predicted, p.reviewed_class AS actual, COUNT(*) AS n
            {frm} AND p.review_state = 'done' AND p.reviewed_class != p.predicted_class
            GROUP BY predicted, actual ORDER BY n DESC LIMIT 5""", params)]

    lines = [dict(r) for r in db.execute(
        f"""SELECT p.line_id AS line, COUNT(*) AS n,
                   AVG(p.route = 'auto') AS auto_rate, AVG(p.confidence) AS avg_conf
            {frm} GROUP BY p.line_id ORDER BY p.line_id""", params)]

    demo_rows = db.execute(
        "SELECT COUNT(*) FROM predictions WHERE model_version IN ('demo', 'synthetic-demo')").fetchone()[0]

    return {
        "hours": hours, "threshold": threshold, "demo_rows": demo_rows,
        "kpi": {
            "total": total,
            "auto_rate": k["auto_sorted"] / total if total else 0,
            "pending_review": k["pending"],
            "avg_confidence": k["avg_conf"],
            "recyclable_share": k["recyclable"] / total if total else 0,
            "reviewed": k["reviewed"],
            "reviewed_accuracy": k["reviewed_correct"] / k["reviewed"] if k["reviewed"] else None,
        },
        "by_class": by_class, "timeline": timeline, "confidence_hist": hist,
        "bands": bands, "confusions": confusions, "lines": lines,
    }
