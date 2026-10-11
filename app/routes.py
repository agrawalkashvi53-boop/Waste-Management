import base64
import csv
import io
import random
import re

from flask import Blueprint, Response, current_app, jsonify, render_template, request
from PIL import Image, ImageOps

from .db import get_db, get_stats, now

bp = Blueprint("main", __name__)
Image.MAX_IMAGE_PIXELS = 50_000_000
LINE_RE = re.compile(r"[^A-Za-z0-9_-]")


# ---------- pages ----------
@bp.get("/")
def classify_page():
    return render_template("index.html", page="classify")


@bp.get("/dashboard")
def dashboard_page():
    return render_template("dashboard.html", page="dashboard")


@bp.get("/review")
def review_page():
    return render_template("review.html", page="review")


@bp.get("/health")
def health():
    clf = current_app.classifier
    return jsonify(status="ok", model=clf.version, demo_mode=clf.demo)


# ---------- helpers ----------
def load_image(raw: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")
    except Exception as exc:  # corrupt, unsupported, decompression bomb ...
        raise ValueError("File is not a valid image") from exc


def make_thumb(img: Image.Image) -> bytes:
    t = img.copy()
    t.thumbnail((192, 192))
    buf = io.BytesIO()
    t.save(buf, "JPEG", quality=70)
    return buf.getvalue()


def csv_safe(v):
    """Neutralise spreadsheet formula injection in exported text."""
    return "'" + v if isinstance(v, str) and v[:1] in ("=", "+", "-", "@") else v


# ---------- API ----------
@bp.post("/api/predict")
def predict():
    clf, cfg = current_app.classifier, current_app.config
    files = request.files.getlist("files")
    if not files:
        return jsonify(error="No files uploaded (use form field 'files')"), 400
    if len(files) > cfg["MAX_FILES"]:
        return jsonify(error=f"At most {cfg['MAX_FILES']} images per request"), 400
    line = LINE_RE.sub("", request.form.get("line_id", "LINE-1"))[:32] or "LINE-1"

    db, results = get_db(), []
    for f in files:
        name = (f.filename or "image").replace("\\", "/").split("/")[-1][:100]
        raw = f.read()
        try:
            img = load_image(raw)
        except ValueError as exc:
            results.append({"source_name": name, "error": str(exc)})
            continue

        probs = clf.predict(img, raw)
        order = sorted(range(len(probs)), key=lambda i: -probs[i])
        label, conf = clf.labels[order[0]], float(probs[order[0]])

        manual = conf < cfg["CONF_THRESHOLD"]
        audit = (not manual) and random.random() < cfg["AUDIT_RATE"]
        route = "manual" if manual else "auto"
        state = "pending" if (manual or audit) else "none"
        cur = db.execute(
            """INSERT INTO predictions (created_at, line_id, source_name, predicted_class, confidence,
                                        route, review_state, model_version, thumb)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (now(), line, name, label, conf, route, state, clf.version,
             make_thumb(img) if state == "pending" else None),
        )
        results.append({
            "id": cur.lastrowid, "source_name": name, "line_id": line,
            "predicted_class": label, "confidence": round(conf, 4),
            "recyclable": bool(clf.recyclable.get(label, True)),
            "route": route, "audit_sample": audit,
            "top3": [{"label": clf.labels[i], "prob": round(float(probs[i]), 4)} for i in order[:3]],
            "model_version": clf.version, "demo": clf.demo,
        })
    db.commit()
    return jsonify(results=results, threshold=cfg["CONF_THRESHOLD"])


@bp.get("/api/stats")
def stats():
    hours = max(1, min(request.args.get("hours", 168, type=int), 24 * 365))
    line = LINE_RE.sub("", request.args.get("line", "")) or None
    return jsonify(get_stats(get_db(), hours, line, current_app.config["CONF_THRESHOLD"]))


@bp.get("/api/review")
def review_queue():
    rows = get_db().execute(
        """SELECT id, created_at, line_id, source_name, predicted_class, confidence, route, thumb
           FROM predictions WHERE review_state = 'pending'
           ORDER BY (route = 'manual') DESC, created_at LIMIT 24""").fetchall()
    total = get_db().execute("SELECT COUNT(*) FROM predictions WHERE review_state='pending'").fetchone()[0]
    items = []
    for r in rows:
        d = dict(r)
        t = d.pop("thumb")
        d["thumb"] = "data:image/jpeg;base64," + base64.b64encode(t).decode() if t else None
        items.append(d)
    return jsonify(items=items, pending_total=total, labels=current_app.classifier.labels)


@bp.post("/api/review/<int:pred_id>")
def review_submit(pred_id: int):
    label = (request.get_json(silent=True) or {}).get("label")
    if label not in current_app.classifier.labels:
        return jsonify(error="Invalid label"), 400
    db = get_db()
    cur = db.execute(
        """UPDATE predictions SET reviewed_class = ?, reviewed_at = ?, review_state = 'done', thumb = NULL
           WHERE id = ? AND review_state = 'pending'""", (label, now(), pred_id))
    db.commit()
    if cur.rowcount == 0:
        return jsonify(error="Item not found or already reviewed"), 404
    return jsonify(ok=True)


@bp.get("/api/export.csv")
def export_csv():
    rows = get_db().execute(
        """SELECT p.id, p.created_at, p.line_id, p.source_name, p.predicted_class, p.confidence,
                  p.route, p.review_state, p.reviewed_class, p.reviewed_at, p.model_version,
                  c.recyclable
           FROM predictions p JOIN waste_classes c ON c.class_name = COALESCE(p.reviewed_class, p.predicted_class)
           ORDER BY p.id""").fetchall()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(rows[0].keys() if rows else
               ["id", "created_at", "line_id", "source_name", "predicted_class", "confidence", "route",
                "review_state", "reviewed_class", "reviewed_at", "model_version", "recyclable"])
    for r in rows:
        w.writerow([csv_safe(v) for v in tuple(r)])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=predictions.csv"})
