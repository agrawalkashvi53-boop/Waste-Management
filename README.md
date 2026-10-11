# ♻ Waste Sorting QC: AI quality control for material-recovery facilities

**UN SDG 12: Responsible Consumption and Production** (Target 12.5: substantially reduce waste through prevention, reduction, recycling and reuse)

## The industrial problem
In a Material Recovery Facility (MRF), mixed waste moves along a conveyor and is sorted into material streams
(cardboard, glass, metal, paper, plastic) before baling. Sorting errors **contaminate bales**, which lowers their resale
value and sends recyclable material to landfill. Plants usually have no measured view of *how often* their sorting
is wrong, *where*, or *how confident* an automated system really is.

## What this project does
| Step | Feature |
|---|---|
| 1. Classify | A MobileNetV2 model (transfer learning, 6 TrashNet classes) classifies belt-camera frames. |
| 2. Route | Confidence ≥ threshold (default 0.8) → **auto-sort lane**. Below it → **manual-review queue**. |
| 3. Audit | A random 5% of auto-sorted items are also queued for a human check, so accuracy is *measured*, not assumed. |
| 4. Log | Every decision is stored in SQL (`waste_classes`, `predictions`). |
| 5. Analyse | Dashboard: throughput, material mix, recyclable share (diversion), confidence distribution, operator-verified accuracy by confidence band, top confusions (e.g. glass→metal), per-line view. |
| 6. Export | `/api/export.csv` feeds Power BI; `sql/analysis_queries.sql` has the JOIN / GROUP BY / CASE WHEN analyses. |

This extends the original *Smart Waste Classifier with Analytics* (see `docs/`) from a single-image demo into a
human-in-the-loop plant tool. The 0.8 rule comes from the project's finding that ≥0.8 predictions were 90.7% correct
versus ~56% below.

## Architecture
```
 belt camera / upload ──► Flask API ──► ONNX Runtime (MobileNetV2) ──► route: auto | manual
                              │                                              │
                              ▼                                              ▼
                        SQLite (predictions) ◄──── operator corrections ◄── review queue
                              │
                  Dashboard (Chart.js) · CSV → Power BI
```

## Quick start (local)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest -q                                              # 10 tests
python scripts/seed_demo_data.py 1500                  # optional SYNTHETIC rows to preview the dashboard
flask --app wsgi run                                   # http://127.0.0.1:5000
```
Without a model file the app starts in **DEMO MODE** (orange banner; predictions are placeholders, not real).

## Get the real model
1. Download TrashNet (resized) into `data/trashnet/<class>/*.jpg`.
2. `pip install -r requirements-train.txt && python training/train.py` (Colab GPU is easiest).
3. `python training/export_onnx.py` → creates `model/waste_classifier.onnx` (~10 MB).
4. Commit that file. Already have a trained Keras model? Skip step 2, just run step 3 on it.
   If its preprocessing is *not* embedded, set `"preprocess"` in `model/labels.json` to `mobilenet_v2` or `rescale`.

> The training/export scripts follow the project deck but were not run in the authoring environment: run the
> parity check printed by `export_onnx.py`. Your reported numbers (85.3% test accuracy) come from your own run.

## Push to GitHub
```bash
git init && git add . && git commit -m "Waste Sorting QC"
git branch -M main
git remote add origin https://github.com/<you>/waste-sorting-qc.git
git push -u origin main
```
Replace `<Your Name>` in `LICENSE`.

## Deploy on Render
1. Render dashboard → **New → Blueprint** → select your repo (it reads `render.yaml`), or **New → Web Service** with
   build `pip install -r requirements.txt` and start `gunicorn wsgi:app --workers 1 --threads 4 --timeout 120`.
2. Set `APP_PASSWORD` (HTTP basic auth; username is ignored). **Do this**, otherwise anyone can write to your database.
3. Open the service URL. `/health` shows `demo_mode: false` once the ONNX file is deployed.

**Free-tier caveat:** Render's free disk is ephemeral and the service sleeps when idle, so the SQLite file resets on
redeploy/restart. For persistence add a Render persistent disk and set `DB_PATH=/var/data/waste.db`, or move to
Render Postgres (the schema in `sql/schema.sql` is portable).

## Configuration
| Variable | Default | Meaning |
|---|---|---|
| `CONF_THRESHOLD` | 0.8 | Below → manual review |
| `AUDIT_RATE` | 0.05 | Share of auto items spot-checked |
| `APP_PASSWORD` | empty | Enables basic auth |
| `DB_PATH` | `data/waste.db` | SQLite file |
| `MODEL_PATH` / `LABELS_PATH` | `model/…` | Model and class config |

## API
| Method | Path | Notes |
|---|---|---|
| POST | `/api/predict` | multipart `files` (≤20 images), optional `line_id` |
| GET | `/api/stats?hours=168&line=LINE-1` | dashboard data |
| GET / POST | `/api/review`, `/api/review/<id>` | queue; body `{"label": "glass"}` |
| GET | `/api/export.csv` | all predictions |
| GET | `/health` | status, model version, demo flag |

```bash
curl -u x:$APP_PASSWORD -F files=@frame.jpg -F line_id=LINE-2 https://<your-app>.onrender.com/api/predict
```

## Limitations (honest list)
- TrashNet has plain backgrounds; real belts are cluttered, so expect lower accuracy until you add real belt images (RealWaste, TACO, your own).
- No "not waste" class: unrelated objects can be classified confidently. Add one before any real deployment.
- Some errors are high-confidence (e.g. glass→metal), which is why audit sampling exists.
- Single-item images only; a real belt needs object detection/segmentation first.
- Accuracy shown on the dashboard is only as good as operator labels.

## Structure
```
app/ (Flask app, templates, static) · model/ · training/ · sql/ · scripts/ · tests/ · docs/ · render.yaml
```
