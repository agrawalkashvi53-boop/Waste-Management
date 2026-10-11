"""Fill the database with SYNTHETIC rows so the dashboard can be previewed.

These are NOT real plant measurements. Rows are tagged model_version='synthetic-demo'
and the dashboard shows a warning while they exist.   Usage: python scripts/seed_demo_data.py [N]
"""
import os
import random
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import create_app  # noqa: E402
from app.db import ISO, connect  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
app = create_app()
labels = app.classifier.labels
weights = [0.18, 0.12, 0.14, 0.22, 0.24, 0.10]  # plausible-looking mix, invented
confusable = {"glass": "metal", "plastic": "metal", "paper": "cardboard"}
thr = app.config["CONF_THRESHOLD"]
rng = random.Random(7)
conn = connect(app.config["DB_PATH"])
now = datetime.now(timezone.utc)

for _ in range(N):
    truth = rng.choices(labels, weights)[0]
    conf = min(0.999, max(0.25, rng.betavariate(9, 1.4)))
    wrong = rng.random() < (0.05 if conf >= thr else 0.42)
    pred = confusable.get(truth, rng.choice(labels)) if wrong else truth
    if pred == truth and wrong:
        pred = rng.choice([l for l in labels if l != truth])
    route = "auto" if conf >= thr else "manual"
    verified = route == "manual" and rng.random() < 0.8 or route == "auto" and rng.random() < 0.05
    ts = (now - timedelta(minutes=rng.randint(0, 7 * 24 * 60))).strftime(ISO)
    conn.execute(
        """INSERT INTO predictions (created_at, line_id, source_name, predicted_class, confidence, route,
                                    review_state, reviewed_class, reviewed_at, model_version)
           VALUES (?,?,?,?,?,?,?,?,?,'synthetic-demo')""",
        (ts, rng.choice(["LINE-1", "LINE-2", "LINE-3"]), "synthetic.jpg", pred, round(conf, 4), route,
         "done" if verified else "none", truth if verified else None, ts if verified else None))
conn.commit()
print(f"Inserted {N} synthetic rows into {app.config['DB_PATH']}")
