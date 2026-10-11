import hmac
import os

from flask import Flask, Response, request

from .db import close_db, init_db
from .model import Classifier
from .routes import bp

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        DB_PATH=os.environ.get("DB_PATH", os.path.join(BASE, "data", "waste.db")),
        MODEL_PATH=os.environ.get("MODEL_PATH", os.path.join(BASE, "model", "waste_classifier.onnx")),
        LABELS_PATH=os.environ.get("LABELS_PATH", os.path.join(BASE, "model", "labels.json")),
        CONF_THRESHOLD=float(os.environ.get("CONF_THRESHOLD", "0.8")),  # below this -> manual review
        AUDIT_RATE=float(os.environ.get("AUDIT_RATE", "0.05")),         # share of auto items spot-checked
        APP_PASSWORD=os.environ.get("APP_PASSWORD", ""),                # optional HTTP basic auth
        MAX_CONTENT_LENGTH=32 * 1024 * 1024,
        MAX_FILES=20,
    )
    if test_config:
        app.config.update(test_config)

    os.makedirs(os.path.dirname(os.path.abspath(app.config["DB_PATH"])), exist_ok=True)
    app.classifier = Classifier(app.config["MODEL_PATH"], app.config["LABELS_PATH"])
    init_db(app, app.classifier)
    app.teardown_appcontext(close_db)
    app.register_blueprint(bp)

    @app.before_request
    def require_password():
        pwd = app.config["APP_PASSWORD"]
        if not pwd or request.path == "/health":
            return None
        auth = request.authorization
        if auth and hmac.compare_digest(auth.password or "", pwd):
            return None
        return Response("Authentication required", 401, {"WWW-Authenticate": 'Basic realm="Waste QC"'})

    @app.after_request
    def security_headers(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        return resp

    return app
