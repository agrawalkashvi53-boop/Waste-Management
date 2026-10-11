import io
import json

import pytest
from PIL import Image

from app import create_app


def png(color=(120, 80, 40), size=(64, 48)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def make_client(tmp_path):
    def _make(**cfg):
        app = create_app({"DB_PATH": str(tmp_path / "t.db"), "MODEL_PATH": str(tmp_path / "none.onnx"), **cfg})
        return app.test_client()
    return _make


def upload(client, *imgs, line="LINE-1"):
    data = {"files": [(io.BytesIO(b), f"img{i}.png") for i, b in enumerate(imgs)], "line_id": line}
    return client.post("/api/predict", data=data, content_type="multipart/form-data")


def test_health_reports_demo(make_client):
    j = make_client().get("/health").get_json()
    assert j["status"] == "ok" and j["demo_mode"] is True


def test_predict_and_routing(make_client):
    c = make_client(CONF_THRESHOLD=1.1, AUDIT_RATE=0)   # everything below threshold -> manual
    r = upload(c, png()).get_json()["results"][0]
    assert r["route"] == "manual" and r["demo"] is True
    c2 = make_client(CONF_THRESHOLD=0.0, AUDIT_RATE=0)  # everything confident -> auto
    assert upload(c2, png()).get_json()["results"][0]["route"] == "auto"


def test_invalid_file_rejected_per_item(make_client):
    c = make_client()
    res = upload(c, b"not an image", png()).get_json()["results"]
    assert "error" in res[0] and "id" in res[1]


def test_no_files_and_too_many(make_client):
    c = make_client(MAX_FILES=1)
    assert c.post("/api/predict", data={}).status_code == 400
    assert upload(c, png(), png()).status_code == 400


def test_review_flow_and_stats(make_client):
    c = make_client(CONF_THRESHOLD=1.1, AUDIT_RATE=0)
    rid = upload(c, png()).get_json()["results"][0]["id"]
    q = c.get("/api/review").get_json()
    assert q["pending_total"] == 1 and q["items"][0]["thumb"].startswith("data:image/jpeg")
    assert c.post(f"/api/review/{rid}", json={"label": "bogus"}).status_code == 400
    assert c.post(f"/api/review/{rid}", json={"label": "glass"}).status_code == 200
    assert c.post(f"/api/review/{rid}", json={"label": "glass"}).status_code == 404  # already done
    s = c.get("/api/stats?hours=24").get_json()
    assert s["kpi"]["total"] == 1 and s["kpi"]["reviewed"] == 1 and s["kpi"]["pending_review"] == 0
    assert s["by_class"][0]["label"] == "glass"          # reviewed label wins
    assert s["demo_rows"] == 1


def test_stats_empty_db(make_client):
    s = make_client().get("/api/stats").get_json()
    assert s["kpi"]["total"] == 0 and s["kpi"]["reviewed_accuracy"] is None


def test_csv_export_neutralises_formulas(make_client):
    c = make_client()
    data = {"files": [(io.BytesIO(png()), "=HYPERLINK(1).png")]}
    c.post("/api/predict", data=data, content_type="multipart/form-data")
    body = c.get("/api/export.csv").get_data(as_text=True)
    assert "'=HYPERLINK" in body and ",=HYPERLINK" not in body


def test_password_protection(make_client):
    c = make_client(APP_PASSWORD="s3cret")
    assert c.get("/").status_code == 401
    assert c.get("/health").status_code == 200
    import base64
    ok = {"Authorization": "Basic " + base64.b64encode(b"u:s3cret").decode()}
    assert c.get("/", headers=ok).status_code == 200


def test_pages_render(make_client):
    c = make_client()
    for p in ("/", "/dashboard", "/review"):
        assert c.get(p).status_code == 200


def test_real_onnx_path(tmp_path, make_client):
    onnx = pytest.importorskip("onnx")
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper
    # tiny model: mean over H,W -> Gemm(3->6) -> Softmax, input NHWC 224x224x3
    w = numpy_helper.from_array(np.eye(3, 6, dtype=np.float32).T.copy().T * 0.05, "W")
    b = numpy_helper.from_array(np.zeros(6, dtype=np.float32), "B")
    g = helper.make_graph(
        [helper.make_node("ReduceMean", ["input"], ["m"], axes=[1, 2], keepdims=0),
         helper.make_node("Gemm", ["m", "W", "B"], ["z"]),
         helper.make_node("Softmax", ["z"], ["out"], axis=1)],
        "tiny", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 224, 224, 3])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [None, 6])], [w, b])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    path = tmp_path / "m.onnx"
    onnx.save(m, str(path))
    c = make_client(MODEL_PATH=str(path))
    j = c.get("/health").get_json()
    assert j["demo_mode"] is False
    r = upload(c, png((250, 0, 0))).get_json()["results"][0]
    assert r["demo"] is False and abs(sum(t["prob"] for t in r["top3"])) <= 1.0001
