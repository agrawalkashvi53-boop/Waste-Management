"""Image classifier wrapper (ONNX Runtime) with a clearly-labelled demo fallback."""
import hashlib
import json
import os

import numpy as np


class Classifier:
    def __init__(self, model_path: str, labels_path: str):
        with open(labels_path, encoding="utf-8") as f:
            cfg = json.load(f)
        self.labels = cfg["labels"]
        self.recyclable = cfg.get("recyclable", {})
        self.size = int(cfg.get("input_size", 224))
        self.preprocess = cfg.get("preprocess", "embedded")
        self.session = None
        self.input_name = None
        if os.path.exists(model_path):
            import onnxruntime as ort

            self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.version = cfg.get("version", "v1")
        else:
            self.version = "demo"

    @property
    def demo(self) -> bool:
        return self.session is None

    def _prepare(self, img) -> np.ndarray:
        img = img.resize((self.size, self.size))
        x = np.asarray(img, dtype=np.float32)
        if self.preprocess == "mobilenet_v2":
            x = x / 127.5 - 1.0
        elif self.preprocess == "rescale":
            x = x / 255.0
        # "embedded": preprocessing lives inside the model, feed raw 0-255 pixels
        return x[None, ...]

    def predict(self, img, raw_bytes: bytes = b"") -> np.ndarray:
        """Return a probability vector (len == number of labels)."""
        if self.demo:
            # NOT a real prediction: deterministic pseudo-output so the pipeline can be tested.
            seed = int.from_bytes(hashlib.sha256(raw_bytes).digest()[:4], "big")
            rng = np.random.RandomState(seed)
            p = rng.dirichlet(np.ones(len(self.labels)) * 0.6)
            return p.astype(np.float32)
        out = self.session.run(None, {self.input_name: self._prepare(img)})[0][0]
        out = out.astype(np.float64)
        if out.min() < 0 or abs(out.sum() - 1.0) > 1e-3:  # logits -> softmax
            e = np.exp(out - out.max())
            out = e / e.sum()
        return out
