"""Convert the trained Keras model to ONNX so the web app needs only onnxruntime (small, fast on Render).

Run:  python training/export_onnx.py [model/waste_classifier.keras]
"""
import subprocess
import sys

import numpy as np
import tensorflow as tf

src = sys.argv[1] if len(sys.argv) > 1 else "model/waste_classifier.keras"
dst = "model/waste_classifier.onnx"
model = tf.keras.models.load_model(src)
model.export("model/_saved_model")  # SavedModel (inference graph; augmentation layers are inactive)
subprocess.check_call([sys.executable, "-m", "tf2onnx.convert", "--saved-model", "model/_saved_model",
                       "--opset", "13", "--output", dst])

# Parity check: Keras vs ONNX on random pixels
import onnxruntime as ort  # noqa: E402

sess = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
x = np.random.randint(0, 256, (4, 224, 224, 3)).astype("float32")
a = model.predict(x, verbose=0)
b = sess.run(None, {sess.get_inputs()[0].name: x})[0]
print("max abs diff Keras vs ONNX:", float(np.abs(a - b).max()))
assert np.abs(a - b).max() < 1e-3, "Conversion mismatch, check preprocessing"
print("Saved", dst, "- commit it to Git (about 10 MB).")
