"""Train the 6-class waste classifier (MobileNetV2 transfer learning), as described in docs/.

Expected data layout (TrashNet):  data/trashnet/<cardboard|glass|metal|paper|plastic|trash>/*.jpg
Run:  pip install -r requirements-train.txt && python training/train.py
Outputs: model/waste_classifier.keras  (then run training/export_onnx.py)

NOTE: written to mirror the project deck (stratified 70/15/15 split, class weights, two-stage
fine-tuning of the top 60 layers, test set used once). Run it yourself (Colab GPU recommended).
"""
import glob
import json
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

DATA, OUT, SIZE, BATCH, SEED = "data/trashnet", "model/waste_classifier.keras", 224, 32, 42
LABELS = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

paths, ys = [], []
for i, name in enumerate(LABELS):
    files = glob.glob(os.path.join(DATA, name, "*"))
    paths += files
    ys += [i] * len(files)
paths, ys = np.array(paths), np.array(ys)
assert len(paths), f"No images found in {DATA}"

# stratified 70 / 15 / 15
p_tr, p_tmp, y_tr, y_tmp = train_test_split(paths, ys, test_size=0.30, stratify=ys, random_state=SEED)
p_va, p_te, y_va, y_te = train_test_split(p_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
print(len(p_tr), len(p_va), len(p_te))

cw = compute_class_weight("balanced", classes=np.arange(len(LABELS)), y=y_tr)
class_weight = {i: float(w) for i, w in enumerate(cw)}


def load(path, y):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(img, (SIZE, SIZE))
    img.set_shape((SIZE, SIZE, 3))
    return img, y


def ds(p, y, train=False):
    d = tf.data.Dataset.from_tensor_slices((p, y))
    if train:
        d = d.shuffle(len(p), seed=SEED)
    return d.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH).prefetch(tf.data.AUTOTUNE)


train_ds, val_ds, test_ds = ds(p_tr, y_tr, True), ds(p_va, y_va), ds(p_te, y_te)

augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.RandomContrast(0.2),
])  # active only in training

base = tf.keras.applications.MobileNetV2(input_shape=(SIZE, SIZE, 3), include_top=False, weights="imagenet")
base.trainable = False

inp = tf.keras.Input((SIZE, SIZE, 3), name="input")                     # raw 0-255 pixels
x = augment(inp)
x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1, name="preprocess")(x)  # == MobileNetV2 preprocessing, embedded
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
out = tf.keras.layers.Dense(len(LABELS), activation="softmax")(x)
model = tf.keras.Model(inp, out)

cbs = [tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
       tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.3)]

# Stage 1: train the new head
model.compile(tf.keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", ["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=15, class_weight=class_weight, callbacks=cbs)

# Stage 2: fine-tune the top 60 layers at a low learning rate
base.trainable = True
for layer in base.layers[:-60]:
    layer.trainable = False
model.compile(tf.keras.optimizers.Adam(1e-5), "sparse_categorical_crossentropy", ["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=15, class_weight=class_weight, callbacks=cbs)

# Final evaluation: test set used ONCE
pred = np.argmax(model.predict(test_ds), axis=1)
print(classification_report(y_te, pred, target_names=LABELS, digits=3))
print(confusion_matrix(y_te, pred))

os.makedirs("model", exist_ok=True)
model.save(OUT)
print("Saved", OUT)
