"""
DeforestAI - Model Retraining Script (No-pickle version)
Fixes collapsed MobileNetV2 model with 2-phase fine-tuning.

USAGE:
    python retrain.py --data "b:\DEFORESTATION\DEFORESTATION DETECTION"

Requirements: tensorflow, numpy  (no sklearn needed)
Dataset layout:
    <data_dir>/
        deforestation/       <- class 0
        no_deforestation/    <- class 1
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE    = (128, 128)
BATCH_SIZE  = 32
EPOCHS_1    = 10      # Phase 1: frozen base
EPOCHS_2    = 20      # Phase 2: unfrozen top layers
UNFREEZE    = 50      # How many top MobileNetV2 layers to unfreeze
OUTPUT_PATH = "deforestation.h5"

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CALLBACK — avoids deepcopy/pickle entirely
# ─────────────────────────────────────────────────────────────────────────────
class BestModelSaver(tf.keras.callbacks.Callback):
    """
    Saves best model weights using get_weights/set_weights (numpy only).
    Supports: save_path, patience for early stopping, LR reduction.
    No deepcopy, no pickle — works with frozen/unfrozen submodels.
    """
    def __init__(self, save_path, patience=5, lr_patience=3, lr_factor=0.5, min_lr=1e-7):
        super().__init__()
        self.save_path      = save_path
        self.patience       = patience
        self.lr_patience    = lr_patience
        self.lr_factor      = lr_factor
        self.min_lr         = min_lr
        self.best_val_acc   = -1.0
        self.best_weights   = None
        self.wait           = 0
        self.lr_wait        = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_acc = logs.get("val_accuracy", 0.0)
        val_loss = logs.get("val_loss", 999.0)
        lr = float(self.model.optimizer.learning_rate)

        if val_acc > self.best_val_acc:
            print(f"\n  [SAVE] val_accuracy improved: {self.best_val_acc:.4f} -> {val_acc:.4f}  Saving {self.save_path}")
            self.best_val_acc = val_acc
            self.best_weights = [w.copy() for w in self.model.get_weights()]
            self.model.save(self.save_path)
            self.wait    = 0
            self.lr_wait = 0
        else:
            self.wait    += 1
            self.lr_wait += 1
            print(f"\n  [--]  val_accuracy did not improve ({val_acc:.4f} <= {self.best_val_acc:.4f}). "
                  f"Patience {self.wait}/{self.patience}")

            # LR reduction
            if self.lr_wait >= self.lr_patience:
                new_lr = max(lr * self.lr_factor, self.min_lr)
                self.model.optimizer.learning_rate = new_lr
                print(f"  [LR]  Reducing LR: {lr:.2e} -> {new_lr:.2e}")
                self.lr_wait = 0

            # Early stopping
            if self.wait >= self.patience:
                print(f"\n  [STOP] Early stopping triggered. Best val_accuracy: {self.best_val_acc:.4f}")
                self.model.stop_training = True

    def restore_best(self):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            print(f"  [OK]  Restored best weights (val_accuracy={self.best_val_acc:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def compute_class_weights(labels):
    """Balanced class weights via pure numpy."""
    labels = np.array(labels)
    classes = np.unique(labels)
    n = len(labels)
    k = len(classes)
    return {int(c): n / (k * np.sum(labels == c)) for c in classes}


# ─────────────────────────────────────────────────────────────────────────────
# ARGS
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="DEFORESTATION DETECTION")
args   = parser.parse_args()
DATA_PATH = args.data

if not os.path.isdir(DATA_PATH):
    print(f"[ERR] Dataset not found: '{DATA_PATH}'")
    exit(1)

classes = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
print(f"\n[OK]  Dataset : {DATA_PATH}")
for c in classes:
    n = len(os.listdir(os.path.join(DATA_PATH, c)))
    print(f"      {c}: {n} images")

# ─────────────────────────────────────────────────────────────────────────────
# DATA PIPELINES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[>>]  Loading data...")
AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_PATH, validation_split=0.2, subset="training",
    seed=42, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="int",
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_PATH, validation_split=0.2, subset="validation",
    seed=42, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="int",
)

class_names = train_ds.class_names
print(f"      Classes: {list(enumerate(class_names))}")

# Class weights
labels_list = [y for _, batch_y in train_ds for y in batch_y.numpy()]
class_weights = compute_class_weights(labels_list)
print(f"      Class weights: {class_weights}")

train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)

# ─────────────────────────────────────────────────────────────────────────────
# DATA AUGMENTATION (as a separate Sequential — NOT embedded in functional graph)
# ─────────────────────────────────────────────────────────────────────────────
augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
], name="augmentation")

# Apply augmentation to training data only via dataset .map()
train_ds_aug = train_ds.map(
    lambda x, y: (augment(x, training=True), y),
    num_parallel_calls=AUTOTUNE
)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE (clean functional model, no augmentation layer inside)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[>>]  Building model...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(128, 128, 3), name="image_input")
x = layers.Rescaling(1.0 / 255, name="rescale")(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D(name="gap")(x)
x = layers.Dense(128, activation="relu", name="dense_head")(x)
x = layers.BatchNormalization(name="bn")(x)
x = layers.Dropout(0.4, name="dropout")(x)
outputs = layers.Dense(2, activation="softmax", name="predictions")(x)

model = tf.keras.Model(inputs, outputs, name="DeforestAI")

trainable = sum(np.prod(v.shape) for v in model.trainable_variables)
total     = model.count_params()
print(f"      Total params    : {total:,}")
print(f"      Trainable params: {trainable:,}")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Train head only
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  PHASE 1: Training head ({EPOCHS_1} epochs max, base frozen)")
print(f"{'='*60}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

cb1 = BestModelSaver(save_path="best_phase1.keras", patience=4, lr_patience=2, lr_factor=0.5)

model.fit(
    train_ds_aug,
    validation_data=val_ds,
    epochs=EPOCHS_1,
    callbacks=[cb1],
    class_weight=class_weights,
    verbose=1,
)
cb1.restore_best()
print(f"\n  Phase 1 complete. Best val_accuracy: {cb1.best_val_acc:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Fine-tune top MobileNetV2 layers
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  PHASE 2: Fine-tuning top {UNFREEZE} MobileNetV2 layers ({EPOCHS_2} epochs max)")
print(f"{'='*60}")

base_model.trainable = True
for layer in base_model.layers[:-UNFREEZE]:
    layer.trainable = False

trainable2 = sum(np.prod(v.shape) for v in model.trainable_variables)
print(f"  Trainable params now: {trainable2:,}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

cb2 = BestModelSaver(save_path=OUTPUT_PATH, patience=6, lr_patience=3, lr_factor=0.3)

model.fit(
    train_ds_aug,
    validation_data=val_ds,
    epochs=EPOCHS_2,
    callbacks=[cb2],
    class_weight=class_weights,
    verbose=1,
)
cb2.restore_best()

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SAVE & SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
model.save(OUTPUT_PATH)

print(f"\n{'='*60}")
print(f"  TRAINING COMPLETE")
print(f"{'='*60}")
print(f"  Best Val Accuracy : {cb2.best_val_acc * 100:.2f}%")
print(f"  Model saved to    : {OUTPUT_PATH}")
print(f"  Class mapping     : {dict(enumerate(class_names))}")
print(f"{'='*60}")
print(f"\n  Restart Streamlit to use the new model:")
print(f"    streamlit run app.py")

# Cleanup
if os.path.exists("best_phase1.keras"):
    os.remove("best_phase1.keras")
