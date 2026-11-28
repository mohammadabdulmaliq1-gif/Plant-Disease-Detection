# -*- coding: utf-8 -*-
"""
Plant Disease Detection - Training Script
Trains EfficientNetB0 on the PlantVillage dataset via transfer learning.

Usage:
    C:\\Users\\moham\\anaconda3\\python.exe scripts/train_model.py
"""
from __future__ import annotations

import os
import sys
import json
import numpy as np
import pathlib

# Fix stdout encoding for Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_DIR = r"C:\Users\moham\Downloads\archive (5)\PlantVillage\PlantVillage"

# Resolve paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
MODEL_SAVE_DIR = os.path.join(BACKEND_DIR, "models")
# Save in Keras native format (.keras) - does not require h5py
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "best_model.keras")
LABELS_SAVE_PATH = os.path.join(BACKEND_DIR, "labels.json")
CLASS_INDICES_PATH = os.path.join(MODEL_SAVE_DIR, "class_indices.json")

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS_PHASE1 = 15      # Train classification head (base frozen)
EPOCHS_PHASE2 = 10      # Fine-tune top layers
VALIDATION_SPLIT = 0.2
SEED = 42


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    print("=" * 60)
    print("  Plant Disease Detection - Training Pipeline")
    print("=" * 60)
    print(f"Dataset  : {DATASET_DIR}")
    print(f"Model out: {MODEL_SAVE_PATH}")
    print(f"Img size : {IMG_SIZE}x{IMG_SIZE}, Batch: {BATCH_SIZE}")

    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"Dataset not found at: {DATASET_DIR}")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU      : {gpus[0].name}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("GPU      : None - using CPU (this will take ~1-3 hours)")

    class_dirs = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])
    total = 0
    print(f"\n{len(class_dirs)} classes found:")
    for cls in class_dirs:
        cnt = len(list(pathlib.Path(os.path.join(DATASET_DIR, cls)).glob("*.*")))
        total += cnt
        print(f"  {cls}: {cnt}")
    print(f"\nTotal images: {total}")
    print("=" * 60)
    return class_dirs


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_datasets(class_dirs):
    print("\nLoading datasets...")

    common_kwargs = dict(
        directory=DATASET_DIR,
        validation_split=VALIDATION_SPLIT,
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='categorical',
    )

    train_ds = keras.utils.image_dataset_from_directory(
        subset="training", shuffle=True, **common_kwargs
    )
    val_ds = keras.utils.image_dataset_from_directory(
        subset="validation", shuffle=False, **common_kwargs
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    # Save class order to labels.json (must match model output order)
    with open(LABELS_SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(class_names, f, indent=2)
    print(f"Saved {num_classes} labels -> {LABELS_SAVE_PATH}")

    # Save class index mapping for debugging
    class_indices = {name: idx for idx, name in enumerate(class_names)}
    with open(CLASS_INDICES_PATH, 'w', encoding='utf-8') as f:
        json.dump(class_indices, f, indent=2)
    print(f"Saved class indices -> {CLASS_INDICES_PATH}")

    # Data augmentation
    augment = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.10),
        layers.RandomBrightness(0.10),
        layers.RandomContrast(0.10),
    ], name="augmentation")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = (train_ds
                .map(lambda x, y: (augment(x, training=True), y),
                     num_parallel_calls=AUTOTUNE)
                .cache()
                .prefetch(AUTOTUNE))
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, num_classes, class_names


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes):
    print(f"\nBuilding EfficientNetB0 for {num_classes} classes ...")

    # NOTE: EfficientNetB0 in Keras 3 (TF 2.17+) includes its own internal
    # preprocessing/rescaling. We pass raw 0-255 uint8 images directly.
    # Do NOT add an external Rescaling layer - it causes duplicate name errors.
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image_input")

    base = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base.trainable = False   # Phase 1: freeze base

    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(256, activation='relu', name="fc1")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.20)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="predictions")(x)

    model = keras.Model(inputs, outputs, name="plant_disease_efficientnetb0")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    total = model.count_params()
    trainable = sum(np.prod(v.shape) for v in model.trainable_variables)
    print(f"Params total={total:,}  trainable(phase1)={trainable:,}")
    return model, base


def _callbacks(phase):
    return [
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy',
                        save_best_only=True, verbose=1, mode='max'),
        EarlyStopping(monitor='val_accuracy', patience=5,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, min_lr=1e-7, verbose=1),
        CSVLogger(os.path.join(MODEL_SAVE_DIR, f"log_phase{phase}.csv")),
    ]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def phase1(model, train_ds, val_ds):
    print("\n" + "-" * 60)
    print("Phase 1: training classification head (base frozen)")
    print("-" * 60)
    h = model.fit(train_ds, epochs=EPOCHS_PHASE1,
                  validation_data=val_ds, callbacks=_callbacks(1), verbose=1)
    best = max(h.history['val_accuracy'])
    print(f"\nPhase 1 best val_accuracy: {best:.4f} ({best*100:.2f}%)")
    return h


def phase2(model, base, train_ds, val_ds):
    print("\n" + "-" * 60)
    print("Phase 2: fine-tuning top 30 layers of EfficientNetB0")
    print("-" * 60)

    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    trainable = sum(np.prod(v.shape) for v in model.trainable_variables)
    print(f"Trainable params (phase 2): {trainable:,}")

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )

    h = model.fit(train_ds, epochs=EPOCHS_PHASE2,
                  validation_data=val_ds, callbacks=_callbacks(2), verbose=1)
    best = max(h.history['val_accuracy'])
    print(f"\nPhase 2 best val_accuracy: {best:.4f} ({best*100:.2f}%)")
    return h


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, val_ds, class_names):
    print("\n" + "-" * 60)
    print("Final evaluation on validation set")
    print("-" * 60)
    results = model.evaluate(val_ds, verbose=1)
    print(f"\nVal loss    : {results[0]:.4f}")
    print(f"Val accuracy: {results[1]:.4f} ({results[1]*100:.2f}%)")

    # Collect predictions and ground truth for confusion matrix
    print("\nCollecting predictions for classification report...")
    all_true = []
    all_pred = []

    for imgs, lbls in val_ds:
        preds = model.predict(imgs, verbose=0)
        all_true.extend(np.argmax(lbls.numpy(), axis=1))
        all_pred.extend(np.argmax(preds, axis=1))

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    # Classification report
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        report = classification_report(all_true, all_pred,
                                       target_names=class_names, digits=4)
        print("\nClassification Report:")
        print(report)

        # Save report
        report_path = os.path.join(MODEL_SAVE_DIR, "classification_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Val accuracy: {results[1]*100:.2f}%\n\n")
            f.write(report)
        print(f"Saved report -> {report_path}")

        # Confusion matrix summary (top misclassifications)
        cm = confusion_matrix(all_true, all_pred)
        print("\nConfusion Matrix (top-5 misclassifications):")
        errors = []
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if i != j and cm[i, j] > 0:
                    errors.append((cm[i, j], class_names[i], class_names[j]))
        errors.sort(reverse=True)
        for count, true_cls, pred_cls in errors[:5]:
            print(f"  {count}x  {true_cls} -> {pred_cls}")
    except ImportError:
        print("sklearn not available - skipping classification report")

    # Sample predictions
    print("\nSample predictions:")
    for imgs, lbls in val_ds.take(1):
        preds = model.predict(imgs[:5], verbose=0)
        for i in range(min(5, len(imgs))):
            t = np.argmax(lbls[i])
            p = np.argmax(preds[i])
            conf = preds[i][p]
            ok = "OK" if t == p else "WRONG"
            tl = class_names[t] if t < len(class_names) else f"cls_{t}"
            pl = class_names[p] if p < len(class_names) else f"cls_{p}"
            print(f"  [{ok}] True: {tl:45s} | Pred: {pl:45s} ({conf*100:.1f}%)")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"TensorFlow: {tf.__version__}")
    class_dirs = setup()
    train_ds, val_ds, num_classes, class_names = load_datasets(class_dirs)
    model, base = build_model(num_classes)

    phase1(model, train_ds, val_ds)
    phase2(model, base, train_ds, val_ds)
    results = evaluate(model, val_ds, class_names)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Model  : {MODEL_SAVE_PATH}")
    print(f"  Labels : {LABELS_SAVE_PATH}")
    print(f"  Val acc: {results[1]*100:.2f}%")
    print("\nTo start the backend:")
    print("  cd backend")
    print("  .\\venv\\Scripts\\python.exe -m uvicorn app.main:app --reload")
    print("  OR: start_server.bat")
    print("=" * 60)


if __name__ == "__main__":
    main()
