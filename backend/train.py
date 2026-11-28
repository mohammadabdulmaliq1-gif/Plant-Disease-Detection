"""
Plant Disease Detection - Genuine Deep Learning Training Pipeline
Architecture: EfficientNetV2B0 with Transfer Learning & Fine-Tuning
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image

# Ensure local .keras directory with pre-downloaded ImageNet weights is recognized
BACKEND_DIR = Path(__file__).resolve().parent
LOCAL_KERAS = BACKEND_DIR / ".keras"
if LOCAL_KERAS.exists():
    os.environ["KERAS_HOME"] = str(LOCAL_KERAS)

import tensorflow as tf
import keras
from keras import layers, models, callbacks
from keras.applications import EfficientNetV2B0
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("TrainPipeline")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42


def scan_and_validate_dataset(dataset_dir: Path):
    """
    Discovers all dataset class folders, validates image integrity,
    and returns verified image paths, labels, and deterministic class mapping.
    """
    logger.info(f"Scanning dataset in: {dataset_dir}")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # Discover directories
    class_folders = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    if not class_folders:
        raise ValueError(f"No class folders found in {dataset_dir}")

    class_names = [d.name for d in class_folders]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_class = {idx: name for idx, name in enumerate(class_names)}

    logger.info(f"Discovered {len(class_names)} classes in deterministic order:")
    for idx, name in enumerate(class_names):
        logger.info(f"  [{idx:02d}] {name}")

    valid_filepaths = []
    valid_labels = []
    corrupted_count = 0

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    for class_folder in class_folders:
        idx = class_to_idx[class_folder.name]
        files = [f for f in class_folder.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
        class_valid = 0

        for f in files:
            try:
                # Fast validation
                with Image.open(f) as img:
                    img.verify()
                valid_filepaths.append(str(f))
                valid_labels.append(idx)
                class_valid += 1
            except Exception as e:
                logger.warning(f"Corrupted image skipped: {f} ({e})")
                corrupted_count += 1

        logger.info(f"  Class '{class_folder.name}': {class_valid} valid images")

    logger.info(f"Total valid images: {len(valid_filepaths)}, Corrupted images removed: {corrupted_count}")
    if len(valid_filepaths) < len(class_names) * 3:
        raise ValueError("Insufficient images per class for train/val/test split.")

    return valid_filepaths, valid_labels, class_names, class_to_idx


def build_data_generators(filepaths, labels, num_classes, test_size=0.15, val_size=0.15):
    """
    Creates stratified Train / Val / Test splits and returns tf.data.Dataset pipelines.
    Augmentation is strictly applied ONLY to the training split to prevent data leakage.
    """
    # First split off test set
    train_val_files, test_files, train_val_labels, test_labels = train_test_split(
        filepaths, labels,
        test_size=test_size,
        stratify=labels,
        random_state=SEED
    )

    # Then split train and val
    val_relative_size = val_size / (1.0 - test_size)
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_val_files, train_val_labels,
        test_size=val_relative_size,
        stratify=train_val_labels,
        random_state=SEED
    )

    logger.info(f"Stratified Dataset Split:")
    logger.info(f"  Training samples:   {len(train_files)} ({len(train_files)/len(filepaths)*100:.1f}%)")
    logger.info(f"  Validation samples: {len(val_files)} ({len(val_files)/len(filepaths)*100:.1f}%)")
    logger.info(f"  Test samples:       {len(test_files)} ({len(test_files)/len(filepaths)*100:.1f}%)")

    # Data augmentation pipeline for training
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.15),
    ], name="training_augmentation")

    def load_and_preprocess(filepath, label):
        img_bytes = tf.io.read_file(filepath)
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32)
        one_hot = tf.one_hot(label, depth=num_classes)
        return img, one_hot

    # Prepare tf.data pipelines
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    train_ds = train_ds.shuffle(buffer_size=min(len(train_files), 1000), seed=SEED)
    train_ds = train_ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.prefetch(AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
    val_ds = val_ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((test_files, test_labels))
    test_ds = test_ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, test_files, test_labels


def build_efficientnetv2_model(num_classes: int, dropout_rate: float = 0.3):
    """
    Builds the model architecture:
    1. Input layer: 224 x 224 x 3
    2. EfficientNetV2B0 pretrained backbone (ImageNet)
    3. Global Average Pooling
    4. Batch Normalization
    5. Dropout
    6. Dense Classification layers with Regularization
    7. Softmax output
    """
    inputs = layers.Input(shape=(*IMG_SIZE, 3), name="input_image")

    # EfficientNetV2B0 has built-in rescaling [0, 255] -> [-1, 1]
    base_model = EfficientNetV2B0(
        weights="imagenet",
        include_top=False,
        input_tensor=inputs,
        pooling="avg"
    )

    # Freeze base model for Phase 1
    base_model.trainable = False

    x = base_model.output
    x = layers.BatchNormalization(name="head_bn1")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout1")(x)

    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-4),
        name="head_dense1"
    )(x)
    x = layers.BatchNormalization(name="head_bn2")(x)
    x = layers.Dropout(dropout_rate / 2, name="head_dropout2")(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions"
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="PlantDiseaseEfficientNetV2")
    return model, base_model


def train_model(
    dataset_dir: str,
    epochs_phase1: int = 15,
    epochs_phase2: int = 10,
    output_dir: str = "models",
    reports_dir: str = "reports"
):
    """
    Executes two-stage transfer learning and saves model and full metrics.
    """
    dataset_path = Path(dataset_dir)
    out_path = Path(output_dir)
    rep_path = Path(reports_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rep_path.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Scan & Validation
    filepaths, labels, class_names, class_to_idx = scan_and_validate_dataset(dataset_path)
    num_classes = len(class_names)

    # 2. Split and Dataset pipeline
    train_ds, val_ds, test_ds, test_files, test_labels = build_data_generators(
        filepaths, labels, num_classes
    )

    # 3. Model construction
    logger.info("Building EfficientNetV2B0 Transfer Learning Architecture...")
    model, base_model = build_efficientnetv2_model(num_classes)
    model.summary(print_fn=logger.info)

    best_model_path = out_path / "best_model.keras"

    # Define callbacks for Phase 1
    phase1_callbacks = [
        callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            mode="max",
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.CSVLogger(
            str(rep_path / "training_history_phase1.csv"),
            append=False
        )
    ]

    # --- STAGE 1: Train Classification Head (Frozen Backbone) ---
    logger.info("=" * 70)
    logger.info("STAGE 1: TRAINING CLASSIFICATION HEAD (BACKBONE FROZEN)")
    logger.info(f"Optimizer: Adam(lr=1e-3), Loss: CategoricalCrossentropy")
    logger.info("=" * 70)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=[
            "accuracy",
            keras.metrics.TopKCategoricalAccuracy(k=min(3, num_classes), name="top3_accuracy")
        ]
    )

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_phase1,
        callbacks=phase1_callbacks,
        verbose=1
    )

    # --- STAGE 2: Fine-Tuning Backbone ---
    logger.info("=" * 70)
    logger.info("STAGE 2: FINE-TUNING PRETRAINED BACKBONE")
    logger.info("=" * 70)

    # Unfreeze top layers of base model, keeping BatchNormalization frozen
    base_model.trainable = True
    fine_tune_from = max(0, len(base_model.layers) - 30)
    logger.info(f"Unfreezing upper backbone layers from layer {fine_tune_from} to {len(base_model.layers)}")

    for layer in base_model.layers[:fine_tune_from]:
        layer.trainable = False
    for layer in base_model.layers[fine_tune_from:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False  # Keep BN frozen during fine-tuning

    # Re-compile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=[
            "accuracy",
            keras.metrics.TopKCategoricalAccuracy(k=min(3, num_classes), name="top3_accuracy")
        ]
    )

    phase2_callbacks = [
        callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            mode="max",
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.CSVLogger(
            str(rep_path / "training_history_phase2.csv"),
            append=False
        )
    ]

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_phase2,
        callbacks=phase2_callbacks,
        verbose=1
    )

    # 4. Save Final Class Mappings
    logger.info("Saving class mappings and configuration...")
    class_names_path = out_path / "class_names.json"
    with open(class_names_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    class_indices_path = out_path / "class_indices.json"
    with open(class_indices_path, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)

    # Also update backend/labels.json
    backend_labels_path = BACKEND_DIR / "labels.json"
    with open(backend_labels_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    logger.info(f"Saved class names to: {class_names_path} & {backend_labels_path}")

    # 5. Combine and save training history
    combined_history = {}
    for k in history1.history.keys():
        combined_history[k] = history1.history[k] + history2.history.get(k, [])

    import csv
    history_csv = rep_path / "training_history.csv"
    with open(history_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        headers = ["epoch"] + list(combined_history.keys())
        writer.writerow(headers)
        total_epochs = len(combined_history["loss"])
        for ep in range(total_epochs):
            row = [ep + 1] + [combined_history[k][ep] for k in combined_history.keys()]
            writer.writerow(row)
    logger.info(f"Saved full training history to: {history_csv}")

    # Plot Accuracy and Loss curves
    plot_training_curves(combined_history, len(history1.history["loss"]), rep_path)

    # 6. Comprehensive Test Set Evaluation
    logger.info("=" * 70)
    logger.info("EVALUATING BEST SAVED MODEL ON INDEPENDENT TEST SET")
    logger.info("=" * 70)
    best_model = keras.models.load_model(str(best_model_path))

    y_true = []
    y_pred_probs = []
    for imgs, batch_labels in test_ds:
        preds = best_model.predict(imgs, verbose=0)
        y_pred_probs.append(preds)
        y_true.append(np.argmax(batch_labels.numpy(), axis=1))

    y_true = np.concatenate(y_true)
    y_pred_probs = np.concatenate(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Classification report
    report_text = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4
    )
    logger.info("\n" + report_text)

    report_file = rep_path / "classification_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info(f"Saved classification report to: {report_file}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, rep_path / "confusion_matrix.png")

    logger.info("🎉 Complete Genuine Training & Evaluation Process Finished Successfully!")
    logger.info(f"Best model saved at: {best_model_path}")
    return str(best_model_path)


def plot_training_curves(history, phase1_len, reports_dir):
    """Plot and save accuracy and loss graphs."""
    epochs = range(1, len(history["loss"]) + 1)

    # Accuracy Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history.get("accuracy", []), "b-o", label="Training Accuracy")
    plt.plot(epochs, history.get("val_accuracy", []), "r-s", label="Validation Accuracy")
    if phase1_len < len(epochs):
        plt.axvline(x=phase1_len, color="g", linestyle="--", label="Phase 2 Fine-Tuning Start")
    plt.title("EfficientNetV2B0 - Training vs Validation Accuracy", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(reports_dir / "accuracy.png", dpi=300)
    plt.close()

    # Loss Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history.get("loss", []), "b-o", label="Training Loss")
    plt.plot(epochs, history.get("val_loss", []), "r-s", label="Validation Loss")
    if phase1_len < len(epochs):
        plt.axvline(x=phase1_len, color="g", linestyle="--", label="Phase 2 Fine-Tuning Start")
    plt.title("EfficientNetV2B0 - Training vs Validation Loss", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Categorical Crossentropy Loss", fontsize=12)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(reports_dir / "loss.png", dpi=300)
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix heatmap."""
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Test Dataset", fontsize=14, fontweight="bold")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90, fontsize=8)
    plt.yticks(tick_marks, class_names, fontsize=8)

    # Normalize values for text display
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if val > 0:
                plt.text(
                    j, i, format(val, "d"),
                    horizontalalignment="center",
                    color="white" if val > thresh else "black",
                    fontsize=7
                )

    plt.tight_layout()
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNetV2B0 Plant Disease Detection Model")
    parser.add_argument(
        "--dataset-dir",
        default=str(BACKEND_DIR / "data" / "plantvillage"),
        help="Path to dataset directory containing class subfolders"
    )
    parser.add_argument("--epochs-phase1", type=int, default=12, help="Epochs for training head")
    parser.add_argument("--epochs-phase2", type=int, default=8, help="Epochs for fine-tuning")
    parser.add_argument("--output-dir", default=str(BACKEND_DIR / "models"), help="Output directory for model")
    parser.add_argument("--reports-dir", default=str(BACKEND_DIR / "reports"), help="Output directory for reports")
    args = parser.parse_args()

    train_model(
        dataset_dir=args.dataset_dir,
        epochs_phase1=args.epochs_phase1,
        epochs_phase2=args.epochs_phase2,
        output_dir=args.output_dir,
        reports_dir=args.reports_dir
    )
