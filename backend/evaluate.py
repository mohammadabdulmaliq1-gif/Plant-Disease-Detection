"""
Plant Disease Detection - Independent Model Evaluation
Computes accuracy, precision, recall, F1-score, top-3 accuracy, per-class accuracy,
confusion matrix, and misclassification pattern analysis.
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parent
LOCAL_KERAS = BACKEND_DIR / ".keras"
if LOCAL_KERAS.exists():
    os.environ["KERAS_HOME"] = str(LOCAL_KERAS)

import tensorflow as tf
import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    top_k_accuracy_score
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Evaluate")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16


def evaluate_model(
    model_path: str,
    dataset_dir: str,
    class_names_path: str = None,
    output_dir: str = "reports"
):
    """Run comprehensive evaluation on a dataset directory."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load model
    logger.info(f"Loading trained model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at: {model_path}")

    model = keras.models.load_model(model_path)
    logger.info(f"Loaded model input shape: {model.input_shape}, output shape: {model.output_shape}")

    # 2. Load class names
    if class_names_path and os.path.exists(class_names_path):
        with open(class_names_path, "r", encoding="utf-8") as f:
            class_names = json.load(f)
    else:
        # Fallback to class_names.json in models
        fallback = BACKEND_DIR / "models" / "class_names.json"
        if fallback.exists():
            with open(fallback, "r", encoding="utf-8") as f:
                class_names = json.load(f)
        else:
            fallback_labels = BACKEND_DIR / "labels.json"
            with open(fallback_labels, "r", encoding="utf-8") as f:
                class_names = json.load(f)

    num_classes = len(class_names)
    logger.info(f"Evaluation classes ({num_classes}): {class_names}")

    # 3. Scan images from dataset_dir
    data_path = Path(dataset_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Evaluation dataset path does not exist: {dataset_dir}")

    filepaths = []
    y_true = []
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for class_name in class_names:
        class_folder = data_path / class_name
        if not class_folder.exists():
            continue
        idx = class_to_idx[class_name]
        for f in class_folder.iterdir():
            if f.is_file() and f.suffix.lower() in valid_exts:
                filepaths.append(str(f))
                y_true.append(idx)

    if not filepaths:
        raise ValueError(f"No valid images found in evaluation directory: {dataset_dir}")

    logger.info(f"Found {len(filepaths)} test images across {len(set(y_true))} classes.")

    # 4. Predict in batches
    y_true = np.array(y_true)
    y_pred_probs = []

    for i in range(0, len(filepaths), BATCH_SIZE):
        batch_files = filepaths[i:i + BATCH_SIZE]
        batch_imgs = []
        for fp in batch_files:
            with Image.open(fp) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img = img.resize(IMG_SIZE, Image.LANCZOS)
                batch_imgs.append(np.array(img, dtype=np.float32))

        batch_arr = np.array(batch_imgs)
        preds = model.predict(batch_arr, verbose=0)
        y_pred_probs.append(preds)

    y_pred_probs = np.concatenate(y_pred_probs, axis=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 5. Compute metrics
    acc = accuracy_score(y_true, y_pred)
    k_val = min(3, num_classes)
    top3_acc = top_k_accuracy_score(y_true, y_pred_probs, k=k_val, labels=list(range(num_classes)))

    # Per-class accuracy
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    per_class_acc = {}
    for idx, name in enumerate(class_names):
        total_class = np.sum(cm[idx, :])
        correct_class = cm[idx, idx]
        per_class_acc[name] = float(correct_class / total_class) if total_class > 0 else 0.0

    # Classification report
    clf_report = classification_report(
        y_true, y_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    # Identify frequently confused classes
    confused_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    "true_class": class_names[i],
                    "predicted_class": class_names[j],
                    "count": int(cm[i, j]),
                    "error_rate": float(cm[i, j] / np.sum(cm[i, :])) if np.sum(cm[i, :]) > 0 else 0.0
                })

    confused_pairs.sort(key=lambda x: x["count"], reverse=True)

    # Summary
    logger.info("=" * 70)
    logger.info("MODEL EVALUATION RESULTS")
    logger.info("=" * 70)
    logger.info(f"Overall Accuracy:       {acc:.4f} ({acc*100:.2f}%)")
    logger.info(f"Top-{k_val} Accuracy:         {top3_acc:.4f} ({top3_acc*100:.2f}%)")
    logger.info("\n" + clf_report)

    logger.info(f"Top Most Confused Class Pairs:")
    for pair in confused_pairs[:5]:
        logger.info(f"  {pair['true_class']} -> {pair['predicted_class']}: {pair['count']} misclassifications ({pair['error_rate']*100:.1f}%)")

    # 6. Save evaluation outputs
    eval_summary = {
        "overall_accuracy": float(acc),
        f"top_{k_val}_accuracy": float(top3_acc),
        "total_test_samples": len(filepaths),
        "per_class_accuracy": per_class_acc,
        "top_misclassifications": confused_pairs[:10],
        "evaluated_at": datetime.utcnow().isoformat()
    }

    with open(out_path / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    with open(out_path / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Overall Accuracy: {acc:.4f}\nTop-{k_val} Accuracy: {top3_acc:.4f}\n\n")
        f.write(clf_report)

    # Plot Confusion Matrix
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Evaluation Confusion Matrix (Accuracy: {acc*100:.2f}%)", fontsize=14, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90, fontsize=8)
    plt.yticks(tick_marks, class_names, fontsize=8)

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
    plt.ylabel("True Class", fontsize=12)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.savefig(out_path / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"All evaluation outputs saved to: {out_path.resolve()}")
    return eval_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Plant Disease Detection Model")
    parser.add_argument(
        "--model-path",
        default=str(BACKEND_DIR / "models" / "best_model.keras"),
        help="Path to trained .keras model"
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(BACKEND_DIR / "data" / "plantvillage"),
        help="Path to evaluation dataset directory"
    )
    parser.add_argument(
        "--class-names",
        default=str(BACKEND_DIR / "models" / "class_names.json"),
        help="Path to class names json"
    )
    parser.add_argument(
        "--output-dir",
        default=str(BACKEND_DIR / "reports"),
        help="Reports output directory"
    )
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        dataset_dir=args.dataset_dir,
        class_names_path=args.class_names,
        output_dir=args.output_dir
    )
