"""
Plant Disease Detection Model Training Script Wrapper.

NOTE: Synthetic/random weight generation has been completely removed in Phase 4.
This script now interfaces directly with the genuine EfficientNetV2B0 transfer
learning training pipeline in backend/train.py.
"""

import sys
import os
from pathlib import Path

# Add backend directory to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    from train import train_model, BACKEND_DIR as TRAIN_BACKEND_DIR
except ImportError as e:
    print(f"Error importing training pipeline from {BACKEND_DIR}: {e}")
    sys.exit(1)


def main():
    print("=" * 70)
    print("Plant Disease Detection - Genuine Transfer Learning Model")
    print("Synthetic/random weights have been removed.")
    print("Using EfficientNetV2B0 pretrained on ImageNet.")
    print("=" * 70)

    dataset_path = BACKEND_DIR / "data" / "plantvillage"
    if not dataset_path.exists():
        print(f"Dataset path {dataset_path} does not exist.")
        print("Please run scripts/download_dataset.py first or specify the dataset directory.")
        sys.exit(1)

    print(f"Training on dataset: {dataset_path}")
    train_model(
        dataset_dir=str(dataset_path),
        output_model_path=str(BACKEND_DIR / "models" / "best_model.keras"),
        epochs_phase1=10,
        epochs_phase2=8,
        batch_size=16
    )


if __name__ == "__main__":
    main()
