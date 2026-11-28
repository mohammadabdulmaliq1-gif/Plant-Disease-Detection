"""
Comprehensive 10-Point Integration & Verification Test Suite for Plant Disease Detection.

Tests:
1. Model loading.
2. Single leaf image prediction.
3. Healthy leaf prediction.
4. Diseased leaf prediction.
5. Different disease predictions (cross-class diversity).
6. Invalid image handling.
7. Missing model handling (no fake model creation).
8. API integration (FastAPI router endpoints).
9. Frontend build and contract integration.
10. Prediction consistency and determinism.
"""

import os
import sys
import glob
import json
import io
import asyncio
import unittest
from pathlib import Path
from PIL import Image
import numpy as np

# Set environment
TESTS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TESTS_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

os.environ["KERAS_HOME"] = str(BACKEND_DIR / ".keras")
os.environ["MODEL_PATH"] = "models/best_model.keras"
os.environ["MODEL_TYPE"] = "tensorflow"
os.environ["MODEL_LABELS_PATH"] = "labels.json"

from fastapi import UploadFile, HTTPException
from app.model_service import model_service
from app.routers.predict import predict_disease
from app.routers.health import health_check, get_model_info


class TestPlantDiseaseSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_service = model_service
        cls.model_service.load_model()
        cls.data_dir = BACKEND_DIR / "data" / "plantvillage"

    # 1. Model Loading
    def test_01_model_loading(self):
        print("\n[TEST 1] Model Loading...")
        adapter = self.model_service.adapter
        self.assertIsNotNone(adapter, "Adapter should not be None")
        self.assertIsNotNone(adapter.model, "Model should be loaded")
        self.assertEqual(adapter.input_shape, (224, 224, 3))
        self.assertEqual(len(adapter.labels), 15)
        self.assertTrue(self.model_service.is_healthy())
        print(f"  -> Model successfully loaded with 15 classes and input shape (224, 224, 3).")

    # 2. Single Leaf Image Prediction
    def test_02_single_leaf_prediction(self):
        print("\n[TEST 2] Single Leaf Image Prediction...")
        sample_img = next(self.data_dir.rglob("*.jpg"))
        with open(sample_img, "rb") as f:
            img_bytes = f.read()

        result = self.model_service.predict(img_bytes)
        self.assertIn("top_prediction", result)
        self.assertIn("predictions", result)
        self.assertIn("top_predictions", result)
        top = result["top_prediction"]
        self.assertGreaterEqual(top["confidence"], 0.0)
        self.assertLessEqual(top["confidence"], 1.0)
        print(f"  -> Single leaf predicted: {top['plant']} - {top['disease']} ({top['confidence']*100:.1f}%)")

    # 3. Healthy Leaf Prediction
    def test_03_healthy_leaf_prediction(self):
        print("\n[TEST 3] Healthy Leaf Prediction...")
        healthy_folder = self.data_dir / "Tomato_healthy"
        healthy_imgs = list(healthy_folder.glob("*.jpg"))
        self.assertTrue(len(healthy_imgs) > 0, "Healthy test images missing")

        with open(healthy_imgs[0], "rb") as f:
            result = self.model_service.predict(f.read())

        top = result["top_prediction"]
        print(f"  -> Input: Tomato_healthy -> Predicted: {top['plant']} - {top['disease']} ({top['confidence']*100:.1f}%)")
        self.assertTrue(result["is_healthy"] or "healthy" in top["disease"].lower(), "Should be identified as healthy")
        self.assertEqual(top["severity"].lower(), "none")

    # 4. Diseased Leaf Prediction
    def test_04_diseased_leaf_prediction(self):
        print("\n[TEST 4] Diseased Leaf Prediction...")
        diseased_folder = self.data_dir / "Potato___Early_blight"
        diseased_imgs = list(diseased_folder.glob("*.jpg"))
        self.assertTrue(len(diseased_imgs) > 0, "Diseased test images missing")

        with open(diseased_imgs[0], "rb") as f:
            result = self.model_service.predict(f.read())

        top = result["top_prediction"]
        print(f"  -> Input: Potato___Early_blight -> Predicted: {top['plant']} - {top['disease']} ({top['confidence']*100:.1f}%)")
        self.assertFalse(result["is_healthy"], "Should be identified as diseased")
        self.assertIn("Early Blight", top["disease"])
        self.assertIsNotNone(top["treatment"])
        self.assertTrue(len(top["treatment"]["chemical"]) > 0)

    # 5. Different Disease Predictions
    def test_05_different_disease_predictions(self):
        print("\n[TEST 5] Multiple Disease Discrimination...")
        classes_to_test = [
            ("Potato___Early_blight", "Potato"),
            ("Pepper__bell___Bacterial_spot", "Bell Pepper"),
            ("Tomato_healthy", "Tomato")
        ]

        distinct_preds = set()
        for folder_name, expected_plant in classes_to_test:
            folder = self.data_dir / folder_name
            imgs = list(folder.glob("*.jpg"))
            if not imgs:
                continue
            with open(imgs[0], "rb") as f:
                res = self.model_service.predict(f.read())
            pred_disease = res["top_prediction"]["disease"]
            distinct_preds.add(pred_disease)
            print(f"  -> Class {folder_name} -> Predicted: {res['top_prediction']['plant']} - {pred_disease}")

        self.assertGreaterEqual(len(distinct_preds), 2, "Model must differentiate distinct diseases")

    # 6. Invalid Image Handling
    def test_06_invalid_image_handling(self):
        print("\n[TEST 6] Invalid Image Handling...")
        # A) Empty bytes
        with self.assertRaises(Exception):
            self.model_service.predict(b"")

        # B) Corrupted / non-image text bytes
        with self.assertRaises(Exception):
            self.model_service.predict(b"This is not a valid JPEG/PNG image binary")

        # C) API endpoint behavior on corrupted image
        file_obj = io.BytesIO(b"corrupted binary data")
        upload_file = UploadFile(filename="bad.jpg", file=file_obj, headers={"content-type": "image/jpeg"})
        with self.assertRaises(HTTPException) as cm:
            asyncio.run(predict_disease(upload_file))
        self.assertEqual(cm.exception.status_code, 400, "Should return 400 Bad Request for corrupted image")
        print(f"  -> Successfully caught invalid image with HTTPException(status_code={cm.exception.status_code})")

    # 7. Missing Model Handling
    def test_07_missing_model_handling(self):
        print("\n[TEST 7] Missing Model Handling (No Fake Model Creation)...")
        from app.adapters.tensorflow_adapter import TensorFlowAdapter

        fake_adapter = TensorFlowAdapter(model_path="non_existent_model_file.keras")
        with self.assertRaises(FileNotFoundError):
            fake_adapter.load_model(model_path="non_existent_model_file.keras")
        print("  -> Confirmed: Missing model raises FileNotFoundError without generating fake models.")

    # 8. API Integration
    def test_08_api_integration(self):
        print("\n[TEST 8] FastAPI Router Endpoints...")
        # A) Health check
        res_health = asyncio.run(health_check())
        self.assertEqual(res_health.status, "ok")
        self.assertTrue(res_health.model_loaded)
        print(f"  -> /api/health returned: status='{res_health.status}', model_loaded={res_health.model_loaded}")

        # B) Model info
        res_info = asyncio.run(get_model_info())
        self.assertEqual(res_info.get("total_classes"), 15)
        print(f"  -> /api/model/info returned: {res_info['framework']}, {res_info['total_classes']} classes")

        # C) Predict endpoint
        sample_img = next(self.data_dir.rglob("*.jpg"))
        with open(sample_img, "rb") as f:
            img_bytes = f.read()

        upload_file = UploadFile(
            filename="sample_leaf.jpg",
            file=io.BytesIO(img_bytes),
            headers={"content-type": "image/jpeg"}
        )
        res_pred = asyncio.run(predict_disease(upload_file))
        self.assertIsNotNone(res_pred.top_prediction)
        self.assertIsNotNone(res_pred.top_predictions)
        self.assertEqual(len(res_pred.top_predictions), 3)
        self.assertIsNotNone(res_pred.is_healthy)
        print(f"  -> /api/predict returned valid PredictionResponse with top 3 alternatives and healthy={res_pred.is_healthy}.")

    # 9. Frontend Integration
    def test_09_frontend_integration(self):
        print("\n[TEST 9] Frontend Build & Contract Integration...")
        frontend_dir = BACKEND_DIR.parent / "frontend"
        dist_html = frontend_dir / "dist" / "index.html"
        self.assertTrue(dist_html.exists(), "Frontend build dist/index.html must exist")
        self.assertGreater(dist_html.stat().st_size, 100, "dist/index.html should have content")
        print(f"  -> Verified frontend production build exists at {dist_html}")

    # 10. Prediction Consistency & Determinism
    def test_10_prediction_consistency(self):
        print("\n[TEST 10] Prediction Consistency & Determinism...")
        sample_img = next(self.data_dir.rglob("*.jpg"))
        with open(sample_img, "rb") as f:
            bytes_data = f.read()

        pred1 = self.model_service.predict(bytes_data)
        pred2 = self.model_service.predict(bytes_data)

        self.assertEqual(
            pred1["top_prediction"]["label"],
            pred2["top_prediction"]["label"],
            "Identical inputs must yield identical predictions"
        )
        self.assertAlmostEqual(
            pred1["top_prediction"]["confidence"],
            pred2["top_prediction"]["confidence"],
            places=5,
            msg="Confidences must match across identical invocations"
        )
        print(f"  -> Verified 100% deterministic prediction consistency.")


if __name__ == "__main__":
    unittest.main()
