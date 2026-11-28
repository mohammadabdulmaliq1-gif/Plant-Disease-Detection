import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


class TensorFlowAdapter:
    """TensorFlow/Keras model adapter - clean implementation without fake predictions."""

    def __init__(self, model_path: str = None, labels_path: str = None, model_json_path: str = None):
        self.model_path = model_path
        self.labels_path = labels_path
        self.model_json_path = model_json_path
        self.model = None
        self.labels = []
        self.model_type = "tensorflow"
        self.input_shape = (224, 224, 3)
        # Flag: does the model include a Rescaling layer internally?
        # Set True after loading so preprocess_image skips manual /255
        self._model_rescales_internally = False

        # Load config manager
        try:
            from ..config_manager import config_manager
            self.config = config_manager
        except ImportError:
            self.config = None

    def load_labels(self):
        """Load class labels from config or fallback to labels file."""
        if self.config:
            return self.config.get_class_names()
        import json
        from pathlib import Path
        if self.labels_path and Path(self.labels_path).exists():
            with open(self.labels_path, 'r') as f:
                return json.load(f)
        return []

    def load_model(self, model_path: str = None, model_config: dict = None):
        """Load TensorFlow/Keras model from the models directory."""
        # Resolve models directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # app/adapters -> app -> backend -> models
        backend_dir = os.path.dirname(os.path.dirname(current_dir))
        model_dir = os.path.join(backend_dir, "models")

        logger.info(f"Looking for models in: {model_dir}")

        # Check if an explicit model path was requested
        target_path = model_path or self.model_path
        if target_path:
            if os.path.exists(target_path):
                candidates = [target_path]
            elif os.path.exists(os.path.join(backend_dir, target_path)):
                candidates = [os.path.join(backend_dir, target_path)]
            else:
                raise FileNotFoundError(f"Specified model path does not exist: {target_path}")
        else:
            # Try model files in default priority order
            candidates = [
                os.path.join(model_dir, "best_model.keras"),
                os.path.join(model_dir, "best_model.h5"),
                os.path.join(model_dir, "plant_disease_model_improved.keras"),
                os.path.join(model_dir, "plant_disease_model.keras"),
                os.path.join(model_dir, "tensorflow_model"),
                os.path.join(model_dir, "plant_disease_model.h5"),
            ]

        try:
            try:
                from tensorflow import keras
                import tensorflow as tf
            except (ImportError, AttributeError):
                try:
                    import keras
                    import tensorflow as tf
                except (ImportError, AttributeError):
                    import tensorflow as tf
                    keras = getattr(tf, 'keras', None)
                    if keras is None:
                        raise ImportError("Cannot import keras from tensorflow or standalone keras")

            loaded_path = None
            for candidate in candidates:
                if os.path.exists(candidate):
                    logger.info(f"Loading model from: {candidate}")
                    self.model = keras.models.load_model(candidate)
                    loaded_path = candidate
                    logger.info(f"Model loaded successfully from {candidate}")
                    logger.info(f"Input shape: {self.model.input_shape}")
                    logger.info(f"Output shape: {self.model.output_shape}")
                    break

            if self.model is None:
                raise FileNotFoundError(
                    f"No trained model file found in {model_dir}. "
                    f"Please run the training pipeline first: python backend/train.py"
                )

            # Detect if model has an internal Rescaling layer (trained with /255 inside)
            self._model_rescales_internally = self._check_rescaling_layer()
            logger.info(f"Model rescales internally: {self._model_rescales_internally}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # Load class labels - prefer class_names.json or labels.json for accuracy
        self.labels = self._load_labels_from_file_or_config(backend_dir)
        logger.info(f"Loaded {len(self.labels)} class labels: {self.labels}")

        return True

    def _check_rescaling_layer(self) -> bool:
        """Check if the model has an internal Rescaling layer."""
        try:
            for layer in self.model.layers:
                if 'rescaling' in layer.name.lower() or 'rescal' in type(layer).__name__.lower():
                    return True
            return False
        except Exception:
            return False

    def _load_labels_from_file_or_config(self, backend_dir: str):
        """Load labels matching model output indices."""
        import json
        from pathlib import Path

        # Priority 1: models/class_names.json (generated by train.py)
        class_names_file = Path(backend_dir) / "models" / "class_names.json"
        if class_names_file.exists():
            with open(class_names_file, 'r', encoding='utf-8') as f:
                labels = json.load(f)
            logger.info(f"Labels loaded from class_names file: {class_names_file}")
            return labels

        # Priority 2: labels.json
        labels_file = Path(backend_dir) / "labels.json"
        if labels_file.exists():
            with open(labels_file, 'r', encoding='utf-8') as f:
                labels = json.load(f)
            logger.info(f"Labels loaded from file: {labels_file}")
            return labels

        # Priority 3: config manager
        if self.config:
            labels = self.config.get_class_names()
            logger.info(f"Labels loaded from config manager")
            return labels

        # Priority 4: provided labels path
        if self.labels_path and Path(self.labels_path).exists():
            with open(self.labels_path, 'r') as f:
                labels = json.load(f)
            logger.info(f"Labels loaded from labels_path: {self.labels_path}")
            return labels

        logger.warning("No labels found - using empty list")
        return []

    def preprocess_image(self, image):
        """Preprocess image for inference.

        EfficientNetB0 (Keras 3 / TF 2.17+) includes its own internal
        preprocessing pipeline that expects 0-255 uint8 pixel values.
        We pass raw values without any additional /255 normalisation.
        """
        from PIL import Image as PILImage
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Get target size from config
        if self.config:
            target_size = tuple(self.config.model_config.input_shape[:2])
        else:
            target_size = self.input_shape[:2]

        image = image.resize(target_size, PILImage.LANCZOS)
        # Keep as 0-255 float32 — EfficientNetB0 handles its own normalisation
        image_array = np.array(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)
        logger.debug(f"Preprocessed image shape: {image_array.shape}, "
                     f"min={image_array.min():.1f}, max={image_array.max():.1f}")
        return image_array

    def predict(self, image_array: np.ndarray) -> np.ndarray:
        """Make prediction on image array — returns real model probabilities only."""
        if self.model is None:
            raise ValueError("Model not loaded. Run load_model() first.")

        logger.info(f"Running inference on input shape: {image_array.shape}")

        # Run model inference
        predictions = self.model.predict(image_array, verbose=0)
        logger.info(f"Prediction output shape: {predictions.shape}, max: {np.max(predictions):.4f}")

        return predictions

    def postprocess_predictions(self, predictions: np.ndarray, top_k: int = 5):
        """Convert raw model predictions to (label, confidence) pairs."""
        if len(predictions.shape) > 1:
            probs = predictions[0]
        else:
            probs = predictions

        top_k = min(top_k, len(probs))
        top_indices = np.argsort(probs)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if idx < len(self.labels):
                label = self.labels[idx]
                confidence = float(probs[idx])
                results.append((label, confidence))
                logger.info(f"  -> {label}: {confidence:.4f} ({confidence*100:.2f}%)")

        return results

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if self.model is None:
            return {
                "framework": "tensorflow",
                "status": "not_loaded",
                "architecture": "EfficientNetV2B0 for plant disease detection"
            }

        return {
            "framework": "tensorflow",
            "status": "loaded",
            "input_shape": str(self.model.input_shape),
            "output_shape": str(self.model.output_shape),
            "total_params": self.model.count_params(),
            "num_classes": len(self.labels),
            "architecture": "EfficientNetV2B0 for plant disease detection",
            "rescales_internally": self._model_rescales_internally,
        }
