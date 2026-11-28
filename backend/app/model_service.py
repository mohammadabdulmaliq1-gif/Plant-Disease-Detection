import time
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image
import io

from .adapters import TensorFlowAdapter, PyTorchAdapter, ONNXAdapter
from .config import settings
from .config_manager import config_manager
from .models import PredictionResult, PredictionResponse, TreatmentInfo


class ModelService:
    """Service for managing model loading and inference."""
    
    def __init__(self):
        self.adapter = None
        self.model_loaded = False
        self.model_info = {}
        self.config = config_manager
        
    def load_model(self) -> None:
        """Load the appropriate model based on configuration."""
        import os
        try:
            model_type = settings.model_type.lower()
            
            if model_type == "tensorflow":
                try:
                    self.adapter = TensorFlowAdapter(
                        model_path=settings.model_path,
                        labels_path=getattr(settings, 'model_labels_path', None),
                        model_json_path=getattr(settings, 'model_json_path', None)
                    )
                    self.adapter.load_model()
                except Exception as tf_err:
                    print(f"TensorFlow model failed to load ({tf_err}). Checking for ONNX fallback...")
                    onnx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "plant_disease_model.onnx")
                    if os.path.exists(onnx_path):
                        self.adapter = ONNXAdapter(
                            model_path=onnx_path,
                            labels_path=getattr(settings, 'model_labels_path', None)
                        )
                        self.adapter.load_model()
                        model_type = "onnx"
                    else:
                        raise tf_err
            elif model_type == "pytorch":
                self.adapter = PyTorchAdapter(
                    model_path=settings.model_path,
                    labels_path=getattr(settings, 'model_labels_path', None)
                )
                self.adapter.load_model()
            elif model_type == "onnx":
                self.adapter = ONNXAdapter(
                    model_path=settings.model_path,
                    labels_path=getattr(settings, 'model_labels_path', None)
                )
                self.adapter.load_model()
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            self.model_loaded = True
            self.model_info = self.adapter.get_model_info()
            
            print(f"Model loaded successfully: {model_type}")
            print(f"Classes loaded: {len(self.config.get_class_names())}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False
            raise
    
    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        """Make prediction on image bytes using dynamic configuration with uncertainty detection."""
        if not self.model_loaded or not self.adapter:
            raise ValueError("Model not loaded")
        
        start_time = time.time()
        
        try:
            # Load image from bytes
            image = Image.open(io.BytesIO(image_bytes))
            
            # Preprocess image
            processed_image = self.adapter.preprocess_image(image)
            
            # Make prediction
            predictions = self.adapter.predict(processed_image)
            
            # Check prediction confidence
            max_confidence = float(np.max(predictions[0])) if len(predictions.shape) > 1 else float(np.max(predictions))
            uncertainty_threshold = getattr(settings, 'uncertainty_threshold', 0.20)
            
            # Postprocess results
            results = self.adapter.postprocess_predictions(predictions, top_k=5)
            print(f"[DEBUG] Raw predictions from adapter: {results}")
            
            # Calculate inference time
            inference_ms = int((time.time() - start_time) * 1000)
            
            # Build detailed predictions with dynamic configuration
            detailed_predictions = []
            for label, confidence in results:
                print(f"[DEBUG] Processing label: {label}, confidence: {confidence}")
                disease_class = self.config.get_class_by_id(label)
                print(f"[DEBUG] Disease class for label {label}: {disease_class}")
                if disease_class:
                    severity_info = self.config.get_severity_for_class(label)
                    is_healthy = "healthy" in label.lower() or "healthy" in disease_class.disease.lower()
                    
                    prediction_result = PredictionResult(
                        label=label,
                        confidence=confidence,
                        class_name=disease_class.name,
                        plant=disease_class.plant,
                        disease=disease_class.disease,
                        severity=disease_class.severity,
                        severity_level=severity_info.level if severity_info else 0,
                        severity_color=severity_info.color if severity_info else "#10B981",
                        treatment=TreatmentInfo(
                            chemical=disease_class.treatment.chemical,
                            cultural=disease_class.treatment.cultural,
                            preventive=disease_class.treatment.preventive
                        ),
                        symptoms=disease_class.symptoms or [],
                        causes=disease_class.causes or "",
                        urgency=disease_class.urgency or "",
                        economic_impact=disease_class.economic_impact or "",
                        is_healthy=is_healthy
                    )
                    detailed_predictions.append(prediction_result)
                else:
                    parts = label.split("___") if "___" in label else label.split("_", 1)
                    plant_name = parts[0].replace("__", " ").replace("_", " ").title()
                    disease_name = parts[1].replace("_", " ").title() if len(parts) > 1 else "Condition"
                    is_healthy = "healthy" in label.lower()
                    prediction_result = PredictionResult(
                        label=label,
                        confidence=confidence,
                        class_name=f"{plant_name} - {disease_name}",
                        plant=plant_name,
                        disease=disease_name,
                        severity="none" if is_healthy else "moderate",
                        severity_level=0 if is_healthy else 2,
                        severity_color="#10B981" if is_healthy else "#F59E0B",
                        treatment=TreatmentInfo(
                            chemical="No chemical treatment needed" if is_healthy else "Apply appropriate treatment",
                            cultural="Maintain good plant care",
                            preventive="Monitor plant regularly"
                        ),
                        symptoms=["Healthy foliage"] if is_healthy else ["Visible disease symptoms"],
                        causes="Normal healthy growth" if is_healthy else "Pathogenic infection",
                        urgency="None" if is_healthy else "Monitor and treat within 1-2 weeks",
                        economic_impact="None" if is_healthy else "Potential yield reduction",
                        is_healthy=is_healthy
                    )
                    detailed_predictions.append(prediction_result)
            
            # Handle uncertain predictions
            if max_confidence < uncertainty_threshold:
                print(f"[DEBUG] Uncertain prediction detected (max confidence: {max_confidence:.2%})")
                uncertain_result = PredictionResult(
                    label="uncertain",
                    confidence=max_confidence,
                    class_name="Uncertain Diagnosis",
                    plant="Unknown",
                    disease="Uncertain",
                    severity="unknown",
                    severity_level=0,
                    severity_color="#6B7280",
                    treatment=TreatmentInfo(
                        chemical="Consult agricultural expert for proper diagnosis",
                        cultural="Ensure proper plant care and monitor for changes",
                        preventive="Consider professional plant pathology services"
                    ),
                    symptoms=[
                        "Image quality or plant condition not clear enough for confident diagnosis",
                        "Disease symptoms may be early stage or atypical",
                        "Multiple diseases may be present"
                    ],
                    causes="Insufficient visual information or unsupported image for accurate classification",
                    urgency="Recommend professional consultation within 1-2 weeks",
                    economic_impact="Variable - depends on actual condition",
                    is_healthy=False
                )
                detailed_predictions.insert(0, uncertain_result)
            
            # Get top prediction and confidence level
            top_prediction = detailed_predictions[0] if detailed_predictions else None
            top_3 = detailed_predictions[:3] if detailed_predictions else []
            is_healthy = top_prediction.is_healthy if top_prediction else False
            confidence_level = self.config.get_confidence_level(
                top_prediction.confidence if top_prediction else 0.0
            )
            
            return {
                "predictions": [pred.model_dump() if hasattr(pred, 'model_dump') else pred.dict() for pred in detailed_predictions],
                "top_predictions": [pred.model_dump() if hasattr(pred, 'model_dump') else pred.dict() for pred in top_3],
                "top_prediction": (top_prediction.model_dump() if hasattr(top_prediction, 'model_dump') else top_prediction.dict()) if top_prediction else None,
                "confidence_level": confidence_level,
                "is_healthy": is_healthy,
                "model_info": {
                    "name": self.model_info.get("architecture", "EfficientNetV2B0 for plant disease detection"),
                    "version": "2.0.0",
                    "framework": self.model_info.get("framework", "tensorflow"),
                    "input_shape": str(self.model_info.get("input_shape", "(None, 224, 224, 3)")),
                    "total_params": self.model_info.get("total_params", 0)
                },
                "inference_ms": inference_ms,
                "timestamp": datetime.utcnow().isoformat(),
                "uncertainty_detected": max_confidence < uncertainty_threshold,
                "max_confidence": float(max_confidence)
            }
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.model_loaded:
            return {"error": "Model not loaded"}
        
        model_config = self.config.model_config
        return {
            "name": model_config.name,
            "version": model_config.version,
            "description": model_config.description,
            "framework": model_config.framework,
            "input_shape": model_config.input_shape,
            "classes": self.config.get_class_names(),
            "total_classes": len(self.config.get_class_names()),
            "supported_formats": model_config.supported_formats,
            "severity_levels": len(self.config.severity_levels),
            "treatment_types": list(self.config.treatment_types.keys())
        }
    
    def is_healthy(self) -> bool:
        """Check if the model service is healthy."""
        try:
            # Check model and configuration
            return (self.model_loaded and 
                   self.adapter is not None and 
                   self.config is not None and
                   len(self.config.get_class_names()) > 0)
        except Exception:
            return False


# Global model service instance
model_service = ModelService()