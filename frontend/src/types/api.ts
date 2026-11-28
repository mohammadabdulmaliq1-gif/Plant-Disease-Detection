export interface TreatmentInfo {
  chemical: string;
  cultural: string;
  preventive: string;
}

export interface PredictionResult {
  label: string;
  confidence: number;
  class_name: string;
  plant: string;
  disease: string;
  severity: string;
  severity_level: number;
  severity_color: string;
  treatment: TreatmentInfo;
  symptoms?: string[];
  causes?: string;
  urgency?: string;
  economic_impact?: string;
  is_healthy?: boolean;
  top_predictions?: PredictionResult[];
  timestamp?: string; // Optional since we add it in API client
}

export interface PredictionResponse {
  predictions: PredictionResult[];
  top_prediction: PredictionResult;
  top_predictions?: PredictionResult[];
  is_healthy?: boolean;
  confidence_level: string;
  model_info: {
    name: string;
    version: string;
    framework: string;
    input_shape?: string;
    total_params?: number;
  };
  inference_ms: number;
  timestamp: string;
  uncertainty_detected?: boolean;
  max_confidence?: number;
}

export interface ModelInfo {
  name: string;
  version: string;
  type: "tensorflow" | "pytorch" | "onnx";
  classes: string[];
  accuracy?: number;
  input_shape?: number[];
  description?: string;
}

export interface HealthStatus {
  status: "healthy" | "degraded" | "unhealthy";
  timestamp: string;
  checks: {
    database: boolean;
    ml_model: boolean;
    dependencies: boolean;
  };
  info: {
    model: ModelInfo;
    uptime: number;
    version: string;
  };
}

export interface ApiError {
  detail: string;
  type?: string;
  code?: string;
}

export interface UploadProgress {
  progress: number;
  status: "idle" | "uploading" | "processing" | "completed" | "error";
  error?: string;
}
