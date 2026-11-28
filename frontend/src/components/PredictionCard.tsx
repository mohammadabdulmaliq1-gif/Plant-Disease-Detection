import { PredictionResult } from "@/types/api";
import {
  AlertTriangle,
  Clock,
  Shield,
  Zap,
  Eye,
  AlertCircle,
  DollarSign,
  Activity,
  CheckCircle2,
  Leaf,
  BarChart2,
} from "lucide-react";

interface PredictionCardProps {
  prediction: PredictionResult;
}

export function PredictionCard({ prediction }: PredictionCardProps) {
  const isHealthy = Boolean(
    prediction.is_healthy ||
    prediction.disease.toLowerCase() === "healthy" ||
    prediction.label.toLowerCase().endsWith("healthy")
  );

  const getSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case "none":
        return "text-emerald-700 bg-emerald-100 border border-emerald-200";
      case "low":
        return "text-green-700 bg-green-100 border border-green-200";
      case "medium":
      case "moderate":
        return "text-yellow-700 bg-yellow-100 border border-yellow-200";
      case "high":
        return "text-red-700 bg-red-100 border border-red-200";
      default:
        return "text-gray-700 bg-gray-100 border border-gray-200";
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity.toLowerCase()) {
      case "none":
        return <CheckCircle2 className="h-4 w-4 text-emerald-600" />;
      case "low":
        return <Shield className="h-4 w-4 text-green-600" />;
      case "medium":
      case "moderate":
        return <Clock className="h-4 w-4 text-yellow-600" />;
      case "high":
        return <AlertTriangle className="h-4 w-4 text-red-600" />;
      default:
        return <Zap className="h-4 w-4 text-gray-600" />;
    }
  };

  const formatConfidence = (confidence: number) => {
    return `${(confidence * 100).toFixed(1)}%`;
  };

  const formatDate = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  const topPredictions = prediction.top_predictions && prediction.top_predictions.length > 0
    ? prediction.top_predictions
    : [prediction];

  return (
    <div className="card p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border-b border-gray-100 pb-4">
        <div className="flex items-center space-x-2">
          <Leaf className="h-6 w-6 text-green-600" />
          <h2 className="text-2xl font-bold text-gray-900">
            Diagnosis & Recommendations
          </h2>
        </div>
        <div className="text-sm text-gray-500">
          {prediction.timestamp
            ? formatDate(prediction.timestamp)
            : new Date().toLocaleString()}
        </div>
      </div>

      {/* Health Status Banner */}
      <div
        className={`p-4 rounded-xl flex items-center justify-between border ${
          isHealthy
            ? "bg-emerald-50 border-emerald-300 text-emerald-900"
            : "bg-amber-50 border-amber-300 text-amber-900"
        }`}
      >
        <div className="flex items-center space-x-3">
          {isHealthy ? (
            <CheckCircle2 className="h-7 w-7 text-emerald-600 flex-shrink-0" />
          ) : (
            <AlertTriangle className="h-7 w-7 text-amber-600 flex-shrink-0" />
          )}
          <div>
            <div className="font-bold text-lg">
              {isHealthy ? "Plant Status: Healthy" : "Plant Status: Diseased / Infected"}
            </div>
            <div className="text-sm opacity-90">
              {isHealthy
                ? `No pathogen symptoms detected. The ${prediction.plant} leaf appears healthy and vigorous.`
                : `Signs of ${prediction.disease} identified on the ${prediction.plant} leaf.`}
            </div>
          </div>
        </div>
        <span
          className={`px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wider ${
            isHealthy
              ? "bg-emerald-200 text-emerald-900"
              : "bg-amber-200 text-amber-900"
          }`}
        >
          {isHealthy ? "Healthy" : "Infected"}
        </span>
      </div>

      {/* Primary Attributes Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-gray-50 rounded-lg p-4 border border-gray-100">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-1">
            Plant Species
          </h3>
          <p className="text-lg font-bold text-gray-900">
            {prediction.plant || "Unknown"}
          </p>
        </div>

        <div className="bg-gray-50 rounded-lg p-4 border border-gray-100">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-1">
            Identified Condition
          </h3>
          <p className="text-lg font-bold text-gray-900">
            {prediction.disease}
          </p>
        </div>

        <div className="bg-gray-50 rounded-lg p-4 border border-gray-100">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-1">
            Confidence Score
          </h3>
          <div className="flex items-baseline space-x-2">
            <p className="text-lg font-bold text-gray-900">
              {formatConfidence(prediction.confidence)}
            </p>
            <span className="text-xs text-gray-500 font-medium">EfficientNetV2B0</span>
          </div>
        </div>

        <div className="bg-gray-50 rounded-lg p-4 border border-gray-100">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-1">
            Severity Level
          </h3>
          <div
            className={`inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-xs font-semibold ${getSeverityColor(
              prediction.severity
            )}`}
          >
            {getSeverityIcon(prediction.severity)}
            <span className="capitalize">{prediction.severity}</span>
          </div>
        </div>
      </div>

      {/* Top Predictions Ranking */}
      {topPredictions.length > 0 && (
        <div className="bg-gray-50 rounded-xl p-5 border border-gray-200 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-semibold text-gray-900 flex items-center gap-2">
              <BarChart2 className="h-5 w-5 text-primary-600" />
              Top Predictions (Ranked Probabilities)
            </h3>
            <span className="text-xs text-gray-500">Sorted by model confidence</span>
          </div>
          <div className="space-y-2.5 pt-1">
            {topPredictions.slice(0, 3).map((item, idx) => {
              const itemPercent = (item.confidence * 100).toFixed(1);
              const isTop = idx === 0;
              return (
                <div key={item.label || idx} className="space-y-1">
                  <div className="flex items-center justify-between text-sm">
                    <span className="flex items-center gap-2">
                      <span
                        className={`w-5 h-5 flex items-center justify-center rounded-full text-xs font-bold ${
                          isTop
                            ? "bg-primary-600 text-white"
                            : "bg-gray-300 text-gray-700"
                        }`}
                      >
                        {idx + 1}
                      </span>
                      <span className={`font-medium ${isTop ? "text-gray-900 font-semibold" : "text-gray-700"}`}>
                        {item.plant ? `${item.plant} - ` : ""}{item.disease}
                      </span>
                    </span>
                    <span className="font-semibold text-gray-900">{itemPercent}%</span>
                  </div>
                  {/* Progress bar */}
                  <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                    <div
                      className={`h-2 rounded-full transition-all duration-500 ${
                        isTop ? "bg-primary-600" : "bg-gray-400"
                      }`}
                      style={{ width: `${Math.max(item.confidence * 100, 2)}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Symptoms & Causes Section */}
      {(prediction.symptoms?.length ||
        prediction.causes ||
        prediction.urgency ||
        prediction.economic_impact) && (
        <div className="space-y-4">
          <h3 className="text-lg font-bold text-gray-900 flex items-center gap-2">
            <Eye className="h-5 w-5 text-gray-700" />
            Disease Profile & Symptoms
          </h3>

          {/* Symptoms List */}
          {prediction.symptoms && prediction.symptoms.length > 0 && (
            <div className="bg-blue-50/70 border border-blue-200 rounded-lg p-4">
              <h4 className="font-semibold text-blue-900 mb-2 flex items-center">
                <Eye className="h-4 w-4 mr-2 text-blue-700" />
                Symptoms to Look For
              </h4>
              <ul className="text-blue-900 text-sm space-y-1.5">
                {prediction.symptoms.map((symptom, index) => (
                  <li key={index} className="flex items-start">
                    <span className="mr-2 font-bold">•</span>
                    <span>{symptom}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Causes */}
          {prediction.causes && prediction.causes.trim().length > 0 && (
            <div className="bg-amber-50/70 border border-amber-200 rounded-lg p-4">
              <h4 className="font-semibold text-amber-900 mb-1 flex items-center">
                <AlertCircle className="h-4 w-4 mr-2 text-amber-700" />
                Underlying Causes & Transmission
              </h4>
              <p className="text-amber-900 text-sm leading-relaxed">{prediction.causes}</p>
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Urgency */}
            {prediction.urgency && prediction.urgency.trim().length > 0 && (
              <div className="bg-rose-50/70 border border-rose-200 rounded-lg p-4">
                <h4 className="font-semibold text-rose-900 mb-1 flex items-center">
                  <Activity className="h-4 w-4 mr-2 text-rose-700" />
                  Treatment Urgency
                </h4>
                <p className="text-rose-900 text-sm">{prediction.urgency}</p>
              </div>
            )}

            {/* Economic Impact */}
            {prediction.economic_impact &&
              prediction.economic_impact.trim().length > 0 && (
                <div className="bg-purple-50/70 border border-purple-200 rounded-lg p-4">
                  <h4 className="font-semibold text-purple-900 mb-1 flex items-center">
                    <DollarSign className="h-4 w-4 mr-2 text-purple-700" />
                    Agricultural & Economic Impact
                  </h4>
                  <p className="text-purple-900 text-sm leading-relaxed">
                    {prediction.economic_impact}
                  </p>
                </div>
              )}
          </div>
        </div>
      )}

      {/* Treatment Recommendations */}
      <div className="space-y-4">
        <h3 className="text-lg font-bold text-gray-900 flex items-center gap-2">
          <Shield className="h-5 w-5 text-gray-700" />
          Treatment & Management Strategy
        </h3>

        {/* Chemical Treatments */}
        {prediction.treatment?.chemical &&
          prediction.treatment.chemical.trim().length > 0 && (
            <div className="bg-blue-50/80 border border-blue-200 rounded-lg p-4">
              <h4 className="font-semibold text-blue-900 mb-1 flex items-center">
                <Zap className="h-4 w-4 mr-2 text-blue-600" />
                Chemical Treatments
              </h4>
              <p className="text-blue-900 text-sm leading-relaxed">
                {prediction.treatment.chemical}
              </p>
            </div>
          )}

        {/* Cultural Treatments */}
        {prediction.treatment?.cultural &&
          prediction.treatment.cultural.trim().length > 0 && (
            <div className="bg-green-50/80 border border-green-200 rounded-lg p-4">
              <h4 className="font-semibold text-green-900 mb-1 flex items-center">
                <Shield className="h-4 w-4 mr-2 text-green-600" />
                Cultural & Field Practices
              </h4>
              <p className="text-green-900 text-sm leading-relaxed">
                {prediction.treatment.cultural}
              </p>
            </div>
          )}

        {/* Preventive Measures */}
        {prediction.treatment?.preventive &&
          prediction.treatment.preventive.trim().length > 0 && (
            <div className="bg-teal-50/80 border border-teal-200 rounded-lg p-4">
              <h4 className="font-semibold text-teal-900 mb-1 flex items-center">
                <Clock className="h-4 w-4 mr-2 text-teal-600" />
                Preventive Measures & Monitoring
              </h4>
              <p className="text-teal-900 text-sm leading-relaxed">
                {prediction.treatment.preventive}
              </p>
            </div>
          )}
      </div>

      {/* Advisory Note */}
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 text-xs text-gray-600">
        <strong>Agronomic Advisory:</strong> Predictions are generated by a deep neural network
        (EfficientNetV2B0 transfer learning architecture) trained on PlantVillage disease samples.
        For commercial farming decisions, verify symptoms with your local agricultural extension service.
      </div>
    </div>
  );
}
