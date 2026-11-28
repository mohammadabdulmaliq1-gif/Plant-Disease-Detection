import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { useMutation } from "@tanstack/react-query";
import { Upload, AlertCircle, CheckCircle, Loader2, ImageIcon, RefreshCw } from "lucide-react";
import { predict } from "@/services/api";
import { PredictionResult } from "@/types/api";
import { PredictionCard } from "@/components/PredictionCard";

export function HomePage() {
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const mutation = useMutation({
    mutationFn: predict,
    onSuccess: (data) => {
      setPrediction(data);
    },
    onError: (error) => {
      console.error("Prediction error:", error);
    },
  });

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      const file = acceptedFiles[0];
      if (file) {
        // Clear previous state
        setPrediction(null);

        // Create image preview
        const url = URL.createObjectURL(file);
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        setPreviewUrl(url);

        mutation.mutate(file);
      }
    },
    [mutation, previewUrl]
  );

  const handleReset = () => {
    setPrediction(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }
    mutation.reset();
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "image/*": [".png", ".jpg", ".jpeg", ".webp"],
    },
    multiple: false,
    maxSize: 10 * 1024 * 1024, // 10MB
  });

  // Determine friendly error message
  const getErrorMessage = (error: unknown): string => {
    const msg = (error as any)?.message || "";
    if (msg.includes("503") || msg.toLowerCase().includes("model service") || msg.toLowerCase().includes("not ready")) {
      return "The AI model is still loading. Please wait a moment and try again.";
    }
    if (msg.includes("413") || msg.toLowerCase().includes("too large")) {
      return "Image file is too large. Please use an image under 10 MB.";
    }
    if (msg.includes("400") || msg.toLowerCase().includes("invalid file")) {
      return "Invalid file format. Please upload a JPG, PNG, or WEBP image.";
    }
    if (msg.toLowerCase().includes("failed to fetch") || msg.toLowerCase().includes("network")) {
      return "Cannot connect to the backend server. Make sure the backend is running on port 8000.";
    }
    return msg || "An error occurred while analysing the image. Please try again.";
  };

  return (
    <div className="container py-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Plant Disease Detection
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Upload an image of a plant leaf to detect diseases and get treatment
            recommendations using our AI-powered system.
          </p>
        </div>

        {/* Upload Area */}
        <div className="mb-8">
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer ${
              isDragActive
                ? "border-primary-500 bg-primary-50"
                : "border-gray-300 hover:border-primary-400 hover:bg-gray-50"
            }`}
          >
            <input {...getInputProps()} />
            <div className="flex flex-col items-center space-y-4">
              <Upload className="h-12 w-12 text-gray-400" />
              <div>
                <p className="text-lg font-medium text-gray-900">
                  {isDragActive
                    ? "Drop the image here..."
                    : "Drag and drop an image, or click to select"}
                </p>
                <p className="text-sm text-gray-500 mt-1">
                  PNG, JPG, JPEG, WEBP up to 10MB
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Image Preview + Loading State */}
        {previewUrl && mutation.isPending && (
          <div className="card p-6 mb-8">
            <div className="flex items-center gap-6">
              <div className="flex-shrink-0 w-32 h-32 rounded-lg overflow-hidden border border-gray-200 bg-gray-50 flex items-center justify-center">
                <img
                  src={previewUrl}
                  alt="Uploaded leaf"
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="flex items-center space-x-3">
                <Loader2 className="h-6 w-6 animate-spin text-primary-600 flex-shrink-0" />
                <div>
                  <span className="text-lg font-medium text-gray-900 block">
                    Analysing image...
                  </span>
                  <span className="text-sm text-gray-500">
                    The AI model is processing your leaf image. This may take a few seconds.
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Error State */}
        {mutation.isError && (
          <div className="card p-6 mb-8 border-red-200 bg-red-50">
            <div className="flex items-start space-x-3">
              <AlertCircle className="h-6 w-6 text-red-600 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <h3 className="text-lg font-medium text-red-900">
                  Analysis Failed
                </h3>
                <p className="text-red-700 mt-1">
                  {getErrorMessage(mutation.error)}
                </p>
                <button
                  onClick={handleReset}
                  className="mt-3 inline-flex items-center gap-2 text-sm text-red-700 hover:text-red-900 font-medium"
                >
                  <RefreshCw className="h-4 w-4" />
                  Try another image
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Success State */}
        {prediction && !mutation.isPending && (
          <div className="space-y-6">
            <div className="card p-6 border-green-200 bg-green-50">
              <div className="flex items-start gap-6">
                {/* Preview thumbnail */}
                {previewUrl && (
                  <div className="flex-shrink-0 w-24 h-24 rounded-lg overflow-hidden border border-green-200">
                    <img
                      src={previewUrl}
                      alt="Analysed leaf"
                      className="w-full h-full object-cover"
                    />
                  </div>
                )}
                <div className="flex items-center space-x-3">
                  <CheckCircle className="h-6 w-6 text-green-600 flex-shrink-0" />
                  <div>
                    <h3 className="text-lg font-medium text-green-900">
                      Analysis Complete
                    </h3>
                    <p className="text-green-700 mt-1">
                      Disease detection analysis completed successfully.
                    </p>
                    <button
                      onClick={handleReset}
                      className="mt-2 inline-flex items-center gap-2 text-sm text-green-700 hover:text-green-900 font-medium"
                    >
                      <Upload className="h-4 w-4" />
                      Analyse another image
                    </button>
                  </div>
                </div>
              </div>
            </div>

            <PredictionCard prediction={prediction} />
          </div>
        )}

        {/* Features - only show when idle */}
        {!prediction && !mutation.isPending && !mutation.isError && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-12">
            <div className="text-center">
              <div className="bg-primary-100 rounded-full p-3 w-12 h-12 mx-auto mb-4 flex items-center justify-center">
                <Upload className="h-6 w-6 text-primary-600" />
              </div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                Easy Upload
              </h3>
              <p className="text-gray-600">
                Simply drag and drop or click to upload plant images
              </p>
            </div>

            <div className="text-center">
              <div className="bg-primary-100 rounded-full p-3 w-12 h-12 mx-auto mb-4 flex items-center justify-center">
                <ImageIcon className="h-6 w-6 text-primary-600" />
              </div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                AI Detection
              </h3>
              <p className="text-gray-600">
                EfficientNetB0 model trained on 20,000+ PlantVillage images
              </p>
            </div>

            <div className="text-center">
              <div className="bg-primary-100 rounded-full p-3 w-12 h-12 mx-auto mb-4 flex items-center justify-center">
                <CheckCircle className="h-6 w-6 text-primary-600" />
              </div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                Treatment Guide
              </h3>
              <p className="text-gray-600">
                Get detailed treatment recommendations for identified diseases
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
