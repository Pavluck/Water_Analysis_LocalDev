// ~~~~~~~~~ Water AI Video Page ~~~~~~~~

import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { CgDrop } from "react-icons/cg";
import "./videoupload.css";

// ~~~~~~~~~~~ Preliminaries ~~~~~~~~~~~~
const API_BASE_URL =
  process.env.REACT_APP_API_URL || "SECRET";
const SUPPORTED_TYPES = [
  "image/jpeg",
  "image/png",
  "image/webp",
  "image/bmp",
  "video/mp4",
  "video/webm",
  "video/quicktime",
];

// ~~~~~~~~ Payload & Functions ~~~~~~~~~
function WaterUpload() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  useEffect(
    () => () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    },
    [previewUrl]
  );

  const handleFileChange = (event) => {
    const file = event.target.files?.[0];
    setResult(null);
    setError("");

    if (!file) return;
    if (!SUPPORTED_TYPES.includes(file.type)) {
      setSelectedFile(null);
      setPreviewUrl("");
      setError("Choose a JPEG, PNG, WebP, BMP, MP4, WebM, or MOV file.");
      event.target.value = "";
      return;
    }

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!selectedFile) {
      setError("Choose an image or video before starting the analysis.");
      return;
    }

    setIsLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(
        `${API_BASE_URL.replace(/\/$/, "")}/api/analyze-water`,
        { method: "POST", body: formData }
      );
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "The server could not analyze this file.");
      }
      if (typeof data.potable !== "boolean") {
        throw new Error("The server returned an invalid analysis result.");
      }
      setResult(data);
    } catch (requestError) {
      setError(
        requestError.message ||
          "Could not connect to the Water AI server. Check that the backend is running."
      );
    } finally {
      setIsLoading(false);
    }
  };
  const clearFile = () => {
    setSelectedFile(null);
    setPreviewUrl("");
    setResult(null);
    setError("");
    const input = document.getElementById("water-upload-file");
    if (input) input.value = "";
  };

  const isVideo = selectedFile?.type.startsWith("video/");
  const mediaLabel = isVideo ? "VIDEO FOOTAGE" : "SOURCE IMAGE";

  // ~~~~~~~~~~~ Page Content ~~~~~~~~~~~
  return (
    <main className="water-upload-page">
      <header className="water-upload-header">
        <Link className="water-upload-brand" to="/">
          <CgDrop aria-hidden="true" />
          <span>Water Analysis AI</span>
        </Link>
        <span className="water-upload-model">RESNET · CNNv2.5</span>
      </header>

      <section className="water-upload-content">
        <div className="water-upload-intro">
          <span className="water-upload-eyebrow">COMPUTER VISION ANALYSIS</span>
          <h1>See what&apos;s in your water.</h1>
          <p>
            Upload a water image or video. The ResNet model will analyze the
            sample and highlight the visual features it used.
          </p>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="water-upload-frames">
            <section className="water-upload-frame">
              <div className="water-upload-frame-heading">
                <span><i className="water-upload-dot" /> 01 / SOURCE</span>
                <span>{mediaLabel}</span>
              </div>
              <div className="water-upload-media">
                {previewUrl ? (
                  isVideo ? (
                    <video
                      className="water-upload-preview"
                      src={previewUrl}
                      controls
                      playsInline
                      aria-label="Uploaded source video"
                    />
                  ) : (
                    <img
                      className="water-upload-preview"
                      src={previewUrl}
                      alt="Uploaded water sample"
                    />
                  )
                ) : (
                  <div className="water-upload-empty">
                    <span className="water-upload-empty-icon"><CgDrop /></span>
                    <span>Your image or video will appear here</span>
                    <small>JPEG · PNG · WEBP · MP4 · WEBM · MOV</small>
                  </div>
                )}
              </div>
            </section>

            <section className="water-upload-frame">
              <div className="water-upload-frame-heading">
                <span><i className="water-upload-dot water-upload-dot-ai" /> 02 / MODEL VIEW</span>
                <span>FEATURE MAP OVERLAY</span>
              </div>
              <div className="water-upload-media">
                {result?.overlay_url ? (
                  <video
                    className="water-upload-preview"
                    src={result.overlay_url}
                    controls
                    playsInline
                    aria-label="Video with ResNet feature-map overlay"
                  />
                ) : result?.overlay_image ? (
                  <img
                    className="water-upload-preview"
                    src={result.overlay_image}
                    alt="Water sample with ResNet feature-map overlay"
                  />
                ) : (
                  <div className="water-upload-empty water-upload-empty-model">
                    <span className="water-upload-scan" aria-hidden="true" />
                    <span>
                      {isLoading
                        ? "Mapping visual features…"
                        : "The model overlay will appear here"}
                    </span>
                    <small>RESNET LAYER 4 ACTIVATIONS</small>
                  </div>
                )}
              </div>
            </section>
          </div>

          <div className="water-upload-controls">
            <div className="water-upload-file-info">
              <label className="water-upload-choose" htmlFor="water-upload-file">
                <span aria-hidden="true">＋</span> Choose file
              </label>
              <input
                className="water-upload-input"
                id="water-upload-file"
                type="file"
                accept="image/jpeg,image/png,image/webp,image/bmp,video/mp4,video/webm,video/quicktime"
                onChange={handleFileChange}
              />
              {selectedFile ? (
                <span className="water-upload-filename" title={selectedFile.name}>
                  {selectedFile.name}
                  <small>{(selectedFile.size / (1024 * 1024)).toFixed(1)} MB</small>
                </span>
              ) : (
                <span className="water-upload-hint">
                  Up to 100 MB · Video up to 2 minutes
                </span>
              )}
              {selectedFile && (
                <button
                  className="water-upload-clear"
                  type="button"
                  onClick={clearFile}
                  aria-label="Remove selected file"
                >
                  Remove
                </button>
              )}
            </div>
            <button
              className="water-upload-submit"
              type="submit"
              disabled={!selectedFile || isLoading}
            >
              {isLoading ? (
                <><span className="water-upload-spinner" /> Analyzing</>
              ) : (
                <>Analyze sample <span aria-hidden="true">↗</span></>
              )}
            </button>
          </div>
        </form>

        {error && (
          <div className="water-upload-message water-upload-error" role="alert">
            {error}
          </div>
        )}

        {result && (
          <section
            className={`water-upload-result ${
              result.potable ? "water-upload-result-safe" : "water-upload-result-unsafe"
            }`}
            aria-live="polite"
          >
            <div>
              <span className="water-upload-eyebrow">ANALYSIS RESULT</span>
              <h2>{result.potable ? "Appears potable" : "May not be potable"}</h2>
              <p>
                {result.potable
                  ? "The model classified this sample as potable."
                  : "The model classified this sample as not potable."}
              </p>
            </div>
            <div className="water-upload-result-meta">
              <span>MODEL CONFIDENCE</span>
              <strong>{(result.confidence * 100).toFixed(1)}%</strong>
              <small>{result.model}</small>
            </div>
          </section>
        )}

        <p className="water-upload-disclaimer">
          Visual AI analysis is not a laboratory water-quality test. Do not use
          this result alone to determine whether water is safe to drink.
        </p>
      </section>
    </main>
  );
}

export default WaterUpload;
