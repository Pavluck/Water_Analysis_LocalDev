// ~~~~~~~~~ NP Water AI Video Page ~~~~~~~~
// Formatted Better, Third window fails to display unlike v1 

// Dependencies
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { CgDrop } from "react-icons/cg";
import "./videoupload.css";

// ~~~~~~~~~~~ Preliminaries ~~~~~~~~~~~~
const API_BASE_URL =
  process.env.REACT_APP_API_URL || "SECRET"; 

// valid file types for upload, need to test livestream
const SUPPORTED_TYPES = [
  "image/jpeg",
  "image/png",
  "image/webp",
  "image/bmp",
  "video/mp4",
  "video/webm",
  "video/quicktime",
];

// ~~~~~~~~~~~ Payload ~~~~~~~~~~~~
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
        // Extras  since I keep forgetting to start the backend
        requestError instanceof TypeError
          ? `Oopsie Daisy ~ Could not reach the Water AI at ${API_BASE_URL}. 
          Start the Flask backend and check that this page's origin is allowed. (CORS)`
          : requestError.message ||
          "Hey. Could not connect to the Water AI server. Check that the backend is running."
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

  return (
    <main className="water-upload-page">
      <header className="water-upload-header">
        <Link className="water-upload-brand" to="/">
          <CgDrop aria-hidden="true" />
          <span>Water Analysis AI</span>
        </Link>
        <span className="water-upload-model">
          RESNET · {result?.model || "CNNv2.5"}
        </span>
      </header>

      <section className="water-upload-content">
        <div className="water-upload-intro">
          <span className="water-upload-eyebrow">COMPUTER VISION ANALYSIS</span>
          <h1>Water AI</h1>
          <p>
             
            The Water AI uses a Residual Network model
            It analyzes a body of water, and determines whether it is potable or not. Included is a feature map that shows the model's visual analysis of the water sample. 
            Upload a water image or video to see the model's analysis and a confidence score for whether the water is potable.
          </p>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="water-upload-frames">
            <section className="water-upload-frame">
              <div className="water-upload-frame-heading">
                <span><i className="water-upload-dot" />Video Input</span>
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
                <span><i className="water-upload-dot water-upload-dot-ai" />What the Model Sees</span>
                <span>MODEL ACTIVATIONS</span>
              </div>
              <div className="water-upload-model-output">
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
                {result?.feature_activations && (
                  <div className="water-upload-activations">
                    <div className="water-upload-activations-heading">
                      <span>POOLED LAYER 4 ACTIVATIONS</span>
                      <span>9 CHANNELS</span>
                    </div>
                    <div
                      className="water-upload-activation-chart"
                      role="list"
                      aria-label="Relative activation strength for nine ResNet channels"
                    >
                      {result.feature_activations.map((item) => (
                        <div
                          className="water-upload-activation-row"
                          key={item.channel}
                          role="listitem"
                          aria-label={`ResNet channel ${item.channel}: pooled activation ${item.activation}, ${item.relative}% of strongest channel`}
                        >
                          <span className="water-upload-activation-label">
                            CH {String(item.channel).padStart(3, "0")}
                          </span>
                          <span className="water-upload-activation-track">
                            <span
                              className="water-upload-activation-bar"
                              style={{ width: `${item.relative}%` }}
                            />
                          </span>
                          <span className="water-upload-activation-value">
                            {item.relative.toFixed(0)}%
                          </span>
                        </div>
                      ))}
                    </div>
                    <p className="water-upload-activation-note">
                      Relative channel activation strength, not feature importance.
                    </p>
                  </div>
                )}
              </div>
            </section>

            <section className="water-upload-frame water-upload-saliency-frame">
              <div className="water-upload-frame-heading">
                <span><i className="water-upload-dot water-upload-dot-ai" />Class Saliency</span>
                <span>3 × 3 CLASS-SPECIFIC TILES</span>
              </div>
              <div className="water-upload-saliency-content">
                {result?.class_saliency_tiles ? (
                  <>
                    <img
                      src={result.class_saliency_tiles}
                      alt="Nine channel-specific saliency overlays on the source frame; green shows regions supporting potable and red shows regions supporting not potable"
                    />
                    <div className="water-upload-saliency-legend">
                      <span><i className="water-upload-saliency-green" /> Potable</span>
                      <span><i className="water-upload-saliency-red" /> Not potable</span>
                    </div>
                    <p>
                      Class-specific saliency from the model&apos;s output gradients;
                      visual explanation, not a water-safety measurement.
                    </p>
                  </>
                ) : (
                  <div className="water-upload-saliency-empty">
                    {isLoading
                      ? "Calculating class-specific feature overlays…"
                      : "Green and red class-specific feature overlays will appear here."}
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
                <>
                  <span className="water-upload-spinner" />
                  {isVideo ? "Analyzing video…" : "Analyzing"}
                </>
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
