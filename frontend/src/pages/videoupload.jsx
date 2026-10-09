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

