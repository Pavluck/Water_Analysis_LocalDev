// ~~~~~~~~~ Water AI Video Page ~~~~~~~~

import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { CgDrop } from "react-icons/cg";
import "./videoupload.css";

// ~~~~~~~~~~~ API Logic ~~~~~~~~~~~~
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

