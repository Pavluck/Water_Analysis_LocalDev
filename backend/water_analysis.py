"""
✧ NP ~ Backend | Supports Video Upload Webpage ~ ✧ 

Flask API - Water AI - Potability Analysis
Water AI using the ResNet18 architecture from Pytorch framework
endpoints for analyzing water, retrieving overlay videos, and managing temporary video files

Takes a payload from frontend: uploaded image/video, classifies as potable or not potable, generates visuals for the model's predictions  
Returns a JSON response with the results to view on the frontend
"""

# ~~~~ Necessary Imports ~~~~
# Media Imports
import base64
import os
import tempfile
import time
from pathlib import Path
from uuid import uuid4
# Logic and DL Imports
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from flask import Blueprint, current_app, jsonify, request, send_file, url_for
from PIL import Image, UnidentifiedImageError
from trainingvideo import WaterCNN, validation_transforms

# Model Setup
water_analysis = Blueprint("water_analysis", __name__)
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
VIDEO_OUTPUT_DIR = Path(tempfile.gettempdir()) / "water-analysis-overlays"
VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_OUTPUTS = {}
VIDEO_OUTPUT_TTL_SECONDS = 3600
MAX_VIDEO_SECONDS = 120
FRAME_SAMPLE_SECONDS = 1
MAX_VIDEO_SAMPLES = 5
MAX_OVERLAY_DIMENSION = 960
FEATURE_CHANNEL_COUNT = 9
FEATURE_TILE_GRID_SIZE = 3
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL = None
_MODEL_PATH = None

# Sanity Check 
def _model_path():
    """
    Assumes model is located in the root directory - matches name "CNNv2.5.pth". 
    Raises Error if not found (or renamed)
    """
    path = PROJECT_DIR / "CNNv2.5.pth"

    if not path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {path}"
        )

    return path


def _load_model():
  """
  if model exists - it can be used 
  """
    global _MODEL, _MODEL_PATH
    path = _model_path()
    if _MODEL is not None and path == _MODEL_PATH:
        return _MODEL, path

    model = WaterCNN(num_classes=2, pretrained=False).to(DEVICE)
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Model checkpoint has an unsupported format: {path.name}")

    try:
        model.load_state_dict(checkpoint)
    except RuntimeError as error:
        raise ValueError(
            f"Model checkpoint is incompatible with the WaterCNN ResNet18 "
            f"architecture: {path.name}"
        ) from error

    model.eval()
    _MODEL = model
    _MODEL_PATH = path
    return model, path


def _activation_layer(model):
    backbone = getattr(model, "backbone", model)
    return backbone.layer4


def _pooled_activation_scores(activations):
  """
  Feature / Heatmap of distinct features 
  """
    pooled = torch.nn.AdaptiveAvgPool2d(1)(activations).flatten()
    scores, channels = pooled.topk(min(FEATURE_CHANNEL_COUNT, pooled.numel()))
    scores = scores.cpu().tolist()
    channels = channels.cpu().tolist()
    max_score = max(scores, default=0.0)
    return [
        {
            "channel": channel + 1,
            "activation": round(score, 6),
            "relative": round(score / max_score * 100, 1) if max_score else 0.0,
        }
        for channel, score in zip(channels, scores)
    ]


def _class_saliency_tiles_data_url(
    activations, gradients, frame_rgb
):
  """
  This is under construction, idea is that tiles populate, and fill in with more focused potable/non-potable visual traits
  """
    frame_height, frame_width = frame_rgb.shape[:2]
    scale = min(256 / frame_width, 256 / frame_height)
    tile_width = max(1, round(frame_width * scale))
    tile_height = max(1, round(frame_height * scale))
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    source_tile = cv2.resize(frame_bgr, (tile_width, tile_height))

    channel_contributions = [
        (activations.detach() * gradient).clamp_min(0)
        for gradient in gradients
    ]
    potable_scores = channel_contributions[0].mean(dim=(0, 2, 3))
    not_potable_scores = channel_contributions[1].mean(dim=(0, 2, 3))
    _, potable_channels = potable_scores.topk(5)
    _, not_potable_channels = not_potable_scores.topk(4)
    tiles = [
        (0, int(channel))
        for channel in potable_channels.cpu().tolist()
    ] + [
        (1, int(channel))
        for channel in not_potable_channels.cpu().tolist()
    ]
    gap = 8
    label_height = 24
    tile_canvas_height = tile_height + label_height
    legend_height = 24
    grid_width = (
        FEATURE_TILE_GRID_SIZE * tile_width + (FEATURE_TILE_GRID_SIZE + 1) * gap
    )
    grid_height = (
        legend_height
        + FEATURE_TILE_GRID_SIZE * tile_canvas_height
        + (FEATURE_TILE_GRID_SIZE + 1) * gap
    )
    grid = np.full((grid_height, grid_width, 3), (16, 27, 38), dtype=np.uint8)
    cv2.putText(
        grid,
        "GREEN: POTABLE   |   RED: NOT POTABLE",
        (gap, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (226, 244, 248),
        1,
        cv2.LINE_AA,
    )

    for index, (class_index, channel) in enumerate(tiles):
        contribution = channel_contributions[class_index][0, channel]
        values = F.interpolate(
            contribution[None, None],
            size=(tile_height, tile_width),
            mode="bilinear",
            align_corners=False,
        )[0, 0].cpu().numpy()
        values /= values.max() + 1e-8
        strength = values * 0.56
        color_overlay = np.zeros_like(source_tile)
        color_channel = 1 if class_index == 0 else 2
        color_overlay[:, :, color_channel] = np.uint8(strength * 255)
        opacity = strength[:, :, None]
        tile = np.uint8(
            source_tile.astype(np.float32) * (1 - opacity)
            + color_overlay.astype(np.float32)
        )
        row, column = divmod(index, FEATURE_TILE_GRID_SIZE)
        top = legend_height + gap + row * (tile_canvas_height + gap)
        left = gap + column * (tile_width + gap)
        grid[top : top + label_height, left : left + tile_width] = (8, 18, 27)
        cv2.putText(
            grid,
            f"{'POTABLE' if class_index == 0 else 'NOT POTABLE'}  CH {channel + 1:03d}",
            (left + 6, top + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (112, 242, 157) if class_index == 0 else (112, 112, 255),
            1,
            cv2.LINE_AA,
        )
        grid[
            top + label_height : top + tile_canvas_height,
            left : left + tile_width,
        ] = tile

    encoded, buffer = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not encoded:
        raise RuntimeError("The feature-map overlay tiles could not be encoded.")
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")


def _predict_frame(model, frame_rgb):
  """
  The main course of our meal, is the water potable?
  """
    image = Image.fromarray(frame_rgb).convert("RGB")
    tensor = validation_transforms(image).unsqueeze(0).to(DEVICE)
    activation = {}
    handle = _activation_layer(model).register_forward_hook(
        lambda *args: activation.update(feature_map=args[-1])
    )
    try:
        with torch.enable_grad():
            logits = model(tensor)
            probabilities = torch.softmax(logits.detach(), dim=1)[0].cpu().numpy()
            activations = activation["feature_map"]
            activation_scores = _pooled_activation_scores(activations.detach())
            gradients = [
                torch.autograd.grad(
                    logits[0, class_index],
                    activations,
                    retain_graph=class_index == 0,
                )[0].detach()
                for class_index in (0, 1)
            ]
    finally:
      # cleanup on aisle 5
        handle.remove()

    activations = activations.detach()
    class_saliency_tiles = _class_saliency_tiles_data_url(
        activations, gradients, frame_rgb
    )
    feature_map = activations.mean(dim=1, keepdim=True)
    feature_map = F.interpolate(
        feature_map,
        size=frame_rgb.shape[:2],
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    feature_map = feature_map.cpu().numpy()
    feature_map -= feature_map.min()
    feature_map /= feature_map.max() + 1e-8
    heatmap = cv2.applyColorMap(
        np.uint8(feature_map * 255), cv2.COLORMAP_JET
    )
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(frame_bgr, 0.58, heatmap, 0.42, 0)
    return probabilities, overlay, activation_scores, class_saliency_tiles


def _clean_expired_videos():
  """
  Hiring the janitor 
  """
    now = time.time()
    expired = [
        asset_id
        for asset_id, (_, expires_at) in VIDEO_OUTPUTS.items()
        if expires_at <= now
    ]
    for asset_id in expired:
        path, _ = VIDEO_OUTPUTS[asset_id]
        try:
            path.unlink(missing_ok=True)
        except PermissionError:
            VIDEO_OUTPUTS[asset_id] = (path, now + 60)
        else:
            VIDEO_OUTPUTS.pop(asset_id)


def _result(probabilities, model_path):
  """
  Level of confidence for both potable and non potable predictions
  """
    class_index = int(np.argmax(probabilities))
    return {
        "potable": class_index == 0,
        "label": "Potable" if class_index == 0 else "Not potable",
        "confidence": float(probabilities[class_index]),
        "model": model_path.name,
    }


def _analyze_image(path, model, model_path):
  """
  Sanity check the frames
  Package up the traits
  """
    try:
        with Image.open(path) as source:
            source.load()
            frame_rgb = np.asarray(source.convert("RGB"))
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("The uploaded image could not be decoded.") from error

    (
        probabilities,
        overlay,
        activation_scores,
        class_saliency_tiles,
    ) = _predict_frame(model, frame_rgb)
    encoded, buffer = cv2.imencode(".png", overlay)
    if not encoded:
        raise RuntimeError("The image feature-map overlay could not be encoded.")

    result = _result(probabilities, model_path)
    result["overlay_image"] = (
        "data:image/png;base64,"
        + base64.b64encode(buffer.tobytes()).decode("ascii")
    )
    result["feature_activations"] = activation_scores
    result["class_saliency_tiles"] = class_saliency_tiles
    return result


def _analyze_video(path, model, model_path):
  """
  Recieving input
  """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("The uploaded video could not be opened.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError("The uploaded video has invalid dimensions or frame rate.")
    if duration > MAX_VIDEO_SECONDS:
        capture.release()
        raise ValueError("Videos must be 2 minutes or shorter.")

    asset_id = uuid4().hex
    output_path = VIDEO_OUTPUT_DIR / f"{asset_id}.webm"
    output_scale = min(1.0, MAX_OVERLAY_DIMENSION / max(width, height))
    output_width = max(2, int(width * output_scale) // 2 * 2)
    output_height = max(2, int(height * output_scale) // 2 * 2)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"VP80"),
        fps,
        (output_width, output_height),
    )
    if not writer.isOpened():
        capture.release()
        output_path.unlink(missing_ok=True)
        raise RuntimeError("The server could not create the overlay video.")

    probabilities_by_frame = []
    last_overlay = None
    sample_interval = max(
        int(round(fps * FRAME_SAMPLE_SECONDS)),
        int(np.ceil(frame_count / MAX_VIDEO_SAMPLES)),
    )
    activation_scores = None
    class_saliency_tiles = None
    frame_number = 0
    try:
        while True:
            if frame_number / fps >= MAX_VIDEO_SECONDS:
                raise ValueError("Videos must be 2 minutes or shorter.")
            success, frame_bgr = capture.read()
            if not success:
                break
            if frame_number % sample_interval == 0 or last_overlay is None:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                (
                    probabilities,
                    last_overlay,
                    frame_scores,
                    frame_saliency_tiles,
                ) = _predict_frame(model, frame_rgb)
                probabilities_by_frame.append(probabilities)
                if activation_scores is None:
                    activation_scores = frame_scores
                    class_saliency_tiles = frame_saliency_tiles
            if last_overlay.shape[:2] != (output_height, output_width):
                last_overlay = cv2.resize(
                    last_overlay, (output_width, output_height)
                )
            writer.write(last_overlay)
            frame_number += 1
    except Exception:
        capture.release()
        writer.release()
        output_path.unlink(missing_ok=True)
        raise
    else:
        capture.release()
        writer.release()

    if not probabilities_by_frame:
        output_path.unlink(missing_ok=True)
        raise ValueError("The uploaded video contains no readable frames.")

    VIDEO_OUTPUTS[asset_id] = (
        output_path,
        time.time() + VIDEO_OUTPUT_TTL_SECONDS,
    )
    result = _result(np.mean(probabilities_by_frame, axis=0), model_path)
    result["overlay_url"] = url_for(
        ".get_overlay_video", asset_id=asset_id, _external=True
    )
    result["frames_analyzed"] = len(probabilities_by_frame)
    result["feature_activations"] = activation_scores
    result["class_saliency_tiles"] = class_saliency_tiles
    return result


@water_analysis.route("SECRET", methods=["POST"])
def analyze_water():
  """
  Main API loop
  """
    _clean_expired_videos()
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return jsonify(error="Choose an image or video to analyze."), 400

    extension = Path(uploaded.filename).suffix.lower()
    if extension not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
        return jsonify(error="This image or video format is not supported."), 400

    with tempfile.NamedTemporaryFile(
        suffix=extension, prefix="water-upload-", delete=False
    ) as temporary_file:
        input_path = Path(temporary_file.name)

    try:
        uploaded.save(input_path)
        try:
            model, model_path = _load_model()
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            current_app.logger.exception("Water AI model could not be loaded.")
            return jsonify(error=str(error)), 503

        if extension in IMAGE_EXTENSIONS:
            return jsonify(_analyze_image(input_path, model, model_path))
        return jsonify(_analyze_video(input_path, model, model_path))
    except ValueError as error:
        return jsonify(error=str(error)), 400
    except RuntimeError as error:
        current_app.logger.exception("Water AI media analysis failed.")
        return jsonify(error="The uploaded media could not be analyzed."), 500
    except Exception:
        current_app.logger.exception("Unexpected Water AI upload error.")
        return jsonify(error="An unexpected error occurred while analyzing the upload."), 500
    finally:
        input_path.unlink(missing_ok=True)


@water_analysis.route("SECRET", methods=["GET"])   # don't read this: /api/analyze-water/overlays/<asset_id>
def get_overlay_video(asset_id):
  """
  Edge cases 
  """
    _clean_expired_videos()
    video = VIDEO_OUTPUTS.get(asset_id)
    if video is None or not video[0].is_file():
        return jsonify(error="This overlay video has expired. Analyze the file again."), 404
    return send_file(video[0], mimetype="video/webm", conditional=True)
# ~~~~~~ EOF ~~~~~~~~
