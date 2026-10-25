"""
NP ˚❁༉‧ Oct 26 

Fine-tunes CNNv2.5 with more water images and videos
Calls the June Training module for augmentation and training routines and calls the WaterCNN for the data and model definition.
Still uses ResNet18
"""

# ~~~~~ Necessary Imports ~~~~~
import csv
import tempfile
from pathlib import Path
# Logic Imports
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
# Model Imports
import JuneTraining
from trainingvideo import WaterCNN

# ~~ Global Constants ~~
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "additional_data"    # not in the repo (just make a folder next to the backend folder)
INPUTMODEL = ROOT / "CNNv2.5.pth"  # old 
MODEL = ROOT / "CNNv2.6.pth"       # new
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Similar parameters from June's training, but smaller batch size and fewer epochs for fine-tuning
BATCH_SIZE = 16
EPOCHS = 5
FREEZE_BACKBONE_EPOCHS = 1
LEARNING_RATE = 1e-4
BACKBONE_LR_FACTOR = 0.1
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0
VIDEO_SAMPLE_SECONDS = 1   # consider frames every 10 seconds to prepare for live streams
MAX_FRAMES_PER_VIDEO = 30   
# image prep
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi", ".mkv"}


def _addtargets(path):
    """
    Helper function for my edge case data
    Files beginning with "n" are labeled non-potable; files beginning with "p"
    are labeled potable. Videos are sampled at one frame per second, up to 30
    frames per video. 
    """
    prefix = path.stem[:1].lower()
    # Looks for prefix; p for potable, etc so it can be reused as more images and videos are added 
    if prefix == "n":
        return "not_potable"
    if prefix == "p":
        return "potable"
    raise ValueError(
        f"Additional-data filename must start with 'n' or 'p': {path.name}"
    )


def _extractframes(video_path, label, frame_dir):
    """
    Allows videos for training
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Could not open video: {video_path}")

    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frame_count <= 0:
            raise ValueError(f"Oops - Could not read metadata: {video_path}")

        frame_interval = max(1, round(fps * VIDEO_SAMPLE_SECONDS))
        samples = []
        for frame_index in range(
            0, frame_count, frame_interval
        ):
            if len(samples) >= MAX_FRAMES_PER_VIDEO:
                break

            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success:
                raise OSError(
                    f"Could not decode frame {frame_index} from {video_path}"
                )

            frame_path = frame_dir / (
                f"{video_path.stem}_{frame_index:08d}.jpg"
            )
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb_frame).save(frame_path, format="JPEG")
            samples.append((frame_path.resolve(), label))

        if not samples:
            raise ValueError(f"Error. Check size - No frames could be extracted from: {video_path}")
        return samples
    finally:
        capture.release()


def _prepare_dataset(frame_dir, annotations_path):
    """
    Assumes local directory
    """
    if not DATA.is_dir():
        raise FileNotFoundError(
            f"Additional data directory not found: {DATA}"
        )

    samples = []
    for media_path in sorted(DATA.iterdir()):
        if not media_path.is_file():
            continue

        extension = media_path.suffix.lower()
        if extension not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
            continue

        label = _addtargets(media_path)
        if extension in IMAGE_EXTENSIONS:
            samples.append((media_path.resolve(), label))
        else:
            samples.extend(_extractframes(media_path, label, frame_dir))

    labels = {label for _, label in samples}
    if not samples:
        raise ValueError(f"Oopsie ~ Something is wrong with the images or videos found in {DATA}")
    if labels != {"potable", "not_potable"}:
        raise ValueError(
            "Prefix error - can't find (p) or (n) image/video."
        )

    with annotations_path.open("w", newline="", encoding="utf-8") as annotations:
        writer = csv.writer(annotations)
        writer.writerow(("image", "class"))
        writer.writerows(samples)

    return samples


def _optimizer_for(model):
  """
  Backbone and frontbone for propergations
  ADAM optimizer 
  """
    backbone_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("backbone.") and not name.startswith("backbone.fc.")
    ]
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not (name.startswith("backbone.") and not name.startswith("backbone.fc."))
    ]
    return optim.Adam(
        [
            {
                "params": backbone_parameters,
                "lr": LEARNING_RATE * BACKBONE_LR_FACTOR,
                "weight_decay": WEIGHT_DECAY,
            },
            {
                "params": head_parameters,
                "lr": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
            },
        ]
    )


def main():
    """
    Resuing checkpoint + June's logic 
    """
    if not INPUTMODEL.is_file():
        raise FileNotFoundError(f"Oops - model not found: {INPUTMODEL}")

    with tempfile.TemporaryDirectory(prefix="oct_training_") as temporary_dir:
        temporary_path = Path(temporary_dir)
        frame_dir = temporary_path / "frames"
        frame_dir.mkdir()
        annotations_path = temporary_path / "annotations.csv"
        samples = _prepare_dataset(frame_dir, annotations_path)

        dataset = JuneTraining.DataPrep(
            str(DATA),
            str(annotations_path),
            transform=JuneTraining.train_transforms,
        )
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
        )

        model = WaterCNN(num_classes=2, pretrained=False).to(DEVICE)
        checkpoint = torch.load(
            INPUTMODEL,
            map_location=DEVICE,
            weights_only=True,
        )
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if not isinstance(checkpoint, dict):
            raise ValueError(
                f"Unsupported checkpoint format: {INPUTMODEL}"
            )
        model.load_state_dict(checkpoint)

        for name, parameter in model.named_parameters():
            if name.startswith("backbone.") and not name.startswith("backbone.fc."):
                parameter.requires_grad = False

        optimizer = _optimizer_for(model)
        criterion = nn.CrossEntropyLoss()
        print(
            f"Training... Fine-tuning {INPUTMODEL.name} with {len(samples)} "
            f"additional data from {DEVICE}."
        )

        for epoch in range(EPOCHS):
            if epoch == FREEZE_BACKBONE_EPOCHS:
                for parameter in model.parameters():
                    parameter.requires_grad = True
                optimizer = _optimizer_for(model)

            loss, accuracy = JuneTraining.train_epoch(
                model,
                loader,
                criterion,
                optimizer,
                DEVICE,
                clip_norm=GRADIENT_CLIP_NORM,
            )
            print(
                f"Current Epoch {epoch + 1}/{EPOCHS}: "
                f"loss={loss:.4f}, accuracy={accuracy:.2f}%"
            )

        torch.save(model.state_dict(), MODEL)
        print(f"Success ✧ Saved model: {MODEL}")


if __name__ == "__main__":
    """
    Pipeline
    """
    main()
