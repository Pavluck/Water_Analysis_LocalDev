"""
❀༉‧˚ aka Novtraining.py - NP
CNN model for water potability classification using a 20-layer residual (Resnet) backbone.
Calls the June Training.py module for data augmentation, data loading, and
validation routines. Fine-tunes with additional water images and videos.
Videos are sampled at one frame per second, up to 30
frames per video. 
"""

# ~~~~~ Necessary Imports ~~~~~
import csv
import json
import random
import tempfile
from collections import Counter
from pathlib import Path
# Logic Imports
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
from PIL import Image
# Model Imports
import JuneTraining  # aka Spring Training (retrained spring in summer)

# ~~ Global Constants ~~
# portability items 
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "additional_data"
MODEL = ROOT / "CNNv2.7.pth"
HISTORY = ROOT / "CNNv2.7_history.json"
CLASS_TO_IDX = {"potable": 0, "not_potable": 1}
IDX_TO_CLASS = {value: key for key, value in CLASS_TO_IDX.items()}
LABEL_TO_IDX = {
    **CLASS_TO_IDX,
    "nonpotable": 1,
    "clear": 0,
    "murky": 1,
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Similar parameters from June's training, but smaller batch size for fine-tuning
BATCH_SIZE = 16
EPOCHS = 20
FREEZE_BACKBONE_EPOCHS = 1
LEARNING_RATE = 1e-4
BACKBONE_LR_FACTOR = 0.1
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0
VIDEO_SAMPLE_SECONDS = 1   # consider frames every 10 seconds to prepare for live streams
MAX_FRAMES_PER_VIDEO = 30   
EDGE_VALIDATION_FRACTION = 0.2
RANDOM_SEED = 42
# image prep
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi", ".mkv"}

def _addtargets(path):
    """
    Helper function for my edge case data
    Looks for prefix; like p for potable so it can be reused as more images and videos are added 
    """
    prefix = path.stem[:1].lower()
    if prefix == "n":
        return "not_potable"
    if prefix == "p":
        return "potable"
    raise ValueError(
        f"Additional-data filename must start with 'n' or 'p': {path.name}"
    )


class BasicBlock(nn.Module):
    """The basic residual block for ResNet20 architecture.
    Based on the original ResNet, blocks consists of two convolutional layers with a skip connection.
    The skip connection allows the input to bypass the convolutional layers, enabling better gradient flow and mitigating the vanishing gradient problem.
    The block uses batch normalization and ReLU activation after each convolutional layer."""

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, inputs):
        """
        Forward propagation through the residual block. The input is passed through two convolutional layers, and the output is added to the shortcut connection before applying the final ReLU activation.
        """
        residual = self.shortcut(inputs)
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        return self.relu(outputs + residual)


class ResNet20(nn.Module):
    """ImageNet-sized 20-layer ResNet with 3 stages of 3 residual blocks."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 64, blocks=3, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=3, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=3, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _make_layer(in_channels, out_channels, blocks, stride):
        layers = [BasicBlock(in_channels, out_channels, stride)]
        layers.extend(
            BasicBlock(out_channels, out_channels) for _ in range(blocks - 1)
        )
        return nn.Sequential(*layers)

    def forward(self, inputs):
      """
      Part 2
      Instead of block forward, this defines the forward pass of the entire ResNet20 model. 
      The input is passed through the stem, followed by three stages of residual blocks, and finally through the (avg) pooling layer 
      and a fully connected layer to produce the potable or nonpotable.
      """
        outputs = self.stem(inputs)
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.pool(outputs).flatten(1)
        return self.fc(outputs)


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
        frame_index = 0
        next_sample_frame = 0
        while frame_index < frame_count:
            if len(samples) >= MAX_FRAMES_PER_VIDEO:
                break

            success, frame = capture.read()
            if not success:
                raise OSError(
                    f"Could not decode frame {frame_index} from {video_path}"
                )

            if frame_index >= next_sample_frame:
                frame_path = frame_dir / (
                    f"{video_path.stem}_{frame_index:08d}.jpg"
                )
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                Image.fromarray(rgb_frame).save(frame_path, format="JPEG")
                samples.append((frame_path.resolve(), label))
                next_sample_frame += frame_interval

            frame_index += 1

        if not samples:
            raise ValueError(f"Error. Check size - No frames could be extracted from: {video_path}")
        return samples
    finally:
        capture.release()


def _write_annotations(path, samples):
    with path.open("w", newline="", encoding="utf-8") as annotations:
        writer = csv.writer(annotations)
        writer.writerow(("image", "class"))
        writer.writerows(samples)


def _prepare_dataset(frame_dir, train_annotations_path, validation_annotations_path):
    """
    Assumes local directory
    """
    if not DATA.is_dir():
        raise FileNotFoundError(
            f"Additional data directory not found: {DATA}"
        )

    media_paths = [
        path for path in sorted(DATA.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
    ]
    if not media_paths:
        raise ValueError(f"Oopsie ~ Something is wrong with the images or videos found in {DATA}")

    groups = sorted({path.stem for path in media_paths})
    if len(groups) > 1:
        validation_count = max(1, round(len(groups) * EDGE_VALIDATION_FRACTION))
        validation_groups = set(random.Random(RANDOM_SEED).sample(groups, validation_count))
    else:
        validation_groups = set()

    train_samples = []
    validation_samples = []
    for media_path in media_paths:
        label = _addtargets(media_path)
        target = validation_samples if media_path.stem in validation_groups else train_samples
        if media_path.suffix.lower() in IMAGE_EXTENSIONS:
            target.append((media_path.resolve(), label))
        else:
            target.extend(_extractframes(media_path, label, frame_dir))

    labels = {label for _, label in train_samples + validation_samples}
    if labels != set(CLASS_TO_IDX):
        raise ValueError(
            "Oopsies ~ Prefix error : can't find (p) or (n) image/video."
        )

    _write_annotations(train_annotations_path, train_samples)
    _write_annotations(validation_annotations_path, validation_samples)
    return train_samples, validation_samples


def _set_backbone_frozen(model, frozen):
    for name, parameter in model.named_parameters():
        if not name.startswith("fc."):
            parameter.requires_grad = not frozen


def _set_backbone_mode(model, train):
    model.stem.train(train)
    model.layer1.train(train)
    model.layer2.train(train)
    model.layer3.train(train)


def _class_counts(dataset):
    counts = Counter()
    label_column = next(
        (
            column for column in ("class", "label", "potability", "clarity")
            if column in dataset.annotations.columns
        ),
        dataset.annotations.columns[1],
    )
    for label_value in dataset.annotations[label_column]:
        label = str(label_value).strip().lower()
        if label not in LABEL_TO_IDX:
            raise ValueError(f"Unknown class label in annotations: {label}")
        counts[LABEL_TO_IDX[label]] += 1
    return {IDX_TO_CLASS[index]: counts[index] for index in range(len(CLASS_TO_IDX))}


def _train_epoch(model, loader, criterion, optimizer, frozen_backbone):
    model.train()
    if frozen_backbone:
        _set_backbone_mode(model, False)
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), 100.0 * correct / total


def _validation_metrics(model, loader):
    confusion = [[0 for _ in CLASS_TO_IDX] for _ in CLASS_TO_IDX]
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(DEVICE))
            predictions = outputs.argmax(dim=1).cpu()
            for actual, predicted in zip(labels.tolist(), predictions.tolist()):
                confusion[actual][predicted] += 1

    total = sum(sum(row) for row in confusion)
    class_counts = [sum(row) for row in confusion]
    majority_accuracy = (
        100.0 * max(class_counts) / total if total else 0.0
    )
    return confusion, majority_accuracy


def _optimizer_for(model):
    backbone_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("fc.") and parameter.requires_grad
    ]
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("fc.") and parameter.requires_grad
    ]
    parameter_groups = []
    if backbone_parameters:
        parameter_groups.append(
            {
                "params": backbone_parameters,
                "lr": LEARNING_RATE * BACKBONE_LR_FACTOR,
                "weight_decay": WEIGHT_DECAY,
            }
        )
    if head_parameters:
        parameter_groups.append(
            {
                "params": head_parameters,
                "lr": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
            }
        )
    return optim.Adam(parameter_groups)


def main():
    """
    Resuing June's logic 
    """
    with tempfile.TemporaryDirectory(prefix="nov_training_") as temporary_dir:   # or ResNet20
        temporary_path = Path(temporary_dir)
        frame_dir = temporary_path / "frames"
        frame_dir.mkdir()
        edge_train_annotations = temporary_path / "edge_train_annotations.csv"
        edge_validation_annotations = (
            temporary_path / "edge_validation_annotations.csv"
        )
        edge_train_samples, edge_validation_samples = _prepare_dataset(
            frame_dir,
            edge_train_annotations,
            edge_validation_annotations,
        )

        roboflow_dataset = JuneTraining.DataPrep(
            JuneTraining.Training_Images,
            JuneTraining.Training_Labels,
            transform=JuneTraining.train_transforms,
        )
        edge_case_dataset = JuneTraining.DataPrep(
            str(DATA),
            str(edge_train_annotations),
            transform=JuneTraining.train_transforms,
        )
        dataset = ConcatDataset((roboflow_dataset, edge_case_dataset))
        roboflow_validation_dataset = JuneTraining.DataPrep(
            JuneTraining.Test_Images,
            JuneTraining.Test_Labels,
            transform=JuneTraining.val_transforms,
        )
        edge_validation_dataset = JuneTraining.DataPrep(
            str(DATA),
            str(edge_validation_annotations),
            transform=JuneTraining.val_transforms,
        )
        validation_dataset = ConcatDataset(
            (roboflow_validation_dataset, edge_validation_dataset)
        )
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

        model = ResNet20(num_classes=len(CLASS_TO_IDX)).to(DEVICE)
        checkpoint = {}
        loaded_checkpoint = False
        history = {
            "train_losses": [],
            "train_accs": [],
            "val_losses": [],
            "val_accs": [],
        }
        start_epoch = 0
        # ~~~~ Checkpoint logic ~~~~
        if MODEL.is_file():
            checkpoint = torch.load(
                MODEL,
                map_location=DEVICE,
                weights_only=True,
            )
            if not isinstance(checkpoint, dict):
                raise ValueError(f"Oopsie Daisy! Error- Unsupported checkpoint format: {MODEL}")
            state_dict = checkpoint.get(
                "model_state_dict",
                checkpoint.get("state_dict", checkpoint),
            )
            try:
                model.load_state_dict(state_dict)
                loaded_checkpoint = "model_state_dict" in checkpoint or "state_dict" in checkpoint
                start_epoch = int(checkpoint.get("epoch", 0))
                saved_history = checkpoint.get("history")
                if isinstance(saved_history, dict):
                    history = saved_history
                print(f"Continuing ResNet20 training from {MODEL.name}.")
            except RuntimeError:
                print(
                    f"{MODEL.name} is not compatible with ResNet20; "
                    "initializing from scratch."
                )
        else:
            print("No ResNet20 checkpoint exists; starting the model from scratch.")

        freeze_backbone = (
            loaded_checkpoint and start_epoch == 0 and FREEZE_BACKBONE_EPOCHS > 0
        )
        _set_backbone_frozen(model, freeze_backbone)

        optimizer = _optimizer_for(model)
        if loaded_checkpoint and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        criterion = nn.CrossEntropyLoss()
        print(
            f"Training ResNet20 with {len(roboflow_dataset)} Roboflow samples, "
            f"{len(edge_train_samples)} edge-case training samples, and "
            f"{len(edge_validation_samples)} edge-case validation samples "
            f"from {DEVICE}."
        )

        train_counts = _class_counts(roboflow_dataset)
        edge_train_counts = _class_counts(edge_case_dataset)
        validation_counts = _class_counts(roboflow_validation_dataset)
        edge_validation_counts = _class_counts(edge_validation_dataset)
        print(f"Roboflow train class counts: {train_counts}")
        print(f"Edge-case train class counts: {edge_train_counts}")
        print(f"Validation class counts: {validation_counts}")
        print(f"Edge-case validation class counts: {edge_validation_counts}")

        best_validation_accuracy = checkpoint.get("best_validation_accuracy", -1.0)
        best_validation_loss = checkpoint.get("best_validation_loss", float("inf"))
        last_confusion = checkpoint.get("validation_confusion_matrix")
        last_majority_accuracy = checkpoint.get(
            "validation_majority_accuracy", 0.0
        )
        for epoch in range(start_epoch, EPOCHS):
            if freeze_backbone and epoch == start_epoch + FREEZE_BACKBONE_EPOCHS:
                _set_backbone_frozen(model, False)
                optimizer = _optimizer_for(model)

            currently_frozen = freeze_backbone and (
                epoch < start_epoch + FREEZE_BACKBONE_EPOCHS
            )
            loss, accuracy = _train_epoch(
                model,
                loader,
                criterion,
                optimizer,
                currently_frozen,
            )
            validation_loss, validation_accuracy = JuneTraining.validate(
                model,
                validation_loader,
                criterion,
                DEVICE,
            )
            history["train_losses"].append(loss)
            history["train_accs"].append(accuracy)
            history["val_losses"].append(validation_loss)
            history["val_accs"].append(validation_accuracy)
            confusion, majority_accuracy = _validation_metrics(model, validation_loader)
            last_confusion = confusion
            last_majority_accuracy = majority_accuracy
            print(
                f"Current Epoch {epoch + 1}/{EPOCHS}: "
                f"loss={loss:.4f}, accuracy={accuracy:.2f}% | "
                f"validation_loss={validation_loss:.4f}, "
                f"validation_accuracy={validation_accuracy:.2f}% | "
                f"majority_baseline={majority_accuracy:.2f}%"
            )
            print(f"Validation confusion matrix: {confusion}")

            is_best = (
                validation_accuracy > best_validation_accuracy
                or (
                    validation_accuracy == best_validation_accuracy
                    and validation_loss < best_validation_loss
                )
            )
            if is_best:
                best_validation_accuracy = validation_accuracy
                best_validation_loss = validation_loss
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch + 1,
                        "best_validation_accuracy": best_validation_accuracy,
                        "best_validation_loss": best_validation_loss,
                        "class_to_idx": CLASS_TO_IDX,
                        "class_counts": {
                            "train_roboflow": train_counts,
                            "train_edge_cases": edge_train_counts,
                            "validation": validation_counts,
                            "validation_edge_cases": edge_validation_counts,
                        },
                        "validation_confusion_matrix": confusion,
                        "validation_majority_accuracy": majority_accuracy,
                        "config": {
                            "batch_size": BATCH_SIZE,
                            "epochs": EPOCHS,
                            "learning_rate": LEARNING_RATE,
                            "backbone_lr_factor": BACKBONE_LR_FACTOR,
                            "weight_decay": WEIGHT_DECAY,
                            "architecture": "ResNet20",
                        },
                        "history": history,
                    },
                    MODEL,
                )
                print(f"Saved best validation checkpoint: {MODEL}")

        if best_validation_accuracy < 0:
            raise RuntimeError("No validation checkpoint was produced.")
        with HISTORY.open("w", encoding="utf-8") as history_file:
            json.dump(
                {
                    **history,
                    "class_to_idx": CLASS_TO_IDX,
                    "class_counts": {
                        "train_roboflow": train_counts,
                        "train_edge_cases": edge_train_counts,
                        "validation": validation_counts,
                        "validation_edge_cases": edge_validation_counts,
                    },
                    "validation_confusion_matrix": last_confusion,
                    "validation_majority_accuracy": last_majority_accuracy,
                    "best_validation_accuracy": best_validation_accuracy,
                    "best_validation_loss": best_validation_loss,
                },
                history_file,
                indent=2,
            )
        print(f"Success ✧ Best model saved to: {MODEL}")
        print(f"Saved training history: {HISTORY}")


if __name__ == "__main__":
    """
    Training Pipeline
    We made it! ❀༉‧
    """
    main()
# ~~~~~~~~~~ EOF ~~~~~~~~~~~
