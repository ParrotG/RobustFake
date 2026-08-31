"""Train a frozen CLIP vision encoder plus a binary AI-image classifier.

Dataset layout:
dataset/
    train/FAKE, train/REAL
    test/FAKE,  test/REAL
"""

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPImageProcessor, CLIPVisionModel


# Paths are derived from this script so it can be run from any working directory.
PROJECT_DIR = Path(__file__).resolve().parent
CLIP_DIR = PROJECT_DIR / "clip-vit-base-patch32"
DATASET_DIR = PROJECT_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
OUTPUT_DIR = PROJECT_DIR / "model"

CLASS_TO_INDEX = {"FAKE": 0, "REAL": 1}
CLASS_NAMES = ["FAKE", "REAL"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

SEED = 42
VALIDATION_RATIO = 0.20
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_WORKERS = 0  # Set this above zero after confirming Windows multiprocessing works locally.


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_samples(split_dir: Path) -> list[tuple[Path, int]]:
    """Collect images and enforce the expected FAKE/REAL directory structure."""
    samples: list[tuple[Path, int]] = []
    for class_name, label in CLASS_TO_INDEX.items():
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        samples.extend(
            (path, label)
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    if not samples:
        raise RuntimeError(f"No supported image files found under {split_dir}")
    return sorted(samples, key=lambda item: str(item[0]))


class ClipImageDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, int]],
        image_processor: CLIPImageProcessor,
        augment: bool = False,
    ) -> None:
        self.samples = samples
        self.image_processor = image_processor
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.augment and random.random() < 0.5:
                image = ImageOps.mirror(image)

            pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values[0]

        return pixel_values, label


class FrozenClipBinaryClassifier(nn.Module):
    def __init__(self, vision_encoder: CLIPVisionModel) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.vision_encoder.requires_grad_(False)
        self.vision_encoder.eval()

        hidden_size = vision_encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 2),
        )

    def train(self, mode: bool = True):
        """Keep the frozen encoder in eval mode even while the head is trained."""
        super().train(mode)
        self.vision_encoder.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # No gradients or text encoder are involved in feature extraction.
        with torch.no_grad():
            outputs = self.vision_encoder(pixel_values=pixel_values)
            features = outputs.pooler_output
        return self.classifier(features)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, list[int], list[int]]:
    is_training = optimizer is not None
    model.train(is_training)

    running_loss = 0.0
    predictions: list[int] = []
    targets: list[int] = []

    for pixel_values, labels in loader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(pixel_values)
        loss = criterion(logits, labels)

        if is_training:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * labels.size(0)
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        targets.extend(labels.cpu().tolist())

    mean_loss = running_loss / len(loader.dataset)
    accuracy = accuracy_score(targets, predictions)
    return mean_loss, accuracy, targets, predictions


def save_training_curves(history: dict[str, list[float]], output_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_accuracy"], label="Train")
    axes[1].plot(epochs, history["val_accuracy"], label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_samples = collect_samples(TRAIN_DIR)
    test_samples = collect_samples(TEST_DIR)
    paths = [path for path, _ in train_samples]
    labels = [label for _, label in train_samples]
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths,
        labels,
        test_size=VALIDATION_RATIO,
        random_state=SEED,
        stratify=labels,
    )
    train_split = list(zip(train_paths, train_labels))
    val_split = list(zip(val_paths, val_labels))
    print(f"Images - train: {len(train_split)}, validation: {len(val_split)}, test: {len(test_samples)}")

    image_processor = CLIPImageProcessor.from_pretrained(CLIP_DIR, local_files_only=True)
    vision_encoder = CLIPVisionModel.from_pretrained(CLIP_DIR, local_files_only=True)
    model = FrozenClipBinaryClassifier(vision_encoder).to(device)

    train_loader = DataLoader(
        ClipImageDataset(train_split, image_processor, augment=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        ClipImageDataset(val_split, image_processor),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        ClipImageDataset(test_samples, image_processor),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
    best_val_accuracy = -1.0
    checkpoint_path = OUTPUT_DIR / "clip_binary_classifier_best.pt"

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_accuracy, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_accuracy, _, _ = run_epoch(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)
        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"train loss: {train_loss:.4f}, acc: {train_accuracy:.4f} | "
            f"val loss: {val_loss:.4f}, acc: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "classifier_state_dict": model.classifier.state_dict(),
                    "vision_model_path": str(CLIP_DIR),
                    "class_to_index": CLASS_TO_INDEX,
                    "best_validation_accuracy": best_val_accuracy,
                },
                checkpoint_path,
            )

    save_training_curves(history, OUTPUT_DIR / "training_curves.png")
    with (OUTPUT_DIR / "training_history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
    test_loss, test_accuracy, test_targets, test_predictions = run_epoch(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f}, test accuracy: {test_accuracy:.4f}")

    matrix = confusion_matrix(test_targets, test_predictions, labels=[0, 1])
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=CLASS_NAMES)
    figure, axis = plt.subplots(figsize=(6, 5))
    display.plot(ax=axis, cmap="Blues", colorbar=False)
    axis.set_title("Test Set Confusion Matrix")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "test_confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    metrics = {
        "best_validation_accuracy": best_val_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "confusion_matrix": matrix.tolist(),
        "class_to_index": CLASS_TO_INDEX,
    }
    with (OUTPUT_DIR / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


if __name__ == "__main__":
    main()
