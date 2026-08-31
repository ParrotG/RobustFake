"""Train a frozen-DINOv2 binary image classifier (REAL vs FAKE).

Only ``dataset/train`` and ``dataset/test`` are used.  ``dataset/train2.0``
and ``dataset/detection`` are intentionally not referenced by this script.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # Save figures without opening a GUI window.
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from transformers import AutoImageProcessor, AutoModel


PROJECT_ROOT = Path(r"D:\NTU_project\hackson\DINO-v2")
DINO_MODEL_DIR = PROJECT_ROOT / "DINO_v2_based"
TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"
TEST_DIR = PROJECT_ROOT / "dataset" / "test"
OUTPUT_DIR = PROJECT_ROOT / "model" / "model1"
CLASS_NAMES = ["FAKE", "REAL"]  # FAKE=0, REAL=1 when folder names are correct.


class RGBImageFolder(datasets.ImageFolder):
    """ImageFolder that closes each image file immediately after reading it."""

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a frozen DINOv2 binary classifier")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 on Windows by default.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transform() -> transforms.Compose:
    """Build the preprocessing specified by DINO_v2_based/preprocessor_config.json."""
    processor = AutoImageProcessor.from_pretrained(DINO_MODEL_DIR, local_files_only=True)
    size = processor.size
    crop_size = processor.crop_size
    shortest_edge = size["shortest_edge"]
    crop_height = crop_size["height"]
    crop_width = crop_size["width"]
    return transforms.Compose(
        [
            transforms.Resize(shortest_edge, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop((crop_height, crop_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
        ]
    )


def check_dataset(dataset: datasets.ImageFolder, name: str) -> None:
    if dataset.classes != CLASS_NAMES:
        raise ValueError(
            f"{name} must contain exactly these folders: {CLASS_NAMES}; "
            f"found: {dataset.classes}"
        )
    if not dataset.samples:
        raise ValueError(f"{name} contains no supported image files.")


def split_train_validation(dataset: datasets.ImageFolder, val_ratio: float, seed: int) -> tuple[Subset, Subset]:
    if not 0 < val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1.")
    indices = np.arange(len(dataset))
    targets = np.array(dataset.targets)
    train_indices, val_indices = train_test_split(
        indices, test_size=val_ratio, random_state=seed, stratify=targets
    )
    return Subset(dataset, train_indices.tolist()), Subset(dataset, val_indices.tolist())


def run_epoch(
    dino: nn.Module,
    classifier: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, list[int], list[int]]:
    is_training = optimizer is not None
    classifier.train(is_training)
    dino.eval()  # Frozen backbone must always stay in evaluation mode.

    total_loss = 0.0
    total_samples = 0
    all_predictions: list[int] = []
    all_labels: list[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # No DINOv2 gradients are recorded: only classifier parameters update.
        with torch.no_grad():
            dino_output = dino(pixel_values=images)
            image_features = dino_output.last_hidden_state[:, 0, :]  # CLS: full-image feature

        logits = classifier(image_features)
        loss = criterion(logits, labels)

        if is_training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    return (
        total_loss / total_samples,
        accuracy_score(all_labels, all_predictions),
        all_labels,
        all_predictions,
    )


def save_learning_curves(history: dict[str, list[float]]) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Training loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_accuracy"], marker="o", label="Training accuracy")
    plt.plot(epochs, history["val_accuracy"], marker="o", label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and validation accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accuracy_curve.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DINO_MODEL_DIR.is_dir():
        raise FileNotFoundError(f"DINOv2 model folder not found: {DINO_MODEL_DIR}")
    if not TRAIN_DIR.is_dir() or not TEST_DIR.is_dir():
        raise FileNotFoundError("dataset/train and dataset/test must both exist.")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available. Please install a CUDA-enabled PyTorch build "
            "and confirm that torch.cuda.is_available() returns True."
        )
    device = torch.device("cuda")
    print(f"Using device: {device}")

    image_transform = build_transform()
    full_train_dataset = RGBImageFolder(TRAIN_DIR, transform=image_transform)
    test_dataset = RGBImageFolder(TEST_DIR, transform=image_transform)
    check_dataset(full_train_dataset, "dataset/train")
    check_dataset(test_dataset, "dataset/test")
    train_dataset, val_dataset = split_train_validation(full_train_dataset, args.val_ratio, args.seed)

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)
    print(f"Train: {len(train_dataset)}, validation: {len(val_dataset)}, test: {len(test_dataset)}")

    dino = AutoModel.from_pretrained(DINO_MODEL_DIR, local_files_only=True).to(device)
    for parameter in dino.parameters():
        parameter.requires_grad = False

    classifier = nn.Linear(dino.config.hidden_size, len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []
    }
    best_val_accuracy = -1.0
    checkpoint_path = OUTPUT_DIR / "best_classifier.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _, _ = run_epoch(dino, classifier, train_loader, criterion, device, optimizer)
        val_loss, val_acc, _, _ = run_epoch(dino, classifier, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
        print(
            f"Epoch {epoch:02d}/{args.epochs}: "
            f"train loss={train_loss:.4f}, acc={train_acc:.4f}; "
            f"val loss={val_loss:.4f}, acc={val_acc:.4f}"
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(
                {
                    "classifier_state_dict": classifier.state_dict(),
                    "feature_dim": dino.config.hidden_size,
                    "class_to_idx": full_train_dataset.class_to_idx,
                    "dino_model_dir": str(DINO_MODEL_DIR),
                    "best_validation_accuracy": best_val_accuracy,
                },
                checkpoint_path,
            )

    # Evaluate the classifier selected by validation accuracy exactly once on test data.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classifier.load_state_dict(checkpoint["classifier_state_dict"])
    test_loss, test_acc, test_labels, test_predictions = run_epoch(
        dino, classifier, test_loader, criterion, device
    )
    print(f"Test: loss={test_loss:.4f}, accuracy={test_acc:.4f}")

    save_learning_curves(history)
    matrix = confusion_matrix(test_labels, test_predictions, labels=[0, 1])
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=CLASS_NAMES)
    display.plot(cmap="Blues", values_format="d")
    plt.title("Test-set confusion matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "test_confusion_matrix.png", dpi=200)
    plt.close()

    summary = {
        "device": str(device),
        "dino_model_dir": str(DINO_MODEL_DIR),
        "frozen_dino_parameters": True,
        "classifier": "Linear(768, 2)",
        "class_to_idx": full_train_dataset.class_to_idx,
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "best_validation_accuracy": best_val_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
    }
    (OUTPUT_DIR / "training_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved classifier and reports to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
