"""Train a frozen-DINOv2 local-patch attention classifier for FAKE vs REAL.

Only dataset/train and dataset/test are read. dataset/train2.0 and
dataset/detection are deliberately not used.
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from transformers import AutoImageProcessor, AutoModel


PROJECT_ROOT = Path(r"D:\NTU_project\hackson\DINO-v2")
DINO_MODEL_DIR = PROJECT_ROOT / "DINO_v2_based"
TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"
TEST_DIR = PROJECT_ROOT / "dataset" / "test"
OUTPUT_DIR = PROJECT_ROOT / "model" / "model2"
CLASS_NAMES = ["FAKE", "REAL"]


class RGBImageFolder(datasets.ImageFolder):
    """ImageFolder that closes source image files immediately after reading."""

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


class LocalPatchClassifier(nn.Module):
    """Attention-weighted pooling over DINOv2 patch tokens, then classification."""

    def __init__(self, hidden_size: int = 768, num_classes: int = 2):
        super().__init__()
        # Nonlinear patch scoring: s_i = w^T tanh(Wx_i + b) + b.
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        # patch_tokens: [batch_size, num_patches, hidden_size], e.g. [B, 256, 768]
        attention_scores = self.attention(patch_tokens)  # [B, num_patches, 1]
        attention_weights = torch.softmax(attention_scores, dim=1)
        local_feature = torch.sum(attention_weights * patch_tokens, dim=1)  # [B, 768]
        return self.classifier(local_feature)  # [B, 2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DINOv2 patch-attention FAKE/REAL classifier")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transform() -> transforms.Compose:
    processor = AutoImageProcessor.from_pretrained(DINO_MODEL_DIR, local_files_only=True)
    return transforms.Compose(
        [
            transforms.Resize(
                processor.size["shortest_edge"], interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop((processor.crop_size["height"], processor.crop_size["width"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
        ]
    )


def check_dataset(dataset: datasets.ImageFolder, dataset_name: str) -> None:
    if dataset.classes != CLASS_NAMES:
        raise ValueError(
            f"{dataset_name} must contain exactly {CLASS_NAMES}; found {dataset.classes}."
        )
    if not dataset.samples:
        raise ValueError(f"{dataset_name} contains no supported image files.")


def split_train_validation(dataset: datasets.ImageFolder, ratio: float, seed: int) -> tuple[Subset, Subset]:
    if not 0 < ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1.")
    indices = np.arange(len(dataset))
    train_indices, validation_indices = train_test_split(
        indices, test_size=ratio, random_state=seed, stratify=np.asarray(dataset.targets)
    )
    return Subset(dataset, train_indices.tolist()), Subset(dataset, validation_indices.tolist())


def run_epoch(
    dino: nn.Module,
    patch_classifier: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, list[int], list[int]]:
    training = optimizer is not None
    dino.eval()
    patch_classifier.train(training)
    total_loss = 0.0
    total_count = 0
    labels_all: list[int] = []
    predictions_all: list[int] = []

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # DINO's first token is the global CLS token.  Use all remaining
        # transformer patch tokens for attention-based local feature pooling.
        with torch.no_grad():
            patch_tokens = dino(pixel_values=images).last_hidden_state[:, 1:, :]

        logits = patch_classifier(patch_tokens)
        loss = criterion(logits, labels)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size
        labels_all.extend(labels.detach().cpu().tolist())
        predictions_all.extend(logits.argmax(dim=1).detach().cpu().tolist())

    return total_loss / total_count, accuracy_score(labels_all, predictions_all), labels_all, predictions_all


def save_curves(history: dict[str, list[float]]) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Training loss")
    plt.plot(epochs, history["validation_loss"], marker="o", label="Validation loss")
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
    plt.plot(epochs, history["validation_accuracy"], marker="o", label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accuracy_curve.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available. Install CUDA-enabled PyTorch before training.")
    if not DINO_MODEL_DIR.is_dir() or not TRAIN_DIR.is_dir() or not TEST_DIR.is_dir():
        raise FileNotFoundError("DINO_v2_based, dataset/train, and dataset/test must all exist.")
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    image_transform = build_transform()
    train_complete = RGBImageFolder(TRAIN_DIR, transform=image_transform)
    test_dataset = RGBImageFolder(TEST_DIR, transform=image_transform)
    check_dataset(train_complete, "dataset/train")
    check_dataset(test_dataset, "dataset/test")
    train_dataset, validation_dataset = split_train_validation(
        train_complete, args.val_ratio, args.seed
    )

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)
    print(f"Training: {len(train_dataset)}, validation: {len(validation_dataset)}, test: {len(test_dataset)}")

    dino = AutoModel.from_pretrained(DINO_MODEL_DIR, local_files_only=True).to(device)
    for parameter in dino.parameters():
        parameter.requires_grad = False
    dino.eval()

    patch_classifier = LocalPatchClassifier(hidden_size=dino.config.hidden_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(patch_classifier.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    history: dict[str, list[float]] = {
        "train_loss": [], "validation_loss": [], "train_accuracy": [], "validation_accuracy": []
    }
    checkpoint_path = OUTPUT_DIR / "best_patch_attention_classifier.pt"
    best_validation_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy, _, _ = run_epoch(
            dino, patch_classifier, train_loader, criterion, device, optimizer
        )
        validation_loss, validation_accuracy, _, _ = run_epoch(
            dino, patch_classifier, validation_loader, criterion, device
        )
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_accuracy"].append(validation_accuracy)
        print(
            f"Epoch {epoch}/{args.epochs}: train loss={train_loss:.4f}, acc={train_accuracy:.4f}; "
            f"validation loss={validation_loss:.4f}, acc={validation_accuracy:.4f}"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(
                {
                    "classifier_state_dict": patch_classifier.state_dict(),
                    "hidden_size": dino.config.hidden_size,
                    "class_to_idx": train_complete.class_to_idx,
                    "dino_model_dir": str(DINO_MODEL_DIR),
                    "model_type": "LocalPatchClassifier",
                    "best_validation_accuracy": best_validation_accuracy,
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    patch_classifier.load_state_dict(checkpoint["classifier_state_dict"])
    test_loss, test_accuracy, test_labels, test_predictions = run_epoch(
        dino, patch_classifier, test_loader, criterion, device
    )
    print(f"Test: loss={test_loss:.4f}, accuracy={test_accuracy:.4f}")

    save_curves(history)
    matrix = confusion_matrix(test_labels, test_predictions, labels=[0, 1])
    ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot(cmap="Blues", values_format="d")
    plt.title("Test-set confusion matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "test_confusion_matrix.png", dpi=200)
    plt.close()

    summary = {
        "dino_model_dir": str(DINO_MODEL_DIR),
        "dino_frozen": True,
        "head": "attention: Linear(768, 256) -> Tanh -> Linear(256, 1); classifier: Linear(768, 2)",
        "patch_tokens_used": "last_hidden_state[:, 1:, :] (CLS token excluded)",
        "class_to_idx": train_complete.class_to_idx,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "test_samples": len(test_dataset),
        "best_validation_accuracy": best_validation_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }
    (OUTPUT_DIR / "training_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved model and reports to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
