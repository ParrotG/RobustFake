"""Learn a probability-level ensemble weight for the model1 and model2 classifiers.

model1 supplies global CLS-token probabilities Pg.  model2 supplies local
patch-attention probabilities Pp.  Only alpha is trainable:
Pf = alpha * Pg + (1 - alpha) * Pp, where alpha is constrained to [0, 1].
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
import torch.nn.functional as functional
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms
from transformers import AutoImageProcessor, AutoModel


PROJECT_ROOT = Path(r"D:\NTU_project\hackson\DINO-v2")
DINO_MODEL_DIR = PROJECT_ROOT / "DINO_v2_based"
MODEL1_PATH = PROJECT_ROOT / "model" / "model1" / "best_classifier.pt"
MODEL2_PATH = PROJECT_ROOT / "model" / "model2" / "best_patch_attention_classifier.pt"
ALPHA_TRAIN_DIR = PROJECT_ROOT / "dataset" / "train2.0"
TEST_DIR = PROJECT_ROOT / "dataset" / "test"
OUTPUT_DIR = PROJECT_ROOT / "model" / "model3"
CLASS_NAMES = ["FAKE", "REAL"]


class RGBImageFolder(datasets.ImageFolder):
    def __getitem__(self, index: int):
        path, target = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


class LocalPatchClassifier(nn.Module):
    """The exact model2 head architecture used by train3.0.py."""

    def __init__(self, hidden_size: int = 768, num_classes: int = 2):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attention(patch_tokens), dim=1)
        return self.classifier(torch.sum(weights * patch_tokens, dim=1))


class ProbabilityEnsemble(nn.Module):
    """One learned scalar alpha, represented as a logit for a stable [0, 1] range."""

    def __init__(self):
        super().__init__()
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_logit)

    def forward(self, global_probabilities: torch.Tensor, patch_probabilities: torch.Tensor) -> torch.Tensor:
        return self.alpha * global_probabilities + (1.0 - self.alpha) * patch_probabilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learn alpha for model1/model2 probability ensemble")
    parser.add_argument("--epochs", type=int, default=50, help="Only one scalar alpha is optimized.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size while extracting DINO features.")
    parser.add_argument("--alpha-learning-rate", type=float, default=0.05)
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


def check_dataset(dataset: datasets.ImageFolder, name: str) -> None:
    if dataset.classes != CLASS_NAMES:
        raise ValueError(f"{name} must contain exactly {CLASS_NAMES}; found {dataset.classes}.")
    if not dataset.samples:
        raise ValueError(f"{name} contains no image files.")


def split_dataset(dataset: datasets.ImageFolder, ratio: float, seed: int) -> tuple[Subset, Subset]:
    if not 0 < ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1.")
    indices = np.arange(len(dataset))
    train_indices, validation_indices = train_test_split(
        indices, test_size=ratio, random_state=seed, stratify=np.asarray(dataset.targets)
    )
    return Subset(dataset, train_indices.tolist()), Subset(dataset, validation_indices.tolist())


def load_base_models(device: torch.device) -> tuple[nn.Module, nn.Module, nn.Module, dict[str, int]]:
    model1_checkpoint = torch.load(MODEL1_PATH, map_location=device, weights_only=False)
    model2_checkpoint = torch.load(MODEL2_PATH, map_location=device, weights_only=False)
    class_to_idx = model1_checkpoint["class_to_idx"]
    if class_to_idx != model2_checkpoint["class_to_idx"]:
        raise ValueError("model1 and model2 use different class label mappings.")
    if set(class_to_idx) != set(CLASS_NAMES):
        raise ValueError(f"Unexpected model label mapping: {class_to_idx}")

    dino = AutoModel.from_pretrained(DINO_MODEL_DIR, local_files_only=True).to(device)
    global_classifier = nn.Linear(model1_checkpoint["feature_dim"], len(class_to_idx)).to(device)
    global_classifier.load_state_dict(model1_checkpoint["classifier_state_dict"])
    patch_classifier = LocalPatchClassifier(
        hidden_size=model2_checkpoint["hidden_size"], num_classes=len(class_to_idx)
    ).to(device)
    patch_classifier.load_state_dict(model2_checkpoint["classifier_state_dict"])
    for model in (dino, global_classifier, patch_classifier):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
    return dino, global_classifier, patch_classifier, class_to_idx


def extract_probabilities(
    dataset: Dataset,
    dino: nn.Module,
    global_classifier: nn.Module,
    patch_classifier: nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run frozen DINO once per image and cache Pg, Pp, and its true label."""
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    all_global_probabilities: list[torch.Tensor] = []
    all_patch_probabilities: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            hidden_states = dino(pixel_values=images).last_hidden_state
            global_logits = global_classifier(hidden_states[:, 0, :])
            patch_logits = patch_classifier(hidden_states[:, 1:, :])
            all_global_probabilities.append(torch.softmax(global_logits, dim=1).cpu())
            all_patch_probabilities.append(torch.softmax(patch_logits, dim=1).cpu())
            all_labels.append(labels.cpu())
    return (
        torch.cat(all_global_probabilities),
        torch.cat(all_patch_probabilities),
        torch.cat(all_labels),
    )


def evaluate_alpha(
    ensemble: ProbabilityEnsemble, probability_loader: DataLoader, device: torch.device
) -> tuple[float, float, list[int], list[int]]:
    ensemble.eval()
    total_loss = 0.0
    total_count = 0
    labels_all: list[int] = []
    predictions_all: list[int] = []
    with torch.inference_mode():
        for global_probabilities, patch_probabilities, labels in probability_loader:
            global_probabilities = global_probabilities.to(device)
            patch_probabilities = patch_probabilities.to(device)
            labels = labels.to(device)
            final_probabilities = ensemble(global_probabilities, patch_probabilities)
            loss = functional.nll_loss(torch.log(final_probabilities.clamp_min(1e-8)), labels)
            total_loss += loss.item() * labels.size(0)
            total_count += labels.size(0)
            labels_all.extend(labels.cpu().tolist())
            predictions_all.extend(final_probabilities.argmax(dim=1).cpu().tolist())
    return total_loss / total_count, accuracy_score(labels_all, predictions_all), labels_all, predictions_all


def save_curves(history: dict[str, list[float]]) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Alpha-training loss")
    plt.plot(epochs, history["validation_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Negative log-likelihood")
    plt.title("Ensemble alpha training loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_accuracy"], label="Alpha-training accuracy")
    plt.plot(epochs, history["validation_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("Ensemble alpha training accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accuracy_curve.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for required_path in (DINO_MODEL_DIR, MODEL1_PATH, MODEL2_PATH, ALPHA_TRAIN_DIR, TEST_DIR):
        if not required_path.exists():
            raise FileNotFoundError(f"Required path not found: {required_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available. Install CUDA-enabled PyTorch before training.")
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    transform = build_transform()
    alpha_complete_dataset = RGBImageFolder(ALPHA_TRAIN_DIR, transform=transform)
    test_dataset = RGBImageFolder(TEST_DIR, transform=transform)
    check_dataset(alpha_complete_dataset, "dataset/train2.0")
    check_dataset(test_dataset, "dataset/test")
    alpha_train_dataset, validation_dataset = split_dataset(
        alpha_complete_dataset, args.val_ratio, args.seed
    )
    print(
        f"Alpha train: {len(alpha_train_dataset)}, validation: {len(validation_dataset)}, "
        f"test: {len(test_dataset)}"
    )

    dino, global_classifier, patch_classifier, class_to_idx = load_base_models(device)
    print("Extracting frozen model1/model2 probabilities from train2.0 and test data...")
    train_pg, train_pp, train_labels = extract_probabilities(
        alpha_train_dataset, dino, global_classifier, patch_classifier, device, args.batch_size, args.num_workers
    )
    val_pg, val_pp, val_labels = extract_probabilities(
        validation_dataset, dino, global_classifier, patch_classifier, device, args.batch_size, args.num_workers
    )
    test_pg, test_pp, test_labels = extract_probabilities(
        test_dataset, dino, global_classifier, patch_classifier, device, args.batch_size, args.num_workers
    )

    # After caching probabilities, DINO and both base heads are no longer involved in optimization.
    train_loader = DataLoader(TensorDataset(train_pg, train_pp, train_labels), batch_size=1024, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_pg, val_pp, val_labels), batch_size=1024, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_pg, test_pp, test_labels), batch_size=1024, shuffle=False)
    ensemble = ProbabilityEnsemble().to(device)
    optimizer = torch.optim.Adam([ensemble.alpha_logit], lr=args.alpha_learning_rate)
    history: dict[str, list[float]] = {
        "train_loss": [], "validation_loss": [], "train_accuracy": [], "validation_accuracy": [], "alpha": []
    }
    best_validation_accuracy = -1.0
    checkpoint_path = OUTPUT_DIR / "probability_ensemble_alpha.pt"

    for epoch in range(1, args.epochs + 1):
        ensemble.train()
        total_loss = 0.0
        total_count = 0
        train_labels_all: list[int] = []
        train_predictions_all: list[int] = []
        for global_probabilities, patch_probabilities, labels in train_loader:
            global_probabilities = global_probabilities.to(device)
            patch_probabilities = patch_probabilities.to(device)
            labels = labels.to(device)
            final_probabilities = ensemble(global_probabilities, patch_probabilities)
            loss = functional.nll_loss(torch.log(final_probabilities.clamp_min(1e-8)), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_count += labels.size(0)
            train_labels_all.extend(labels.detach().cpu().tolist())
            train_predictions_all.extend(final_probabilities.argmax(dim=1).detach().cpu().tolist())

        train_loss = total_loss / total_count
        train_accuracy = accuracy_score(train_labels_all, train_predictions_all)
        validation_loss, validation_accuracy, _, _ = evaluate_alpha(ensemble, val_loader, device)
        alpha_value = ensemble.alpha.item()
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_accuracy"].append(validation_accuracy)
        history["alpha"].append(alpha_value)
        print(
            f"Epoch {epoch:02d}/{args.epochs}: alpha={alpha_value:.6f}; "
            f"train loss={train_loss:.4f}, acc={train_accuracy:.4f}; "
            f"val loss={validation_loss:.4f}, acc={validation_accuracy:.4f}"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(
                {
                    "alpha_logit": ensemble.alpha_logit.detach().cpu(),
                    "alpha": alpha_value,
                    "formula": "Pf = alpha * Pg + (1 - alpha) * Pp",
                    "model1_checkpoint": str(MODEL1_PATH),
                    "model2_checkpoint": str(MODEL2_PATH),
                    "class_to_idx": class_to_idx,
                    "best_validation_accuracy": best_validation_accuracy,
                },
                checkpoint_path,
            )

    best_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ensemble.alpha_logit.data.copy_(best_checkpoint["alpha_logit"].to(device))
    test_loss, test_accuracy, test_labels_all, test_predictions = evaluate_alpha(ensemble, test_loader, device)
    print(f"Final alpha={ensemble.alpha.item():.6f}; test loss={test_loss:.4f}, test accuracy={test_accuracy:.4f}")

    save_curves(history)
    matrix = confusion_matrix(test_labels_all, test_predictions, labels=[0, 1])
    ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot(cmap="Blues", values_format="d")
    plt.title("Ensemble test-set confusion matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "test_confusion_matrix.png", dpi=200)
    plt.close()

    summary = {
        "formula": "Pf = alpha * Pg + (1 - alpha) * Pp",
        "alpha": ensemble.alpha.item(),
        "alpha_training_data": str(ALPHA_TRAIN_DIR),
        "model1_checkpoint": str(MODEL1_PATH),
        "model2_checkpoint": str(MODEL2_PATH),
        "class_to_idx": class_to_idx,
        "alpha_train_samples": len(alpha_train_dataset),
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
    print(f"Saved ensemble alpha and reports to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
