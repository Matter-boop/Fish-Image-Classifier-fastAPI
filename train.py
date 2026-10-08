from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


ROOT_DIR = Path(__file__).resolve().parent
INDEX_PATH = ROOT_DIR / "Fish_Data" / "final_all_index.txt"
IMAGE_DIR = ROOT_DIR / "Fish_Data" / "images" / "cropped"
MODEL_PATH = ROOT_DIR / "model" / "fish_resnet18.pth"
IMAGE_SIZE = 224
SEED = 42


def load_records() -> list[tuple[Path, str]]:
    """Read label and cropped-image filename from the dataset index."""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Dataset index not found: {INDEX_PATH}")

    records: list[tuple[Path, str]] = []
    missing = 0
    for line_number, line in enumerate(
        INDEX_PATH.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = line.strip().split("=")
        if len(fields) != 5:
            print(f"Skipping malformed index row {line_number}")
            continue
        _, label, _, image_stem, _ = fields
        image_path = IMAGE_DIR / f"{image_stem}.png"
        if image_path.exists():
            records.append((image_path, label))
        else:
            missing += 1

    if missing:
        print(f"Skipped {missing} rows whose cropped image was not found.")
    if not records:
        raise RuntimeError(f"No indexed images found in {IMAGE_DIR}")
    return records


def split_by_class(
    records: list[tuple[Path, str]], validation_fraction: float = 0.2
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]]]:
    """Make a deterministic per-class split without removing tiny classes from training."""
    grouped: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    for record in records:
        grouped[record[1]].append(record)

    rng = random.Random(SEED)
    training: list[tuple[Path, str]] = []
    validation: list[tuple[Path, str]] = []
    for class_records in grouped.values():
        rng.shuffle(class_records)
        if len(class_records) >= 3:
            validation_count = max(1, round(len(class_records) * validation_fraction))
            validation.extend(class_records[:validation_count])
            training.extend(class_records[validation_count:])
        else:
            training.extend(class_records)
    return training, validation


class FishDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        records: list[tuple[Path, str]],
        class_to_index: dict[str, int],
        image_transform: transforms.Compose,
    ) -> None:
        self.records = records
        self.class_to_index = class_to_index
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.records[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        return self.image_transform(image), self.class_to_index[label]


def make_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    training_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return training_transform, validation_transform


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / max(total, 1), correct / max(total, 1)


def train(epochs: int, batch_size: int) -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)

    records = load_records()
    class_names = sorted({label for _, label in records})
    class_to_index = {name: index for index, name in enumerate(class_names)}
    training_records, validation_records = split_by_class(records)
    training_transform, validation_transform = make_transforms()

    training_dataset = FishDataset(training_records, class_to_index, training_transform)
    training_loader = DataLoader(
        training_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    validation_loader = None
    if validation_records:
        validation_dataset = FishDataset(
            validation_records, class_to_index, validation_transform
        )
        validation_loader = DataLoader(
            validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Training on {device}: {len(training_records)} training images, "
        f"{len(validation_records)} validation images, {len(class_names)} classes."
    )
    print("Downloading ImageNet ResNet18 weights on first run if not already cached.")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.to(device)

    counts = Counter(label for _, label in training_records)
    class_weights = torch.tensor(
        [1.0 / (counts[name] ** 0.5) for name in class_names],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_validation_loss = float("inf")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        training_loss, training_accuracy = run_epoch(
            model, training_loader, criterion, device, optimizer
        )
        if validation_loader is not None:
            validation_loss, validation_accuracy = run_epoch(
                model, validation_loader, criterion, device
            )
        else:
            validation_loss, validation_accuracy = training_loss, training_accuracy

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss {training_loss:.4f}, accuracy {training_accuracy:.1%} | "
            f"validation loss {validation_loss:.4f}, accuracy {validation_accuracy:.1%}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                {
                    "architecture": "resnet18",
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "image_size": IMAGE_SIZE,
                },
                MODEL_PATH,
            )

    print(f"Best checkpoint saved to {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the fish image classifier.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    arguments = parser.parse_args()
    if arguments.epochs < 1 or arguments.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive integers")
    train(epochs=arguments.epochs, batch_size=arguments.batch_size)