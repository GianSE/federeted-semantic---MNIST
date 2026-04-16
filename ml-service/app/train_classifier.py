"""
train_classifier.py
-------------------
Train a lightweight CNN classifier for each dataset (MNIST, Fashion-MNIST,
CIFAR-10, CIFAR-100) to validate semantic preservation.

Usage:
    python -m app.train_classifier --dataset mnist --epochs 5
    python -m app.train_classifier --dataset cifar10 --epochs 10

Weights are saved to: /ml-data/weights/<dataset>_classifier.pth
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from app.core.classifier_utils import SimpleClassifier
from app.core.image_utils import load_dataset, DATASET_META


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_classifier(
    dataset_name: str,
    epochs: int = 5,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> str:
    set_seed(seed)

    meta = DATASET_META.get(dataset_name)
    if meta is None:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose from: {list(DATASET_META)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = load_dataset(dataset_name, train=True)
    test_ds = load_dataset(dataset_name, train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SimpleClassifier(
        input_channels=meta["channels"],
        image_size=meta["height"],
        num_classes=meta.get("classes", 10),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for data, labels in train_loader:
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += labels.size(0)

        train_acc = correct / max(1, total)
        avg_loss = running_loss / max(1, len(train_loader))

        model.eval()
        eval_correct = 0
        eval_total = 0
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(device)
                labels = labels.to(device)
                logits = model(data)
                preds = logits.argmax(dim=1)
                eval_correct += int((preds == labels).sum().item())
                eval_total += labels.size(0)

        eval_acc = eval_correct / max(1, eval_total)
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"loss={avg_loss:.4f} | train_acc={train_acc:.3f} | test_acc={eval_acc:.3f}"
        )

    weights_dir = "/ml-data/weights"
    os.makedirs(weights_dir, exist_ok=True)
    save_path = os.path.join(weights_dir, f"{dataset_name}_classifier.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\nWeights saved -> {save_path}\n")
    return save_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dataset-specific classifiers.")
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_META.keys()),
        required=True,
        help="Dataset to train on.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs.")
    parser.add_argument("--batch", type=int, default=128, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_classifier(
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        seed=args.seed,
    )
