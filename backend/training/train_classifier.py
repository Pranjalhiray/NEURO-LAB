import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small

from backend.training.data import CLASSES, ClassificationDataset, load_rows

ROOT = Path(__file__).resolve().parents[2]


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += int((model(images).argmax(1) == labels).sum())
            total += int(labels.numel())
    return correct / total if total else 0.0


def main():
    parser = argparse.ArgumentParser(description="Train a four-class MobileNetV3-Small on a prepared BRISC manifest.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifests" / "brisc.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "assets" / "models" / "classifier")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("Epoch count and batch size must be positive.")
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(item.name not in {"README.md", ".gitkeep"} for item in output.iterdir()):
        raise SystemExit(f"Refusing to overwrite existing model files: {output}")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rows, val_rows = load_rows(args.manifest, "train"), load_rows(args.manifest, "validation")
    if not train_rows or not val_rows:
        raise SystemExit("Manifest needs non-empty train and validation splits.")
    train_loader = DataLoader(ClassificationDataset(train_rows, True), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(ClassificationDataset(val_rows, False), batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASSES))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    best_score = -1.0
    best_state = None
    for epoch in range(args.epochs):
        model.train(); running = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward(); optimizer.step()
            running += float(loss.detach())
        val_accuracy = evaluate(model, val_loader, device)
        print(f"epoch {epoch + 1}/{args.epochs} loss={running / max(1, len(train_loader)):.4f} validation_accuracy={val_accuracy:.4f}")
        if val_accuracy > best_score:
            best_score = val_accuracy
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output / "model.pt")
    (output / "model.json").write_text(json.dumps({
        "task": "classifier", "name": "MobileNetV3-Small trained on local manifest",
        "architecture": "mobilenet_v3_small", "input_size": 224, "weights_file": "model.pt",
        "classes": CLASSES, "mean": [.485, .456, .406], "std": [.229, .224, .225],
        "class_score_semantics": "uncalibrated_model_scores",
    }, indent=2), encoding="utf-8")
    (output / "training.json").write_text(json.dumps({
        "task": "classifier", "seed": args.seed, "epochs_requested": args.epochs,
        "validation_accuracy_best": best_score, "train_images": len(train_rows),
        "validation_images": len(val_rows), "device": str(device),
        "warning": "Validation result only. This does not replace an untouched test-set evaluation.",
    }, indent=2), encoding="utf-8")
    print(f"Saved trained classifier to {output}. Run the evaluator once on the untouched test split.")


if __name__ == "__main__":
    main()
