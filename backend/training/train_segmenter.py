"""Train an attention U-Net and select its threshold using validation only."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from backend.app.models import build_attention_unet, predict_segmentation_probability
from backend.training.data import SegmentationDataset, load_rows

ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS = tuple(float(value) for value in np.arange(.15, .851, .05))


def segmentation_loss(logits, targets, bce):
    probabilities = torch.sigmoid(logits)
    axes = (1, 2, 3)
    intersection = (probabilities * targets).sum(dim=axes)
    predicted = probabilities.sum(dim=axes)
    actual = targets.sum(dim=axes)
    dice_loss = 1 - ((2 * intersection + 1) / (predicted + actual + 1)).mean()

    true_positive = intersection
    false_positive = (probabilities * (1 - targets)).sum(dim=axes)
    false_negative = ((1 - probabilities) * targets).sum(dim=axes)
    tversky = (true_positive + 1) / (true_positive + .35 * false_positive + .65 * false_negative + 1)
    focal_tversky_loss = (1 - tversky).clamp_min(0).pow(.75).mean()
    return .5 * bce(logits, targets) + dice_loss + .35 * focal_tversky_loss


@torch.inference_mode()
def validation_curve(model, loader, device, thresholds, horizontal_flip_tta):
    totals = np.zeros(len(thresholds), dtype=np.float64)
    count = 0
    mixed_precision = device.type == "cuda"
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True) >= .5
        with torch.autocast(device_type=device.type, enabled=mixed_precision):
            probabilities = predict_segmentation_probability(model, images, horizontal_flip_tta).float()
        for index, threshold in enumerate(thresholds):
            prediction = probabilities >= threshold
            intersection = (prediction & masks).sum(dim=(1, 2, 3)).float()
            predicted = prediction.sum(dim=(1, 2, 3)).float()
            actual = masks.sum(dim=(1, 2, 3)).float()
            denominator = predicted + actual
            dice = torch.where(denominator == 0, 1.0, 2 * intersection / denominator.clamp_min(1))
            totals[index] += float(dice.sum().cpu())
        count += int(images.shape[0])
    if not count:
        raise ValueError("Validation loader is empty.")
    return {str(threshold): float(total / count) for threshold, total in zip(thresholds, totals)}


def main():
    parser = argparse.ArgumentParser(description="Train a residual Attention U-Net on indexed 2D image/mask pairs.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifests" / "brisc.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "assets" / "models" / "segmenter-candidate")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.0004)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--horizontal-flip-tta", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.workers < 0 or args.input_size < 32:
        raise SystemExit("Epochs, patience, and batch size must be positive; input size must be at least 32.")
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(item.name not in {"README.md", ".gitkeep"} for item in output.iterdir()):
        raise SystemExit(f"Refusing to overwrite existing model files: {output}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = SegmentationDataset(load_rows(args.manifest, "train"), True, size=args.input_size)
    val_data = SegmentationDataset(load_rows(args.manifest, "validation"), False, size=args.input_size)
    if not len(train_data) or not len(val_data):
        raise SystemExit("Manifest needs masks in train and validation, plus no-tumor images for empty-mask examples.")

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=device.type == "cuda", generator=generator, persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
    )
    model = build_attention_unet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0], device=device))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=.5, patience=3, min_lr=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_dice = -1.0
    best_threshold = .5
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    history = []
    print(
        f"Training attention U-Net on {len(train_data)} train and {len(val_data)} validation images; "
        f"device={device}, input={args.input_size}, workers={args.workers}, batches/epoch={len(train_loader)}",
        flush=True,
    )

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        for batch_index, (images, masks) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = segmentation_loss(logits, masks, bce)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach())
            if batch_index % 200 == 0:
                print(f"epoch {epoch + 1}: trained {batch_index}/{len(train_loader)} batches", flush=True)

        model.eval()
        curve = validation_curve(model, val_loader, device, THRESHOLDS, args.horizontal_flip_tta)
        threshold_text, val_dice = max(curve.items(), key=lambda item: item[1])
        val_threshold = float(threshold_text)
        scheduler.step(val_dice)
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / max(1, len(train_loader)),
            "validation_dice": val_dice,
            "validation_threshold": val_threshold,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_record)
        print(
            f"epoch {epoch + 1}/{args.epochs} "
            f"loss={epoch_record['train_loss']:.4f} "
            f"validation_dice={val_dice:.4f} threshold={val_threshold:.2f} "
            f"lr={epoch_record['learning_rate']:.6g}",
            flush=True,
        )
        if val_dice > best_dice + 1e-4:
            best_dice = val_dice
            best_threshold = val_threshold
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch + 1
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stop after {epoch + 1} epochs; best epoch={best_epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a validation checkpoint.")
    output.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output / "model.pt")
    metadata = {
        "task": "segmenter",
        "name": "Residual Attention U-Net trained on local BRISC 2D image/mask pairs",
        "architecture": "attention_unet",
        "input_size": args.input_size,
        "workers": args.workers,
        "weights_file": "model.pt",
        "threshold": best_threshold,
        "horizontal_flip_tta": args.horizontal_flip_tta,
        "mean": [.485, .456, .406],
        "std": [.229, .224, .225],
        "mask_semantics": "binary predicted tumor-region mask; not a diagnosis",
    }
    (output / "model.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "training.json").write_text(json.dumps({
        "task": "segmenter",
        "seed": args.seed,
        "best_epoch": best_epoch,
        "epochs_requested": args.epochs,
        "validation_dice_best": best_dice,
        "validation_threshold_selected": best_threshold,
        "train_masks_or_negatives": len(train_data),
        "validation_masks_or_negatives": len(val_data),
        "input_size": args.input_size,
        "device": str(device),
        "horizontal_flip_tta": args.horizontal_flip_tta,
        "training_history": history,
        "warning": "Validation result only. The held-out test split was not used for training, epoch selection, or threshold selection.",
    }, indent=2), encoding="utf-8")
    print(f"Saved candidate to {output}. It has not been installed as the active model.", flush=True)


if __name__ == "__main__":
    main()
