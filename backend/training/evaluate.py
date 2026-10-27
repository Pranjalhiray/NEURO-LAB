"""Evaluate installed models against the manifest's untouched test split."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from backend.app.inference import CLASSES, _load_model, registry
from backend.app.models import predict_segmentation_probability
from backend.training.data import (
    ClassificationDataset,
    SegmentationDataset,
    load_rows,
)

ROOT = Path(__file__).resolve().parents[2]
DATASET_ID = "BRISC 2025 v3 (local manifest)"


def _binary_auc(labels: list[int], scores: list[float]) -> float | None:
    """Mann–Whitney AUC with average ranks for tied scores."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    order = sorted(range(len(scores)), key=scores.__getitem__)
    rank_sum = 0.0
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and scores[order[end]] == scores[order[cursor]]:
            end += 1
        average_rank = ((cursor + 1) + end) / 2
        rank_sum += sum(average_rank for index in order[cursor:end] if labels[index])
        cursor = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def classification_metrics(model, loader, device, classes: list[str] | tuple[str, ...] = CLASSES):
    classes = list(classes)
    if len(classes) != len(CLASSES) or any(not isinstance(name, str) or not name for name in classes):
        raise ValueError("Classifier class labels must match the BRISC manifest classes exactly.")
    if len(set(classes)) != len(CLASSES) or set(classes) != set(CLASSES):
        raise ValueError("Classifier class labels must match the BRISC manifest classes exactly.")
    class_to_index = {name: index for index, name in enumerate(classes)}
    matrix = [[0 for _ in classes] for _ in classes]
    truth_columns: list[list[int]] = [[] for _ in classes]
    score_columns: list[list[float]] = [[] for _ in classes]
    with torch.inference_mode():
        for images, labels in loader:
            probabilities = torch.softmax(model(images.to(device)), dim=1).cpu()
            predictions = probabilities.argmax(1).tolist()
            for label_index, predicted, scores in zip(labels.tolist(), predictions, probabilities.tolist()):
                actual_name = CLASSES[label_index]
                if actual_name not in class_to_index:
                    continue
                actual = class_to_index[actual_name]
                matrix[actual][predicted] += 1
                for index in range(len(classes)):
                    truth_columns[index].append(int(actual == index))
                    score_columns[index].append(float(scores[index]))

    total = sum(map(sum, matrix))
    correct = sum(matrix[index][index] for index in range(len(classes)))
    per_class = {}
    precisions, recalls, f1_values = [], [], []
    auc_values = []
    for index, name in enumerate(classes):
        tp = matrix[index][index]
        fp = sum(matrix[row][index] for row in range(len(classes))) - tp
        fn = sum(matrix[index]) - tp
        support = sum(matrix[index])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        auc = _binary_auc(truth_columns[index], score_columns[index])
        if auc is not None:
            auc_values.append(auc)
        precisions.append(precision); recalls.append(recall); f1_values.append(f1)
        per_class[name] = {
            "precision": precision, "recall": recall, "f1": f1,
            "support": support, "roc_auc_ovr": auc,
        }
    return {
        "accuracy": correct / total if total else None,
        "macro_precision": sum(precisions) / len(precisions) if precisions else None,
        "macro_recall": sum(recalls) / len(recalls) if recalls else None,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "roc_auc_ovr_macro": sum(auc_values) / len(auc_values) if len(auc_values) == len(classes) else None,
        "averaging": "unweighted macro; ROC-AUC is one-vs-rest macro",
        "per_class": per_class,
        "confusion_matrix": matrix,
        "class_order": classes,
        "sample_count": total,
    }


def segmentation_metrics(model, loader, device, threshold=.5, rows=None, classes=CLASSES, horizontal_flip_tta=False):
    dice_values, iou_values = [], []
    class_values: dict[str, dict[str, list[float]]] = {}
    cursor = 0
    mask_resolution = None
    with torch.inference_mode():
        for images, masks in loader:
            prediction = predict_segmentation_probability(
                model, images.to(device), horizontal_flip_tta,
            ) >= threshold
            truth = masks.to(device) >= .5
            if mask_resolution is None:
                mask_resolution = [int(truth.shape[-2]), int(truth.shape[-1])]
            for pred, target in zip(prediction, truth):
                intersection = int((pred & target).sum())
                pred_count = int(pred.sum())
                target_count = int(target.sum())
                union = pred_count + target_count - intersection
                dice = (2 * intersection / (pred_count + target_count)) if pred_count + target_count else 1.0
                iou = intersection / union if union else 1.0
                dice_values.append(dice); iou_values.append(iou)
                row = rows[cursor] if rows and cursor < len(rows) else {}
                label = row.get("label", "unknown")
                class_values.setdefault(label, {"dice": [], "iou": []})
                class_values[label]["dice"].append(dice); class_values[label]["iou"].append(iou)
                cursor += 1
    return {
        "dice": sum(dice_values) / len(dice_values) if dice_values else None,
        "iou": sum(iou_values) / len(iou_values) if iou_values else None,
        "sample_count": len(dice_values),
        "mask_resolution": mask_resolution,
        "per_class": {
            name: {metric: sum(values[metric]) / len(values[metric]) for metric in ("dice", "iou")}
            | {"support": len(values["dice"])}
            for name, values in class_values.items()
        },
        "empty_mask_convention": "Empty/empty cases score 1.0; an empty ground truth with a non-empty prediction scores 0.0.",
    }


def evaluate_task(task: str, manifest: Path = ROOT / "data" / "manifests" / "brisc.csv", output: Path = ROOT / "assets" / "evaluations"):
    rows = load_rows(manifest, "test")
    if not rows:
        raise ValueError("The manifest has no held-out test split.")
    model, device, metadata = _load_model(task)
    if task == "classifier":
        classes = metadata.get("classes", list(CLASSES))
        metrics = classification_metrics(
            model, DataLoader(ClassificationDataset(rows, False), batch_size=16, shuffle=False), device, classes,
        )
    elif task == "segmenter":
        data = SegmentationDataset(rows, False, size=int(metadata.get("input_size", 256)))
        if not len(data):
            raise ValueError("The test split contains no matched masks or no-tumor examples.")
        metrics = segmentation_metrics(
            model, DataLoader(data, batch_size=4, shuffle=False), device,
            float(metadata.get("threshold", .5)), rows=data.rows,
            horizontal_flip_tta=bool(metadata.get("horizontal_flip_tta", False)),
        )
    else:
        raise ValueError("Unsupported evaluation task.")
    manifest_path = Path(manifest).expanduser().resolve()
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.mkdir(parents=True, exist_ok=True)
    report = {
        "task": task,
        "model_name": metadata.get("name"),
        "architecture": metadata.get("architecture"),
        "split": "test",
        "dataset": {"id": DATASET_ID, "manifest": str(manifest_path), "test_rows": len(rows)},
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "limitations": ["BRISC does not provide patient identifiers; the split cannot be verified as patient-independent."],
    }
    destination = output_path / (task + ".json")
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("classifier", "segmenter"), required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifests" / "brisc.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "assets" / "evaluations")
    args = parser.parse_args()
    report = evaluate_task(args.task, args.manifest, args.output)
    print(json.dumps(report, indent=2))
    print(f"Saved test-set evaluation to {Path(args.output) / (args.task + '.json')}")


if __name__ == "__main__":
    main()
