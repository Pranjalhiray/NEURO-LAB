"""Index an extracted BRISC-style image/mask dataset without downloading it."""
import argparse
import csv
import hashlib
import random
import re
from pathlib import Path

from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
LABELS = {
    "glioma": "glioma",
    "meningioma": "meningioma",
    "pituitary": "pituitary",
    "no_tumor": "no_tumor",
    "notumor": "no_tumor",
    "no-tumor": "no_tumor",
    "no tumour": "no_tumor",
    "normal": "no_tumor",
    "no_tumour": "no_tumor",
    "non_tumorous": "no_tumor",
    "non_tumor": "no_tumor",
}


def label_for(path: Path, metadata: dict | None = None) -> str | None:
    if metadata:
        raw = (metadata.get("tumor_label") or metadata.get("label") or "").lower().strip()
        key = re.sub(r"[_\s-]+", "_", raw)
        if key in LABELS:
            return LABELS[key]
    code = re.search(r"_(gl|me|pi|nt|no)_", path.stem.lower())
    if code:
        return {"gl": "glioma", "me": "meningioma", "pi": "pituitary", "nt": "no_tumor", "no": "no_tumor"}[code.group(1)]
    for part in reversed(path.parts[:-1]):
        key = re.sub(r"[_\s]+", "_", part.lower().strip())
        if key in LABELS:
            return LABELS[key]
        if key in ("no_tumour", "no_tumor"):
            return "no_tumor"
    return None


def split_for(path: Path, metadata: dict | None = None) -> str | None:
    if metadata and metadata.get("split", "").lower() in {"train", "test", "validation"}:
        return metadata["split"].lower()
    match = re.search(r"(?:^|[_-])(train|test|validation)(?:[_-]|$)", path.stem.lower())
    if match:
        return match.group(1)
    for part in reversed(path.parts[:-1]):
        if part.lower() in {"train", "test", "validation"}:
            return part.lower()
    return None


def stem_key(stem: str) -> str:
    return re.sub(r"[_-](mask|seg|segmentation)$", "", stem.lower())


def canonical_image_digest(path: Path) -> str:
    """Hash decoded RGB pixels so re-encoded copies cannot cross data splits."""
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        digest = hashlib.sha256()
        digest.update(rgb.width.to_bytes(4, "big"))
        digest.update(rgb.height.to_bytes(4, "big"))
        digest.update(rgb.tobytes())
        return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path, help="Folder containing the already extracted dataset.")
    parser.add_argument("--output", type=Path, default=PROJECT / "data" / "manifests" / "brisc.csv")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"Source folder not found: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_by_name: dict[str, dict] = {}
    metadata_file = source / "manifest.csv"
    if metadata_file.is_file():
        with metadata_file.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                key = Path(row.get("filename") or row.get("relative_path") or "").name
                if key:
                    metadata_by_name[key] = row
    mask_files = []
    image_files = []
    for path in source.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        meta = metadata_by_name.get(path.name, {})
        parts = {part.lower() for part in path.parts}
        flagged = str(meta.get("is_mask", "")).lower() in {"1", "true", "yes"}
        is_mask = flagged or any(part in {"mask", "masks", "segmentation_masks", "ground_truth"} for part in parts) or bool(re.search(r"[_-](mask|seg|segmentation)$", path.stem.lower()))
        (mask_files if is_mask else image_files).append(path)
    mask_index: dict[tuple[str | None, str], list[Path]] = {}
    for path in mask_files:
        mask_index.setdefault((label_for(path, metadata_by_name.get(path.name)), stem_key(path.stem)), []).append(path)
    rows = []
    duplicate_hashes: dict[str, list[dict]] = {}
    ignored = 0
    for path in sorted(image_files):
        meta = metadata_by_name.get(path.name)
        label = label_for(path, meta)
        if label is None:
            ignored += 1
            continue
        candidates = mask_index.get((label, stem_key(path.stem)), [])
        if not candidates:
            candidates = mask_index.get((None, stem_key(path.stem)), [])
        mask_path = str(sorted(candidates)[0]) if candidates else ""
        try:
            digest = canonical_image_digest(path)
        except OSError:
            ignored += 1
            continue
        row = {"image_path": str(path), "mask_path": mask_path, "label": label, "split": "", "source_split": split_for(path, meta)}
        duplicate_hashes.setdefault(digest, []).append(row)
    by_label: dict[str, list[list[dict]]] = {}
    for group in duplicate_hashes.values():
        labels = {row["label"] for row in group}
        if len(labels) != 1:
            names = ", ".join(sorted(labels))
            raise SystemExit(f"The same decoded image has conflicting class labels ({names}); inspect the source dataset before training.")
        label = next(iter(labels))
        by_label.setdefault(label, []).append(group)
    rng = random.Random(args.seed)
    assigned = []
    source_splits = {row["source_split"] for groups in by_label.values() for group in groups for row in group}
    if "train" in source_splits and "test" in source_splits:
        train_groups: dict[str, list[list[dict]]] = {}
        for rows in duplicate_hashes.values():
            label = rows[0]["label"]
            test_rows = [row for row in rows if row["source_split"] == "test"]
            validation_rows = [row for row in rows if row["source_split"] == "validation"]
            training_rows = [row for row in rows if row["source_split"] == "train"]
            if test_rows:
                chosen = sorted(test_rows, key=lambda row: (not bool(row["mask_path"]), row["image_path"]))[0]
                assigned.append({**chosen, "split": "test"})
            elif validation_rows:
                chosen = sorted(validation_rows, key=lambda row: (not bool(row["mask_path"]), row["image_path"]))[0]
                assigned.append({**chosen, "split": "validation"})
            elif training_rows:
                chosen = sorted(training_rows, key=lambda row: (not bool(row["mask_path"]), row["image_path"]))[0]
                train_groups.setdefault(label, []).append([chosen])
        for label, groups in train_groups.items():
            rng.shuffle(groups)
            validation_count = max(1, round(len(groups) * .15)) if len(groups) > 1 else 0
            for index, group in enumerate(groups):
                split = "validation" if index < validation_count else "train"
                assigned.extend({**row, "split": split} for row in group)
        print("Using the published train/test split; decoded-pixel duplicates stay in one split, and validation is carved from training only.")
    else:
        for label, groups in by_label.items():
            rng.shuffle(groups)
            train_count = int(len(groups) * .7)
            val_count = int(len(groups) * .15)
            for index, group in enumerate(groups):
                split = "train" if index < train_count else "validation" if index < train_count + val_count else "test"
                assigned.extend({**row, "split": split} for row in group)
        print("No published split found; created a deterministic 70/15/15 image split.")
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_path", "mask_path", "label", "split"])
        writer.writeheader()
        writer.writerows({key: row[key] for key in ("image_path", "mask_path", "label", "split")} for row in assigned)
    print(f"Manifest: {output}")
    print(f"Indexed images: {len(assigned)} | ignored/unlabeled: {ignored} | masks matched: {sum(bool(row['mask_path']) for row in assigned)}")
    print("Split counts: " + ", ".join(f"{name}={sum(row['split'] == name for row in assigned)}" for name in ("train", "validation", "test")))
    print("Warning: BRISC has no patient identifiers; pixel-identical duplicates are grouped, but patient-independent splits cannot be guaranteed.")
    if not assigned:
        raise SystemExit("No labeled images were found. Check the class-folder names and source path.")


if __name__ == "__main__":
    main()
