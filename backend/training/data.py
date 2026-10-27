import csv
import hashlib
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]
LABEL_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}


def load_rows(csv_path: Path, split: str) -> list[dict]:
    root = Path(__file__).resolve().parents[2]
    rows = []
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row.get("split") != split:
                continue
            item = dict(row)
            item["image_path"] = str((root / item["image_path"]).resolve()) if not Path(item["image_path"]).is_absolute() else item["image_path"]
            item["mask_path"] = str((root / item["mask_path"]).resolve()) if item.get("mask_path") and not Path(item["mask_path"]).is_absolute() else item.get("mask_path", "")
            if not Path(item["image_path"]).is_file():
                raise FileNotFoundError(f"Missing image listed in manifest: {item['image_path']}")
            rows.append(item)
    return rows


class ClassificationDataset(Dataset):
    def __init__(self, rows: list[dict], train: bool, size: int = 224):
        self.rows = rows
        ops = [transforms.Resize((size, size))]
        if train:
            ops += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(8)]
        ops += [transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])]
        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.open(row["image_path"]).convert("RGB")
        label = row["label"]
        if label not in LABEL_TO_INDEX:
            raise ValueError(f"Unknown class label: {label}")
        return self.transform(image), LABEL_TO_INDEX[label]


class SegmentationDataset(Dataset):
    def __init__(self, rows: list[dict], train: bool, size: int = 256):
        self.rows = [row for row in rows if row.get("mask_path") or row.get("label") == "no_tumor"]
        ops = [transforms.Resize((size, size))]
        self.image_transform = transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        self.mask_resize = transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST)
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.open(row["image_path"]).convert("RGB")
        mask_path = Path(row["mask_path"]) if row.get("mask_path") else None
        if mask_path and not mask_path.is_file():
            raise FileNotFoundError(f"Missing mask listed in manifest: {mask_path}")
        mask = Image.open(mask_path).convert("L") if mask_path else Image.new("L", image.size, 0)
        if self.train:
            if random.random() < .5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            if random.random() < .25:
                image, mask = TF.vflip(image), TF.vflip(mask)
            angle = random.uniform(-12, 12)
            translate = (round(image.width * random.uniform(-.035, .035)), round(image.height * random.uniform(-.035, .035)))
            scale = random.uniform(.92, 1.08)
            image = TF.affine(image, angle, translate, scale, shear=(0.0, 0.0), interpolation=InterpolationMode.BILINEAR, fill=0)
            mask = TF.affine(mask, angle, translate, scale, shear=(0.0, 0.0), interpolation=InterpolationMode.NEAREST, fill=0)
            image = TF.adjust_contrast(image, random.uniform(.85, 1.15))
            image = TF.adjust_brightness(image, random.uniform(.90, 1.10))
        image = self.image_transform(image)
        mask = self.mask_resize(mask)
        mask = transforms.functional.pil_to_tensor(mask).float().div_(255)
        mask = (mask >= 0.5).float()
        return image, mask


def deterministic_split(rows: list[dict], seed: int = 17) -> list[dict]:
    """Group exact duplicate files; split each label into 70/15/15 partitions."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        digest = hashlib.sha256(Path(row["image_path"]).read_bytes()).hexdigest()
        groups.setdefault((row["label"], digest), []).append(row)
    by_label: dict[str, list[list[dict]]] = {}
    for (label, _), group in groups.items():
        by_label.setdefault(label, []).append(group)
    rng = random.Random(seed)
    assigned = []
    for label in CLASSES:
        label_groups = by_label.get(label, [])
        rng.shuffle(label_groups)
        n = len(label_groups)
        train_n = int(n * .7)
        val_n = int(n * .15)
        for index, group in enumerate(label_groups):
            split = "train" if index < train_n else "validation" if index < train_n + val_n else "test"
            for row in group:
                assigned.append({**row, "split": split})
    return assigned
