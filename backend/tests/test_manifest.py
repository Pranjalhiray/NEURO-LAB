import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.prepare_brisc_manifest import main


class ManifestTests(unittest.TestCase):
    def test_preserves_published_test_split_and_matches_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_image = root / "classification" / "train" / "brisc2025_train_00001_gl_ax_t1.jpg"
            test_image = root / "classification" / "test" / "brisc2025_test_00001_gl_ax_t1.jpg"
            train_mask = root / "segmentation" / "masks" / "brisc2025_train_00001_gl_ax_t1.png"
            test_mask = root / "segmentation" / "masks" / "brisc2025_test_00001_gl_ax_t1.png"
            for path, color in ((train_image, 60), (test_image, 90)):
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (12, 12), (color, color, color)).save(path, format="JPEG")
            for path in (train_mask, test_mask):
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("L", (12, 12), 255).save(path, format="PNG")
            manifest = root / "manifest.csv"
            rows = []
            for path, split, is_mask in (
                (train_image, "train", "false"), (test_image, "test", "false"),
                (train_mask, "train", "true"), (test_mask, "test", "true"),
            ):
                rows.append({
                    "filename": path.name, "relative_path": path.relative_to(root).as_posix(),
                    "task": "segmentation" if is_mask == "true" else "classification",
                    "split": split, "tumor_label": "glioma", "is_mask": is_mask,
                })
            with manifest.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            output = root / "prepared.csv"
            import sys
            old_argv = sys.argv
            try:
                sys.argv = ["prepare_brisc_manifest", "--source", str(root), "--output", str(output)]
                main()
            finally:
                sys.argv = old_argv
            with output.open(newline="", encoding="utf-8") as stream:
                data = list(csv.DictReader(stream))
            self.assertEqual(sum(row["split"] == "test" for row in data), 1)
            test_row = next(row for row in data if row["split"] == "test")
            self.assertIn("brisc2025_test_", Path(test_row["image_path"]).name)
            self.assertTrue(Path(test_row["mask_path"]).is_file())
            self.assertEqual({row["split"] for row in data}, {"train", "test"})


if __name__ == "__main__":
    unittest.main()
