import io
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from backend.app.imaging import inspect_scan
from backend.app.registry import ModelRegistry


def png_bytes():
    stream = io.BytesIO()
    Image.new("RGB", (32, 24), (80, 100, 120)).save(stream, format="PNG")
    return stream.getvalue()


class ImagingTests(unittest.TestCase):
    def test_inspects_2d_image_without_writing_upload(self):
        scan, image = inspect_scan("sample.png", png_bytes())
        self.assertEqual(scan.width, 32)
        self.assertEqual(scan.height, 24)
        self.assertEqual(scan.input_type, "2D image")
        self.assertEqual(image.mode, "RGB")

    def test_rejects_unsupported_extension(self):
        with self.assertRaisesRegex(ValueError, "Unsupported file type"):
            inspect_scan("scan.pdf", b"pdf")

    def test_rejects_empty_upload(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            inspect_scan("scan.png", b"")

    def test_rejects_multipage_tiff(self):
        stream = io.BytesIO()
        Image.new("RGB", (8, 8)).save(stream, format="TIFF", save_all=True, append_images=[Image.new("RGB", (8, 8))])
        with self.assertRaisesRegex(ValueError, "Multi-page"):
            inspect_scan("scan.tiff", stream.getvalue())


class RegistryTests(unittest.TestCase):
    def test_missing_model_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = ModelRegistry(Path(tmp)).status("classifier")
        self.assertEqual(status.status, "missing")

    def test_reports_ready_only_with_metadata_and_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "classifier"
            folder.mkdir()
            (folder / "weights.pt").write_bytes(b"test-checkpoint")
            (folder / "model.json").write_text(json.dumps({
                "task": "classifier", "architecture": "mobilenet_v3_small",
                "input_size": 224, "weights_file": "weights.pt",
                "classes": ["glioma", "meningioma", "no_tumor", "pituitary"],
            }))
            status = ModelRegistry(root).status("classifier")
        expected = "ready" if importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision") else "runtime_unavailable"
        self.assertEqual(status.status, expected)
        self.assertEqual(status.weights_size_bytes, len(b"test-checkpoint"))

    def test_rejects_weights_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "classifier"
            folder.mkdir()
            (folder / "model.json").write_text(json.dumps({
                "task": "classifier", "architecture": "mobilenet_v3_small",
                "input_size": 224, "weights_file": "../../outside.pt",
                "classes": ["glioma", "meningioma", "no_tumor", "pituitary"],
            }))
            status = ModelRegistry(root).status("classifier")
        self.assertEqual(status.status, "invalid")


if __name__ == "__main__":
    unittest.main()
