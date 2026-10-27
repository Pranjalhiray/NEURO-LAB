import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from backend.app.inference import _heatmap_png
from backend.training.evaluate import _binary_auc, classification_metrics, segmentation_metrics


class _ImageScores(torch.nn.Module):
    def forward(self, values):
        return values


class _MaskEcho(torch.nn.Module):
    def forward(self, values):
        return torch.where(values > 0, 10.0, -10.0)


class EvaluationMetricTests(unittest.TestCase):
    def test_gradcam_heatmap_uses_visible_rgb_values_and_transparency(self):
        heatmap = _heatmap_png(torch.tensor([[0., .5, 1.]]).numpy()).convert("RGBA")
        pixels = [heatmap.getpixel((x, 0)) for x in range(3)]
        self.assertEqual(pixels[0][3], 0)
        self.assertEqual(pixels[2][3], 255)
        self.assertGreater(max(pixels[2][:3]), 100)

    def test_binary_auc_uses_rank_order_and_handles_ties(self):
        self.assertEqual(_binary_auc([0, 0, 1, 1], [.1, .4, .35, .8]), .75)
        self.assertEqual(_binary_auc([0, 1], [.5, .5]), .5)
        self.assertIsNone(_binary_auc([1, 1], [.2, .8]))

    def test_classifier_reports_macro_metrics_and_class_order(self):
        logits = torch.tensor([
            [8., 0., 0., 0.], [0., 8., 0., 0.],
            [0., 0., 8., 0.], [0., 0., 0., 8.],
        ])
        labels = torch.arange(4)
        loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
        result = classification_metrics(_ImageScores(), loader, "cpu")
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["macro_f1"], 1.0)
        self.assertEqual(result["roc_auc_ovr_macro"], 1.0)
        self.assertEqual(result["class_order"], ["glioma", "meningioma", "no_tumor", "pituitary"])

    def test_classifier_rejects_mismatched_manifest_labels(self):
        loader = DataLoader(TensorDataset(torch.zeros((1, 4)), torch.zeros(1, dtype=torch.long)))
        with self.assertRaisesRegex(ValueError, "must match the BRISC manifest classes"):
            classification_metrics(_ImageScores(), loader, "cpu", ["glioma", "other", "no_tumor", "pituitary"])

    def test_segmentation_reports_aggregate_scores_and_empty_convention(self):
        masks = torch.zeros((2, 1, 2, 2))
        masks[1, 0, 0, 0] = 1
        loader = DataLoader(TensorDataset(masks, masks), batch_size=2)
        result = segmentation_metrics(
            _MaskEcho(), loader, "cpu", rows=[{"label": "no_tumor"}, {"label": "glioma"}],
        )
        self.assertEqual(result["dice"], 1.0)
        self.assertEqual(result["iou"], 1.0)
        self.assertNotIn("per_case", result)
        self.assertIn("Empty/empty cases score 1.0", result["empty_mask_convention"])


if __name__ == "__main__":
    unittest.main()
