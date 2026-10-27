# Local baseline experiment

## Data and split

- Dataset: [BRISC 2025, version 3](https://data.mendeley.com/datasets/c7dj4pwhcv/3), licensed CC BY 4.0.
- Processing: exact decoded RGB pixel duplicates are grouped before assigning rows; the published train/test split is preserved and validation is carved only from training.
- Prepared counts: 5,856 unique labeled images total; 4,134 train, 730 validation, and 992 test. The manifest contains 4,659 matched tumor masks.
- A local audit found 91 exact decoded-image groups crossing original train/validation and test partitions. Their train/validation copies have been removed from the corrected manifest. The previous checkpoints and metrics are archived and must not be presented as current results.
- BRISC does not supply patient identifiers. Exact-pixel grouping does not prove subject-level separation or exclude non-identical near-duplicate leakage.

## Corrected training and held-out results

The clean classifier was trained for 10 epochs (batch 16, learning rate 0.0005, seed 17, CUDA); its best validation accuracy was 93.97%. The compact U-Net segmenter baseline was trained for 12 epochs (batch 4, learning rate 0.0005, seed 17, CUDA); its best validation Dice was 61.56%.

### Classifier held-out results

| Test images | Accuracy | Macro precision | Macro recall | Macro F1 | Macro ROC-AUC (OvR) |
|---:|---:|---:|---:|---:|---:|
| 992 | 92.34% | 91.33% | 93.13% | 91.88% | 98.63% |

| Class | Precision | Recall | F1 | Support | ROC-AUC (OvR) |
|---|---:|---:|---:|---:|---:|
| Glioma | 92.40% | 96.05% | 94.19% | 253 | 99.28% |
| Meningioma | 96.02% | 80.33% | 87.48% | 300 | 96.90% |
| No tumor | 82.32% | 97.12% | 89.11% | 139 | 98.62% |
| Pituitary | 94.59% | 99.00% | 96.74% | 300 | 99.72% |

Macro classification metrics are unweighted; multiclass ROC-AUC is one-vs-rest. This result is limited to the deduplicated BRISC test split and does not establish patient-level independence or external performance.

### Segmenter held-out comparison

The active residual Attention U-Net was trained for 20 epochs maximum (batch 16, input 256, seed 29, CUDA, paired image/mask augmentation, validation early stopping). The best checkpoint was epoch 19, with validation Dice 86.47% and threshold 0.45. That threshold was chosen from validation data only. Horizontal-flip test-time averaging was enabled for validation and test inference.

| Segmenter checkpoint | Test images | Mean per-image Dice | Mean per-image IoU |
|---|---:|---:|---:|
| Compact U-Net baseline | 992 | 56.07% | 48.65% |
| Residual Attention U-Net | 992 | **84.08%** | **77.19%** |
| Change | — | **+28.00 percentage points** | **+28.55 percentage points** |

| Class | Baseline Dice | Attention U-Net Dice | Attention U-Net IoU | Support |
|---|---:|---:|---:|---:|
| Glioma | 34.15% | **70.20%** | 60.14% | 253 |
| Meningioma | 47.91% | **87.98%** | 82.59% | 300 |
| No tumor | 92.09% | **98.56%** | 98.56% | 139 |
| Pituitary | 66.05% | **85.17%** | 76.28% | 300 |

Segmentation scores are unweighted means over test images. Empty-mask cases score 1 only when both target and prediction are empty; a non-empty prediction against an empty target scores 0. The active checkpoint was selected and thresholded using validation only, then evaluated once on the held-out test split. The previous compact U-Net report is archived at `assets\evaluations\archive\segmenter-compact-unet-baseline-2026-10-21.json`; its model package is archived at `assets\models\archive\compact-unet-baseline-2026-10-21\`. Reports store aggregate and per-class metrics, not a per-scan test-case list.

## Saved artifacts

- Classifier metadata, weights, and training record: `assets/models/classifier/`
- Segmenter metadata, weights, and training record: `assets/models/segmenter/`
- Test reports: `assets/evaluations/classifier.json` and `assets/evaluations/segmenter.json`
- Prepared manifest: `data/manifests/brisc.csv`

The dataset, weights, and evaluation outputs are local workspace artifacts and are excluded from Git. The committed source includes the code and instructions needed to recreate them after obtaining the dataset.

## Interpretation

These results measure only the supplied BRISC split and these 2D research models. The classifier's scores are uncalibrated model scores. The mask is a binary region and does not distinguish tumor types. Neither pipeline uses 3D context or multiple MRI sequences. BRISC has no patient identifiers, and performance on different hospitals, scanners, or data sources has not been established. This software and these metrics are not for diagnosis, screening, triage, or treatment decisions.
