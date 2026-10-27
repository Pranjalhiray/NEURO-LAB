# Model card: local research baselines

## Model identities

The application has two independent model slots:

| Task | Active architecture | Input | Output |
|---|---|---|---|
| Classification | MobileNetV3-Small | 2D RGB, resized to 224 × 224 | Four image-level scores: glioma, meningioma, no tumor, pituitary |
| Segmentation | Residual Attention U-Net | 2D RGB, resized to 256 × 256 | One binary tumor-region mask |

No weights ship with the repository. This workspace currently has local trained packages, excluded from source control. The front end and API show a missing/invalid status until a compatible model package is installed. A model package consists of model.json and the weights file named there. Training metadata and test evaluation reports are stored separately.

## Intended use

Education and research engineering: demonstrate a local data flow, compare simple 2D baselines, inspect model readiness, and record actual held-out test metrics.

## Out of scope

Diagnosis, screening, triage, treatment selection, patient monitoring, determining tumor grade, or any clinical decision. The software is not validated or approved as a medical device.

## Data and training

The starter scripts are designed for an extracted BRISC-style folder indexed into a CSV manifest. Splits are deterministic, stratified by class, and group exact decoded RGB pixel duplicates, including copies saved with different encodings. BRISC has no patient identifiers; this is not a patient-independent split and may leave near-duplicate or source-level leakage. The scripts keep validation metrics distinct from final test metrics. Never tune or select a model using the held-out test split.

Classification uses cross-entropy. Segmentation combines weighted binary cross-entropy, soft Dice, and focal Tversky losses. The segmenter uses residual blocks, attention gates, paired spatial/intensity augmentation, validation-only threshold selection, and horizontal-flip test-time averaging. The segmentation mask is a binary region; tumor subtype is not segmented. The workflows use single 2D slices, not volumetric context or multi-sequence MRI.

## Preprocessing

Images are converted to RGB, resized with bilinear interpolation, scaled to [0, 1], and normalized with ImageNet channel means and standard deviations. Segmentation training uses paired flips, mild rotation/scale/translation, and mild contrast/brightness variation. Masks are resized with nearest-neighbor interpolation and binarized. These transformations must match when replacing or retraining weights; metadata records expected input size, normalization, threshold, and test-time averaging setting.

## Limitations and risks

- Public, curated datasets do not represent all scanners, hospitals, acquisition protocols, demographics, pathologies, or image quality conditions.
- No patient-level separation can be guaranteed from BRISC because patient identifiers are absent.
- The 2D pipelines ignore neighboring slices, 3D structure, and other modalities.
- Image-level class scores are not calibrated clinical probabilities.
- Segmentation thresholds and mask overlays are not clinically validated.
- A locally installed compatible checkpoint can still be incorrect, biased, or corrupted. “Ready” only means metadata and weights are present; it is not a quality or safety endorsement.

## Monitoring and reporting

The evaluator reports actual test-split accuracy and macro F1 for classification, and mean per-image Dice/IoU for segmentation. Empty-mask cases count as perfect only if both prediction and truth are empty; that convention is stated in the report. The current evaluator does not calculate confidence intervals, calibration, Hausdorff distance, per-site performance, or external validation.

## Corrected local baseline

The original pre-deduplication checkpoints were trained before decoded-pixel duplicate detection. A local audit found 91 exact pixel-identical image groups that crossed the original test and train/validation partitions. Those original checkpoints are archived and their test scores are invalid for reporting. The clean compact U-Net baseline and the active residual Attention U-Net were both evaluated on the corrected manifest; results are recorded in [LOCAL_BASELINE_RESULTS.md](LOCAL_BASELINE_RESULTS.md). The residual Attention U-Net scored 84.08% mean Dice on the deduplicated BRISC test split, up from 56.07% for the clean compact U-Net baseline. This is an internal dataset comparison, not external validation. BRISC still lacks patient identifiers, so patient-level independence cannot be confirmed.

The classifier result is a dataset-split measurement, not an estimate of independent-patient or external-site performance. Results from the prior checkpoint are preserved only for audit.

## Model package example

Classifier metadata:

~~~json
{
  "task": "classifier",
  "name": "Local MobileNetV3-Small",
  "architecture": "mobilenet_v3_small",
  "input_size": 224,
  "weights_file": "model.pt",
  "classes": ["glioma", "meningioma", "no_tumor", "pituitary"],
  "mean": [0.485, 0.456, 0.406],
  "std": [0.229, 0.224, 0.225]
}
~~~

Segmentation uses task segmenter, architecture attention_unet, input_size 256, the validation-selected threshold recorded in metadata, horizontal_flip_tta, and a model.pt state dictionary.
