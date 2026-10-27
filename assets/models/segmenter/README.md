# Active segmentation package

This folder contains the local residual Attention U-Net checkpoint used by the 2D MRI workbench.

- Architecture: `attention_unet`
- Input: RGB image resized to 256 × 256
- Output: binary tumor-region mask; the mask does not identify tumor type
- Threshold: selected on the validation split and recorded in `model.json`
- Inference: horizontal-flip test-time averaging enabled
- Held-out BRISC 2025 test result: 84.08% mean Dice over 992 images

This is a research model, not a diagnostic tool. Results on other datasets, scanners, or hospitals have not been established. See [the model card](../../../docs/MODEL_CARD.md) and [the evaluation notes](../../../docs/LOCAL_BASELINE_RESULTS.md) for limitations and per-class results.
