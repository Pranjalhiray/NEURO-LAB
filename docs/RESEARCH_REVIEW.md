# Research review and implementation choices

This note summarizes the five supplied papers and explains which parts informed this prototype. Published metrics below belong to the cited papers and are not measurements from this project.

## Supplied papers

### Brain Tumor Detection Based on Deep Learning Approaches and MRI (2023)

Abdusalomov et al. describe an adapted YOLOv7 object detector with CBAM attention, an SPPF+ block, BiFPN, and a decoupled detection head. The paper reports a four-class, 10,288-image dataset and 99.5% accuracy with 99.3% precision and recall. The task is bounding-box detection, not pixel-mask segmentation. The paper does not establish patient-level separation, and its dataset description does not make subject independence auditable. It therefore supports the distinction between detection and segmentation, but is not a segmentation recipe or a performance target for this app.

Reference: Abdusalomov et al., Cancers 2023, 15, 4172. DOI: 10.3390/cancers15164172.

### Improved clustering and machine-learning classifier (2024)

Bhimavarapu et al. combine improved fuzzy C-means segmentation, contrast enhancement, handcrafted GLCM/shape/texture/color features, and an extreme learning machine classifier. The paper uses Figshare and Kaggle datasets with different class balances and no-tumor images only in the Kaggle set. Reported figures vary between the abstract, results, and conclusion, including a 96.6% correct-prediction statement despite higher overall claims. A clear independent test protocol and standard overlap measures such as Dice/IoU are not established. These inconsistencies make its headline performance unsuitable as a benchmark.

Reference: Bhimavarapu et al., Bioengineering 2024, 11, 266. DOI: 10.3390/bioengineering11030266.

### Brain Tumor Classification and Segmentation with EfficientNet and U-Net++ (2026)

Alkharaan et al. evaluate classification and segmentation on BRISC2025. They report 6,000 2D T1 images, four image-level classes, and 4,793 paired binary masks. EfficientNet-B1 classification is reported at 99.70% accuracy on 1,000 test images; U-Net++ with an EfficientNet-B1 encoder is reported at Dice 0.9055 and IoU 0.8442. Their classifier validation procedure differs from the segmentation split. The experiments are single-run and 2D, use a fixed public dataset, and do not establish multi-site clinical performance. The paper is the most directly relevant supplied dataset/method reference, but its metrics are not adopted as this project's results.

Reference: Alkharaan et al., Diagnostics 2026, 16, 1745. DOI: 10.3390/diagnostics16101745.

### A brain tumor segmentation enhancement using U-Net and transfer learning (2025)

Pourmahboubi et al. study 120 lower-grade glioma patients from TCIA/TCGA-LGG using FLAIR slices and a VGG19-encoder U-Net with a Focal Tversky loss. The paper reports Dice/F1 0.9679, IoU 0.9378, and recall 0.9821. The method is 2D and single-modality. Patient-level split details are not sufficiently clear for leakage assessment. Its metric table also contains internal inconsistencies, including Dice/F1 pairs that do not agree. The reported high scores are therefore treated cautiously, not used as expected outcomes.

Reference: Pourmahboubi et al., BMC Medical Imaging 2025, 25:307. DOI: 10.1186/s12880-025-01837-4.

### Brain tumor segmentation and detection with CNNs and VGG16 (2025)

Ganesh et al. has substantial internal inconsistencies: the abstract discusses autonomous-vehicle segmentation and mIoU prediction, the methods describe a small Whole Brain Atlas/Kaggle dataset, and other sections cite conflicting dataset sizes, splits, model families, and results. VGG16, YOLO variants, GoogleNet, AlexNet, ANN, SVM, and self-organizing maps are mixed across the narrative. The paper is not used to define the pipeline or establish performance.

Reference: Ganesh et al., Cancer Biomarkers 2025, 42(3). DOI: 10.1177/18758592241311184.

## Dataset choice

BRISC is the practical starting point for separate 2D classification and binary segmentation experiments. The official release describes 6,000 contrast-enhanced T1 2D images across glioma, meningioma, no-tumor, and pituitary classes, plus 4,793 paired binary masks. The Mendeley record lists manifests and checksums and the release is CC BY 4.0. The downloaded version 3 archive did not contain those top-level manifest/checksum files, so this project reconstructed its manifest from the published folder split and filename labels, then grouped byte-identical images. The source data is derived from multiple public collections and has no patient identifiers; patient-independent evaluation cannot be guaranteed. The source publication positions it for research benchmarking, not direct clinical diagnosis.

Sources: [BRISC dataset paper](https://www.nature.com/articles/s41597-026-06753-y), [BRISC version 3 data record](https://data.mendeley.com/datasets/c7dj4pwhcv/3).

The full BraTS2020 Kaggle listing is about 7.96 GB of NIfTI data. It is not downloaded to this workstation. Consider it only for an explicitly planned cloud experiment with a separate storage and compute budget. [BraTS2020 dataset listing](https://www.kaggle.com/awsaf49/brats2020-training-data/tasks).

## Project choices

- Keep four-class classification separate from binary tumor-mask segmentation. An image label, a bounding box, and a pixel mask answer different questions.
- Start with MobileNetV3-Small at 224 pixels and a compact 2D U-Net at 256 pixels. These are lightweight baseline choices for a 4 GB GPU, not claims of clinical or paper-equivalent performance.
- Keep NIfTI viewing local in the browser and provide metadata inspection only in the API. The included trained pipelines expect 2D inputs; they do not invent slice-level or 3D results.
- Record independent training and held-out test metrics from this project. Never copy published paper metrics into the app as local results.
- A future 3D experiment needs an explicitly budgeted cloud workflow and an auditable subject-level split. The supplied evidence does not justify a clinical-use claim.
