# BTDS Research Workstation

A local-first research prototype for brain MRI image classification and binary tumor-mask segmentation. It is not a clinical device and does not diagnose disease.

## Current state

- The React interface and FastAPI service are implemented.
- The classification and segmentation pipelines are separate and load independent local model packages.
- The patient session begins with name, date of birth, email, and contact number. Age is calculated from date of birth. The shared MRI workbench is available to patients and administrators.
- The patient PDF includes the patient details, the model's suggested finding and type, where the image overlay appears, and the original and highlighted images.
- Administrator sign-in unlocks both the shared MRI workbench and researcher/judge evaluation. Patient sessions can access only the MRI workbench.
- Administrators also have an activity dashboard with in-memory sign-in and scan counts, model result summaries, mask/Grad-CAM status, and recent run details. It does not retain names, contact details, filenames, or uploaded images, and its counters reset when the API restarts.
- The evaluation page shows held-out summary metrics and no per-case test list. Its API requires an administrator session.
- BRISC 2025 v3 is indexed locally under `data\datasets\brisc2025`; the dataset and model files are ignored by Git.
- Model weights and evaluation reports are local workspace artifacts excluded from Git. Previous checkpoints were archived after decoded-pixel-identical images were found across train/validation and test; corrected checkpoints and fresh metrics are being generated from the cleaned manifest.
- The patient workflow accepts 2D JPG, PNG, TIFF, and BMP images. Administrators can preview NIfTI volumes locally; the trained models process 2D images only.
- Patient details remain in browser memory for the session and are sent to the local API only when creating the requested PDF. The patient details form is not identity verification or a permanent account.
- Evaluation metrics appear only after a held-out test-set evaluation has been run and saved. Administrators can rerun evaluation; **Export JSON** saves the recorded metrics.

## Run it in VS Code on Windows

1. In VS Code, choose **File → Open Folder…** and open `D:\BTS`.
2. If prompted, select `D:\BTS\.venv\Scripts\python.exe` as the Python interpreter. The workspace setting already points to it.
3. On a fresh checkout, copy `.env.example` to `.env` and replace the admin password with a long random password. This workspace already has a local admin password configured.
4. Open the Command Palette with **Ctrl+Shift+P**, choose **Tasks: Run Task**, then run **BTDS API**.
5. Repeat **Tasks: Run Task** and run **BTDS Web** in a second terminal.
6. Open [http://127.0.0.1:5173](http://127.0.0.1:5173). Use **Ctrl+C** in each task terminal to stop its server.

You can also use two integrated PowerShell terminals. Open them with **Terminal → New Terminal** and run these commands from the project root:

API terminal:

   ~~~powershell
   .\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
   ~~~

Web terminal:

   ~~~powershell
   cd frontend
   npm run dev
   ~~~

For a fresh checkout, install dependencies once from the project root:

   ~~~powershell
   py -3.14 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
   cd frontend
   npm install
   cd ..
   ~~~

The first dependency install needs internet access. If the Python launcher does not offer Python 3.14, use a Python version with a matching Windows PyTorch wheel.

After both development servers are running, choose **Patient** to enter name, date of birth, email, and contact number; age is calculated from date of birth. This patient form creates a local session and is not identity verification or a permanent account. Choose **Administrator** to sign in with the credentials in `.env`. Both roles have the same 2D MRI workbench controls for the original scan, mask overlay, mask-only view, Grad-CAM, comparison, opacity, and advanced mask threshold/region filtering. The patient result explains the model score as uncalibrated, describes the limits of Grad-CAM, and offers general brain-tumor cause and risk information from the National Cancer Institute; none of these outputs is a diagnosis or a personal cause assessment. Administrators additionally have model evaluation, NIfTI volume preview, and the **Activity & results** dashboard at [http://127.0.0.1:5173/activity](http://127.0.0.1:5173/activity). Patient sessions are local, unverified sessions. Evaluation may take a few minutes on CPU.

## Dataset and model training

BRISC is the selected research dataset because it provides image labels, segmentation masks, and a published test split. Version 3 is available from [Mendeley Data](https://data.mendeley.com/datasets/c7dj4pwhcv/3) under CC BY 4.0. The extracted release is already present in this workspace at `data\datasets\brisc2025`; dataset files are excluded from Git. For another checkout, download and extract the release into that folder.

Create a manifest (the indexer does not download or copy images):

~~~powershell
python scripts\prepare_brisc_manifest.py --source data\datasets\brisc2025
~~~

The indexer reads BRISC's manifest when present, recognizes its published train/test split and filename labels, groups images with identical decoded RGB pixels (including files saved with different encodings), and carves validation data from the training portion only. For other class-folder datasets, it creates a deterministic 70/15/15 image split. It matches masks by class and image basename. Inspect the printed image, mask, and split counts before training; unmatched tumor images are omitted from segmentation training. No-tumor images without a mask are treated as negative examples.

This workspace uses the official CUDA 13.0 PyTorch wheels with Python 3.14 and an NVIDIA GPU. To set up the same training environment in a fresh checkout, install PyTorch first, then the ML requirements:

~~~powershell
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r backend\requirements-ml.txt
~~~

For a different GPU or CPU, choose a compatible build from the [official PyTorch selector](https://pytorch.org/get-started/locally/).

Model weights and training records are local artifacts. To train a fresh model package, activate the environment and run these commands after preparing the manifest:

~~~powershell
cd D:\BTS
.\.venv\Scripts\Activate.ps1
python -m backend.training.train_classifier --epochs 10 --batch-size 16
python -m backend.training.train_segmenter --epochs 20 --patience 5 --batch-size 16 --input-size 256 --workers 2 --output assets/models/segmenter-candidate
python -m backend.training.evaluate --task classifier
python -m backend.training.evaluate --task segmenter
~~~

The active segmenter is a residual Attention U-Net trained with paired geometric/intensity augmentation, a Dice/Tversky-weighted objective, validation-only threshold selection, and horizontal-flip test-time averaging. It scored 84.08% mean Dice on the held-out BRISC test split, up from 56.07% for the previous compact U-Net. Reduce the batch size if training runs out of GPU memory. The command above writes a new candidate package separately; review its validation history before installing it into `assets\models\segmenter`, restart the API, and then evaluate. Evaluation reads only the manifest's test split and writes an actual-results report under `assets\evaluations`. See [the local segmentation results](docs/LOCAL_BASELINE_RESULTS.md) for details and the test-set limitations.

The corrected manifest keeps decoded-pixel-identical copies in one split. It contains 5,856 unique images: 4,134 train, 730 validation, and 992 test, with 4,659 matched tumor masks. A local audit found 91 exact decoded-image groups spanning the original test and train/validation partitions; the train/validation copies are now excluded. BRISC does not provide patient identifiers, so patient-independent separation and all near-duplicate leakage cannot be ruled out. Results apply only to this dataset split and cannot guarantee correct predictions on scans from another source, scanner, or patient.

## Models and privacy

Drop model.json and its named weights file into assets\models\classifier or assets\models\segmenter. Metadata must name the supported architecture, task, input size, and weights filename. See the model card before training or loading weights.

2D image uploads are processed by the local API and are not kept after the request. Larger multipart uploads may briefly spool to data\scratch on D:. The interface does not maintain a case history. Patient details exist in browser memory until the session ends; when a patient summary is requested, the details are sent to the local API to generate that PDF. Administrator credentials are stored in the ignored local `.env` file; admin tokens expire after eight hours and are held only in API memory.

Keep patient-identifiable information out of this prototype. It is not hardened, validated, or approved for clinical use.

## Documentation

- Research review: docs/RESEARCH_REVIEW.md
- Local baseline results: docs/LOCAL_BASELINE_RESULTS.md
- Model card and limitations: docs/MODEL_CARD.md
- Cloud training notes: docs/CLOUD_TRAINING.md
- GitHub and Render deployment guide: docs/GITHUB_RENDER_DEPLOYMENT.md
- Storage inventory and cleanup: docs/STORAGE.md
