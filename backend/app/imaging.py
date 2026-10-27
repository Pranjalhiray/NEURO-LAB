from pathlib import Path
import io
import tempfile

from PIL import Image, ImageOps

from .config import MAX_IMAGE_PIXELS, MAX_UPLOAD_BYTES, SCRATCH_ROOT
from .schemas import ScanInspection

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
NIFTI_EXTENSIONS = {".nii", ".gz"}


def _extension(name: str) -> str:
    lower = name.lower()
    return ".nii.gz" if lower.endswith(".nii.gz") else Path(lower).suffix


def inspect_scan(filename: str, content: bytes) -> tuple[ScanInspection, Image.Image | None]:
    if not content:
        raise ValueError("The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError(f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.")
    ext = _extension(filename)
    if ext in IMAGE_EXTENSIONS:
        try:
            with Image.open(io.BytesIO(content)) as image:
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("Multi-page image files are not supported.")
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Image dimensions exceed the safe processing limit.")
                image.load()
                normalized = ImageOps.exif_transpose(image).convert("RGB")
                return ScanInspection(
                    filename=Path(filename).name, input_type="2D image",
                    width=normalized.width, height=normalized.height,
                    note="Single 2D image. Orientation metadata was applied when present.",
                ), normalized
        except (OSError, Image.DecompressionBombError) as exc:
            raise ValueError("Could not read this image. Use a valid JPG or PNG scan.") from exc
    if ext == ".nii.gz" or ext == ".nii":
        try:
            import nibabel as nib
        except ImportError as exc:
            raise ValueError("NIfTI support is unavailable. Install the optional imaging dependencies.") from exc
        SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
        suffix = ".nii.gz" if ext == ".nii.gz" else ".nii"
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=SCRATCH_ROOT, suffix=suffix, delete=False) as tmp:
                temp_path = Path(tmp.name)
                tmp.write(content)
            image = nib.load(str(temp_path), mmap=False)
            shape = tuple(int(n) for n in image.shape)
            if len(shape) < 3 or any(n < 1 for n in shape[:3]):
                raise ValueError("NIfTI volume must contain at least three non-empty dimensions.")
            modality = image.header.get("descrip", b"")
            if isinstance(modality, bytes):
                modality = modality.decode("utf-8", errors="ignore").strip("\x00 ")
            return ScanInspection(
                filename=Path(filename).name, input_type="NIfTI volume",
                width=shape[0], height=shape[1], slices=shape[2],
                modality=modality or None,
                note="Volume metadata only. Inference and slice review require a compatible model and viewer.",
            ), None
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("Could not read this NIfTI volume.") from exc
        finally:
            if temp_path:
                temp_path.unlink(missing_ok=True)
    raise ValueError("Unsupported file type. Upload a 2D JPG/PNG or a NIfTI volume.")
