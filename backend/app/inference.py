import base64
import io
import threading
from functools import lru_cache

import numpy as np
from PIL import Image

from .config import configure_cache_paths
from .registry import ModelRegistry

registry = ModelRegistry()
CLASSES = ("glioma", "meningioma", "no_tumor", "pituitary")
_cam_lock = threading.RLock()


@lru_cache(maxsize=2)
def _load_model(task: str):
    configure_cache_paths()
    try:
        import torch
        from torchvision.models import mobilenet_v3_small
    except Exception as exc:
        raise RuntimeError(f"PyTorch runtime is unavailable: {exc}") from exc
    info = registry.manifest(task)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    architecture = info["architecture"]
    if task == "classifier" and architecture == "mobilenet_v3_small":
        model = mobilenet_v3_small(weights=None)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, len(info.get("classes", CLASSES)))
    elif task == "segmenter" and architecture in {"compact_unet", "attention_unet"}:
        from .models import build_attention_unet, build_compact_unet
        model = build_attention_unet() if architecture == "attention_unet" else build_compact_unet()
    else:
        raise RuntimeError(f"Unsupported {task} architecture: {architecture}")
    try:
        state = torch.load(info["_weights_path"], map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(info["_weights_path"], map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()
    return model, device, info


def _tensor(image: Image.Image, info: dict):
    import torchvision.transforms.functional as F
    size = info.get("input_size", 224)
    if isinstance(size, list):
        target = (int(size[0]), int(size[1]))
    else:
        target = (int(size), int(size))
    image = image.resize((target[1], target[0]), Image.Resampling.BILINEAR)
    tensor = F.to_tensor(image)
    mean = info.get("mean", [0.485, 0.456, 0.406])
    std = info.get("std", [0.229, 0.224, 0.225])
    return F.normalize(tensor, mean=mean, std=std).unsqueeze(0)


def _png_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _bbox(binary: np.ndarray) -> dict | None:
    ys, xs = np.where(binary)
    if not len(xs):
        return None
    return {"x_min": int(xs.min()), "y_min": int(ys.min()), "x_max": int(xs.max()), "y_max": int(ys.max())}


def _gradcam(model, image_tensor, device, class_index: int, output_size: tuple[int, int]):
    """Higher-resolution Grad-CAM from MobileNetV3-Small's 14x14 feature block."""
    import torch
    import torch.nn.functional as F

    # The final 7x7 block is too coarse for a small 2D lesion. This earlier
    # convolutional block retains 14x14 spatial detail while remaining class-aware.
    layer = model.features[8]
    activations = []
    with _cam_lock:
        handle = layer.register_forward_hook(lambda _module, _inputs, output: activations.append(output))
        try:
            with torch.enable_grad():
                model.zero_grad(set_to_none=True)
                logits = model(image_tensor.to(device))
                feature_map = activations[-1]
                gradients = torch.autograd.grad(logits[0, class_index], feature_map, retain_graph=False)[0]
                weights = gradients.mean(dim=(2, 3), keepdim=True)
                cam = torch.relu((weights * feature_map).sum(dim=1, keepdim=True))
                cam = F.interpolate(cam, size=output_size, mode="bilinear", align_corners=False)[0, 0]
                cam = cam.detach().float().cpu().numpy()
                positive = cam[cam > 1e-8]
                if not positive.size:
                    raise RuntimeError("Grad-CAM was flat for this image; no usable attribution map was produced.")
                scale = float(np.percentile(positive, 99.5))
                if not np.isfinite(scale) or scale <= 1e-8:
                    raise RuntimeError("Grad-CAM could not be normalized for this image.")
                return np.clip(cam / scale, 0, 1).astype(np.float32)
        finally:
            handle.remove()


def _heatmap_png(cam: np.ndarray) -> Image.Image:
    values = np.clip(cam, 0, 1)
    red = np.clip(1.5 - np.abs(4 * values - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * values - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * values - 1), 0, 1)
    rgb = np.stack([red, green, blue], axis=-1)
    alpha = np.where(values >= .16, 210, 0)[..., None]
    rgba = np.concatenate([np.round(rgb * 255).astype(np.uint8), alpha.astype(np.uint8)], axis=-1)
    return Image.fromarray(rgba, mode="RGBA")


def analyze_image(image: Image.Image) -> dict:
    statuses = {status.task: status for status in registry.statuses()}
    result = {
        "classification": None, "segmentation": None, "explanation": None,
        "source_png_base64": _png_base64(image), "warnings": [],
    }
    class_cam = None
    for task in ("classifier", "segmenter"):
        if statuses[task].status != "ready":
            result["warnings"].append(f"{task}: {statuses[task].detail}")
            continue
        try:
            model, device, info = _load_model(task)
            x = _tensor(image, info)
            import torch
            if task == "classifier":
                with torch.inference_mode():
                    logits = model(x.to(device))[0]
                    scores = torch.softmax(logits, dim=0).cpu().numpy()
                names = info.get("classes", list(CLASSES))
                predicted_index = int(scores.argmax())
                ordered_scores = {name: float(scores[index]) for index, name in enumerate(names)}
                top_two = sorted((float(score) for score in scores), reverse=True)[:2]
                result["classification"] = {
                    "label": names[predicted_index], "scores": ordered_scores,
                    "score_semantics": "uncalibrated softmax model score; not a calibrated probability",
                    "score_margin": top_two[0] - top_two[1] if len(top_two) > 1 else None,
                    "model_name": info.get("name", "MobileNetV3-Small"),
                    "checkpoint": info.get("weights_file", "model.pt"),
                }
                try:
                    cam = _gradcam(model, x, device, predicted_index, (image.height, image.width))
                    class_cam = cam
                    binary_cam = cam >= .35
                    result["explanation"] = {
                        "method": "Grad-CAM",
                        "target_class": names[predicted_index],
                        "target_score": float(scores[predicted_index]),
                        "heatmap_png_base64": _png_base64(_heatmap_png(cam)),
                        "region_bbox": _bbox(binary_cam),
                        "region_threshold": 0.35,
                        "peak_activation": float(cam.max()) if cam.size else 0.0,
                        "supporting_evidence": "Grad-CAM highlights broad image regions that influenced the selected class score. It is a coarse attribution map, not a tumor boundary, diagnosis, or explanation of biological cause.",
                    }
                except Exception as exc:
                    result["warnings"].append(f"classification explanation unavailable: {exc}")
            else:
                with torch.inference_mode():
                    from .models import predict_segmentation_probability
                    probability = predict_segmentation_probability(
                        model, x.to(device), bool(info.get("horizontal_flip_tta", False)),
                    )[0, 0].float().cpu().numpy()
                    threshold = float(info.get("threshold", 0.5))
                probability_image = Image.fromarray(np.round(np.clip(probability, 0, 1) * 255).astype(np.uint8), mode="L")
                probability_image = probability_image.resize(image.size, Image.Resampling.BILINEAR)
                probability_array = np.asarray(probability_image).astype(np.float32) / 255.0
                preview_scale = min(1.0, 1024 / max(image.size))
                preview_size = tuple(max(1, round(dimension * preview_scale)) for dimension in image.size)
                probability_preview = probability_image.resize(preview_size, Image.Resampling.BILINEAR)
                mask_array = probability_array >= threshold
                mask_image = Image.fromarray(mask_array.astype(np.uint8) * 255, mode="L")
                positive_pixels = int(mask_array.sum())
                colored_mask = np.zeros((image.height, image.width, 4), dtype=np.uint8)
                colored_mask[mask_array] = (225, 72, 83, 255)
                mask_overlay = Image.fromarray(colored_mask, mode="RGBA")
                bbox = _bbox(mask_array)
                result["segmentation"] = {
                    "mask_png_base64": _png_base64(mask_image),
                    "mask_overlay_png_base64": _png_base64(mask_overlay),
                    "probability_png_base64": _png_base64(probability_preview),
                    "threshold": threshold,
                    "positive_pixels": positive_pixels,
                    "image_pixels": int(image.width * image.height),
                    "coverage_percent": 100.0 * positive_pixels / max(1, image.width * image.height),
                    "bounding_box": bbox,
                    "model_name": info.get("name", "2D U-Net segmenter"),
                    "checkpoint": info.get("weights_file", "model.pt"),
                    "mask_semantics": "binary predicted tumor-region mask; pixel coverage is not a severity measure",
                }
        except Exception as exc:
            result["warnings"].append(f"{task}: inference failed: {exc}")

    if result["explanation"] and result["segmentation"] and class_cam is not None:
        cam_region = class_cam >= .35
        mask_region = np.asarray(Image.open(io.BytesIO(base64.b64decode(result["segmentation"]["mask_png_base64"]))).convert("L")) > 0
        intersection = int(np.logical_and(cam_region, mask_region).sum())
        union = int(np.logical_or(cam_region, mask_region).sum())
        result["explanation"]["segmentation_overlap_iou"] = intersection / union if union else 1.0
        result["explanation"]["segmentation_overlap_pixels"] = intersection
        result["explanation"]["segmentation_overlap_note"] = "Overlap describes spatial agreement between two model outputs, not anatomical validation."
    return result
