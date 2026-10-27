import importlib.util
import json
from pathlib import Path

from .config import MODEL_ROOT
from .schemas import ModelStatus

TASKS = ("classifier", "segmenter")
ARCHITECTURES = {"classifier": {"mobilenet_v3_small"}, "segmenter": {"compact_unet", "attention_unet"}}


class ModelRegistry:
    def __init__(self, root: Path = MODEL_ROOT):
        self.root = root.resolve()

    def status(self, task: str) -> ModelStatus:
        if task not in TASKS:
            raise ValueError("Unknown model task.")
        folder = self.root / task
        manifest = folder / "model.json"
        if not manifest.is_file():
            return ModelStatus(task=task, status="missing", detail="Model metadata has not been installed.")
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            if data.get("task") != task or data.get("architecture") not in ARCHITECTURES[task]:
                raise ValueError("task or architecture does not match a supported pipeline")
            size = data.get("input_size")
            if not isinstance(size, (int, list)) or (isinstance(size, int) and not 1 <= size <= 1024):
                raise ValueError("input_size is invalid")
            if not isinstance(data.get("weights_file"), str) or not data["weights_file"]:
                raise ValueError("weights_file is missing")
            if task == "classifier":
                classes = data.get("classes")
                if (not isinstance(classes, list) or len(classes) != 4
                        or any(not isinstance(name, str) or not name for name in classes)
                        or len(set(classes)) != 4):
                    raise ValueError("classifier metadata must contain four unique class names")
            weights = (folder / data["weights_file"]).resolve()
            weights.relative_to(self.root)
            if not weights.is_file():
                return ModelStatus(task=task, status="missing", model_name=data.get("name"), detail="Model weights are missing.")
            runtime_missing = importlib.util.find_spec("torch") is None
            if task == "classifier":
                runtime_missing = runtime_missing or importlib.util.find_spec("torchvision") is None
            if runtime_missing:
                return ModelStatus(
                    task=task, status="runtime_unavailable", model_name=data.get("name"),
                    detail="Model files are present, but the compatible PyTorch runtime is not installed.",
                    weights_size_bytes=weights.stat().st_size,
                )
            return ModelStatus(
                task=task, status="ready", model_name=data.get("name", task),
                detail="Weights and metadata are present; runtime compatibility is checked at inference.",
                weights_size_bytes=weights.stat().st_size,
            )
        except Exception as exc:
            return ModelStatus(task=task, status="invalid", detail=f"Model metadata is invalid: {exc}")

    def statuses(self) -> list[ModelStatus]:
        return [self.status(task) for task in TASKS]

    def manifest(self, task: str) -> dict:
        folder = (self.root / task).resolve()
        folder.relative_to(self.root)
        data = json.loads((folder / "model.json").read_text(encoding="utf-8"))
        path = (folder / data["weights_file"]).resolve()
        path.relative_to(self.root)
        return {**data, "_weights_path": path}
