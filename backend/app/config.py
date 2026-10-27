from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_local_environment() -> None:
    """Read simple KEY=VALUE settings from the ignored project .env file."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.is_file():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


_load_local_environment()

ASSET_ROOT = Path(os.getenv("BTDS_ASSET_ROOT", PROJECT_ROOT / "assets")).resolve()
MODEL_ROOT = ASSET_ROOT / "models"
SCRATCH_ROOT = Path(os.getenv("BTDS_SCRATCH_ROOT", PROJECT_ROOT / "data" / "scratch")).resolve()
MAX_UPLOAD_BYTES = int(os.getenv("BTDS_MAX_UPLOAD_BYTES", str(64 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.getenv("BTDS_MAX_IMAGE_PIXELS", "40000000"))
CORS_ORIGINS = tuple(
    origin.strip().rstrip("/")
    for origin in os.getenv(
        "BTDS_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
)
APP_VERSION = "0.1.0"


def configure_cache_paths() -> None:
    """Keep optional ML and imaging caches inside the project data area."""
    cache = PROJECT_ROOT / "data" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
