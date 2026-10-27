"""Show project storage use and free space without changing files."""
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    ROOT / ".venv", ROOT / "frontend" / "node_modules",
    ROOT / "data" / "datasets", ROOT / "data" / "cache",
    ROOT / "data" / "scratch", ROOT / "assets" / "models",
]


def tree_size(path: Path):
    if not path.exists():
        return 0, 0
    files = [item for item in path.rglob("*") if item.is_file()]
    return len(files), sum(item.stat().st_size for item in files)


def pretty(size: int):
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024


print(f"Project: {ROOT}")
for path in TARGETS:
    count, size = tree_size(path)
    print(f"{path.relative_to(ROOT)}: {count} files, {pretty(size)}")
for drive in ("C:\\", "D:\\"):
    try:
        usage = shutil.disk_usage(drive)
        print(f"{drive} free: {pretty(usage.free)} of {pretty(usage.total)}")
    except OSError:
        pass
try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=5, check=True,
    )
    print("GPU: " + result.stdout.strip().replace("\n", "; "))
except (OSError, subprocess.SubprocessError):
    print("GPU: nvidia-smi unavailable")
