"""Safely list or remove temporary project scratch files; never deletes datasets or model weights."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "data" / "scratch"
CACHE = ROOT / "data" / "cache"
CONFIRMATION = "DELETE TEMP DATA"


def contained_files(folder: Path):
    resolved_root = folder.resolve()
    if not resolved_root.is_relative_to(ROOT.resolve()):
        raise RuntimeError(f"Refusing to inspect path outside this project: {folder}")
    if not folder.exists():
        return []
    result = []
    for item in folder.rglob("*"):
        if item.is_file() or item.is_symlink():
            resolved = item.resolve(strict=False)
            if resolved.is_relative_to(resolved_root):
                result.append(item)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Delete listed temporary files.")
    parser.add_argument("--confirm", help=f"Required with --apply; type {CONFIRMATION!r}.")
    parser.add_argument("--include-cache", action="store_true", help="Also list or remove files in data/cache.")
    args = parser.parse_args()
    targets = [SCRATCH] + ([CACHE] if args.include_cache else [])
    files = [(folder, item) for folder in targets for item in contained_files(folder)]
    total = sum(item.stat().st_size for _, item in files if item.exists() and not item.is_symlink())
    print("Dry run — nothing will be deleted." if not args.apply else "Removing the selected temporary files.")
    for _, item in files:
        print(f"{item} ({item.stat().st_size if item.exists() and not item.is_symlink() else 0} bytes)")
    print(f"{len(files)} files, {total} bytes")
    if not args.apply:
        print(f"To delete scratch files, run again with --apply --confirm {CONFIRMATION!r}.")
        return
    if args.confirm != CONFIRMATION:
        raise SystemExit(f"Deletion refused. Supply --confirm {CONFIRMATION!r} after reviewing the dry run.")
    for folder, item in files:
        resolved_root = folder.resolve()
        resolved_item = item.resolve(strict=False)
        if not resolved_item.is_relative_to(resolved_root):
            raise SystemExit(f"Deletion refused for path outside target: {item}")
        item.unlink(missing_ok=True)
    print("Temporary files removed. Dataset and model folders were not touched.")


if __name__ == "__main__":
    main()
