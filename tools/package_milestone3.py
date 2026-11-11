"""Package the reproducible Milestone 3 submission, excluding local build state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = (
    "read_the_boss", "tests", "tools", "configs", "docs", ".github",
    "report", "evidence", "experiments/milestone3",
)
ROOT_FILES = ("README.md", ".gitignore", ".gitattributes", "run_demo.py", "MANIFEST.json")
EXCLUDED_PARTS = {"__pycache__", ".git", ".venv", "tmp", "dist", ".pytest_cache"}
EXCLUDED_SUFFIXES = {
    ".pyc", ".pyo", ".aux", ".log", ".out", ".toc", ".fls", ".fdb_latexmk",
}
REQUIRED = (
    "README.md", "read_the_boss/agents.py", "read_the_boss/training.py",
    "read_the_boss/evaluation.py", "read_the_boss/experiments.py",
    "configs/milestone3.json", "tests/test_learning.py",
    "experiments/milestone3/final_summary.json",
    "report/milestone3/milestone3_report.pdf",
    "report/milestone3/milestone3_report.tex",
)


def safe_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", value):
        raise argparse.ArgumentTypeError("Use letters, digits, underscores, or hyphens in names.")
    return value


def source_files() -> list[Path]:
    candidates = [ROOT / name for name in ROOT_FILES]
    candidates.extend(ROOT.glob("requirements*.txt"))
    for name in DIRECTORIES:
        directory = ROOT / name
        if directory.exists():
            candidates.extend(directory.rglob("*"))
    selected = set()
    for path in candidates:
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix in EXCLUDED_SUFFIXES or path.name.endswith(".synctex.gz"):
            continue
        if path.is_symlink():
            raise ValueError(f"Refusing to package a symlink: {relative}")
        if path.is_file():
            selected.add(path)
    return sorted(selected, key=lambda path: path.relative_to(ROOT).as_posix())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--last-name", type=safe_name, default="Xu")
    parser.add_argument("--first-name", type=safe_name, default="Qipan")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing:
        parser.error("Build the official experiment and report first; missing: " + ", ".join(missing))
    files = source_files()
    basename = f"{args.last_name}_{args.first_name}_milestone3"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / f"{basename}.zip"
    manifest = {
        "submission": "Milestone 3: RL Implementation & Initial Results",
        "author": f"{args.first_name} {args.last_name}",
        "repository": "https://github.com/XUQIPAN/RL_Final_Project",
        "description": "SHA-256 hashes of every archive member except this manifest; paths are relative to the extracted submission root.",
        "files": {},
    }
    temporary = destination.with_suffix(".zip.tmp")
    # Fixed ZIP timestamps make an unchanged submission byte-reproducible.
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            payload = path.read_bytes()
            manifest["files"][relative] = {
                "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
            }
            member = zipfile.ZipInfo(f"{basename}/{relative}", date_time=(2026, 11, 7, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o100644 << 16
            archive.writestr(member, payload)
        member = zipfile.ZipInfo(f"{basename}/milestone3_manifest.json", date_time=(2026, 11, 7, 0, 0, 0))
        member.compress_type = zipfile.ZIP_DEFLATED
        member.external_attr = 0o100644 << 16
        archive.writestr(member, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    print(json.dumps({
        "archive": str(destination), "files": len(files) + 1,
        "bytes": destination.stat().st_size,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
