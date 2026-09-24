"""Exclude legacy output and optional backends from release source archives."""
import os
from pathlib import Path
import shutil
root = Path(os.environ["MESON_DIST_ROOT"])
for name in ("other", "testing", "tests", "doc", "GPU", "FPPS"):
    shutil.rmtree(root / name, ignore_errors=True)
for path in root.rglob("*"):
    if path.is_file() and path.suffix in {".so", ".pyc"}:
        path.unlink()
