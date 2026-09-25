"""Compatibility entrypoint: compile via the supported editable pip build."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-build-isolation',
                '--editable', str(root)], check=True)
