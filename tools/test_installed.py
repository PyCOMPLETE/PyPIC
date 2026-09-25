"""Run tests away from the source package to prevent import shadowing."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

source = Path(__file__).resolve().parents[1] / 'packaging_tests'
with tempfile.TemporaryDirectory(prefix='installed-package-tests-') as tmp:
    shutil.copytree(source, Path(tmp) / 'tests')
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env['MPLBACKEND'] = 'Agg'
    raise SystemExit(subprocess.call([sys.executable, '-m', 'pytest', 'tests', '-q',
        '--import-mode=importlib', *sys.argv[1:]], cwd=tmp, env=env))
