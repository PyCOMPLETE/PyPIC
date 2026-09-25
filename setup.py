"""Legacy `setup.py build_ext -i` compatibility; pip uses meson-python."""
from pathlib import Path
import runpy
import sys

if __name__ == '__main__':
    if sys.argv[1:] not in (['build_ext', '-i'], ['build_ext', '--inplace']):
        raise SystemExit('Use python -m pip install . (or setup.py build_ext -i for a legacy editable build).')
    runpy.run_path(str(Path(__file__).resolve().parent / 'tools/legacy_build.py'), run_name='__main__')
