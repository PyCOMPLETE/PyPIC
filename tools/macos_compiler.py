"""Select the Homebrew compiler without hard-coding a GCC major version."""
from pathlib import Path
import subprocess
prefix = Path(subprocess.check_output(['brew', '--prefix', 'gcc'], text=True).strip())
candidates = list((prefix / 'bin').glob('gfortran-*'))
compiler = max(candidates, key=lambda p: int(p.name.rsplit('-', 1)[1]))
print(f'CIBW_ENVIRONMENT_MACOS=FC="{compiler}" MACOSX_DEPLOYMENT_TARGET="14.0"')
