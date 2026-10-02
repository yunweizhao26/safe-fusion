import os
from pathlib import Path
import runpy
import sys
import scvi

scvi.settings.logging_dir = str(Path(os.environ['SCALE_ROOT']) / 'scvi_logs' / os.environ['SLURM_JOB_ID'])
script = sys.argv.pop(1)
sys.path.insert(0, str(Path(script).resolve().parent))
runpy.run_path(script, run_name='__main__')
