import subprocess
from pathlib import Path

root = Path.cwd()
(root / 'logs').mkdir(exist_ok=True)
subprocess.run(['sbatch', '--wait', '-A', 'torch_pr_634_general', '-p', 'cs',
                'scripts/analyses/main_tables_v3/run.sh'], check=True)
