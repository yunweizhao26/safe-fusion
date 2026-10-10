import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path.cwd()
OUT = ROOT / 'artifacts/paper_evidence/review_round4/published_downstream'
UNITS = json.loads((OUT / 'units.json').read_text())

def main():
    assert os.environ.get('SLURM_JOB_ID')
    method, index = sys.argv[1], int(sys.argv[2])
    unit = UNITS[index]
    dest = ROOT / unit['out'] / ('kcluster' if method == 'kcluster' else 'fits') / method / unit['key']
    dest.mkdir(parents=True, exist_ok=True)
    inp, split, coord = [str(ROOT / unit[k]) for k in ('input', 'splits', 'coordinates')]
    common = ['--corrupted', inp, '--splits', split, '--output', str(dest)]
    cores = os.environ['SLURM_CPUS_PER_TASK']
    if method == 'kcluster':
        cmd = ['.venv-scanpy/bin/python', 'scripts/standard_imputers/r3_kcluster.py', '--corrupted', inp, '--output', str(dest / 'clusters.json'), '--seed', '1729']
    elif method == 'saver':
        cmd = ['.venv/bin/python', 'scripts/run_saver_baseline.py', *common, '--ncores', cores]
    elif method == 'alra':
        cmd = ['.venv/bin/python', 'scripts/analyses/transductive_comparators/run_alra_baseline.py', *common, '--components', '100', '--power-iterations', '2', '--quantile-prob', '0.001', '--seed', '1729', '--transductive']
    elif method == 'scgpt':
        cmd = ['.venv-baselines/bin/python', 'scripts/run_scgpt_gene_prediction.py', '--corrupted', inp, '--coordinates', coord, '--output', str(dest), '--ckpt-dir', str(ROOT / 'external_data/baselines/scgpt_human'), '--batch-size', '64', '--gene-batch-size', '512', '--max-length', '1200', '--device', 'cuda', '--seed', '1729']
    else:
        cmd = ['.venv/bin/python', 'scripts/standard_imputers/run_r3_comparator.py', '--method', method, *common, '--coordinates', coord, '--kcluster-json', str(ROOT / unit['out'] / 'kcluster/kcluster' / unit['key'] / 'clusters.json'), '--ncores', cores, '--seed', '1729']
    receipt = dict(unit=unit, method=method, command=cmd, job_id=os.environ['SLURM_JOB_ID'], all_input_cells=True)
    (dest / 'attempt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    started = time.monotonic()
    try:
        subprocess.run(cmd, check=True)
        receipt['status'] = 'completed'
        if method != 'kcluster':
            import anndata as ad
            import numpy as np
            a = ad.read_h5ad(inp, backed='r')
            meta = json.loads((dest/'metadata.json').read_text())
            value = np.load(dest/'mean.npy', mmap_mode='r')
            assert value.shape == a.shape
            assert meta['cell_ids'] == a.obs_names.astype(str).tolist()
            assert meta['gene_ids'] == a.var_names.astype(str).tolist()
            assert np.isfinite(value).all(), f"Non-finite {method} output; matched_genes={meta.get('parameters',{}).get('matched_genes','not applicable')}"
            receipt.update(shape=list(a.shape), scale=meta['scale'], method_version=meta['method_version'], parameters=meta['parameters'])

            receipt['actual_input_cells'] = a.n_obs
            receipt['input_sha256'] = hashlib.file_digest(open(inp,'rb'), 'sha256').hexdigest()
            a.file.close()
    except Exception as error:
        receipt.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        receipt['elapsed_seconds'] = time.monotonic()-started
        (dest/'status.json').write_text(json.dumps(receipt, indent=2)+'\n')

if __name__ == '__main__':
    main()
