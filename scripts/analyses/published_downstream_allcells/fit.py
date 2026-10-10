import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import anndata as ad
import numpy as np
import pandas as pd

ROOT=Path.cwd()
OUT=ROOT/'artifacts/paper_evidence/review_round4/published_downstream_allcells'
UNITS=json.loads((OUT/'units.json').read_text())

def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    assert os.environ.get('SLURM_JOB_ID')
    method,index=sys.argv[1],int(sys.argv[2]); base=method.removesuffix('_all_cells')
    u=UNITS[index]; p=ROOT/u['source_dir']; dest=OUT/'fits'/method/u['key']; dest.mkdir(parents=True)
    old=p/'inductive/svd_impute' if base=='svd' else OUT.parent/'reviewer_extras/B_thinning'/u['key']/'alra'
    meta=json.loads((old/'metadata.json').read_text())
    assert meta['parameters']['test_used_for_fit'] is False
    assert meta['seed']==1729 and meta['scale']=='counts'
    script=ROOT/'scripts/analyses/published_downstream_allcells'/f'run_{base}.py'
    common=['--splits',str(p/'splits.parquet'),'--seed',str(meta['seed'])]
    if base=='svd':
        assert meta['parameters']['svd_components']==50
        common+=['--method','svd_impute','--input',str(p/'hybrid.h5ad'),'--coordinates',str(p/'coordinates.parquet'),'--components','50']
    else:
        assert meta['parameters']['power_iterations']==2 and meta['parameters']['quantile_prob']==0.001
        common+=['--corrupted',str(p/'hybrid.h5ad'),'--components','100','--power-iterations','2','--quantile-prob','0.001']
    receipt=dict(method=method,unit=u['key'],job_id=os.environ['SLURM_JOB_ID'],source_contract=str(old),input=str(p/'hybrid.h5ad'),versions={n:importlib.metadata.version(n) for n in ['numpy','scipy','scikit-learn','anndata','pandas']},commands=[])
    started=time.monotonic()
    try:
        a=ad.read_h5ad(p/'hybrid.h5ad',backed='r'); cells=a.obs_names.astype(str).tolist(); genes=a.var_names.astype(str).tolist(); shape=a.shape; a.file.close()
        split=pd.read_parquet(p/'splits.parquet').set_index('cell_id').loc[cells,'split'].to_numpy(); test=split=='test'
        assert meta['cell_ids']==cells and meta['gene_ids']==genes
        receipt['input_sha256']=sha(p/'hybrid.h5ad')
        if base=='svd':assert receipt['input_sha256']==meta['input_sha256']
        replay=dest/'training_replay'
        for target,flag in [(replay,[]),(dest,['--transductive'])]:
            cmd=[str(ROOT/'.venv/bin/python'),str(script),*common,'--output',str(target),*flag]
            receipt['commands'].append(cmd)
            (dest/'attempt.json').write_text(json.dumps(receipt,indent=2)+'\n')
            t=time.monotonic(); subprocess.run(cmd,check=True)
            receipt['allcells_fit_seconds' if flag else 'training_replay_seconds']=time.monotonic()-t
            if not flag:
                previous=np.load(old/'mean.npy',mmap_mode='r'); reproduced=np.load(replay/'mean.npy',mmap_mode='r')

                exact=np.array_equal(previous[test],reproduced[test])
                receipt['training_replay_test_exact']=exact
                receipt['training_replay_full_exact']=np.array_equal(previous,reproduced)
                receipt['training_replay_test_max_difference']=float(np.max(np.abs(previous[test]-reproduced[test])))
                assert exact, receipt
        new=json.loads((dest/'metadata.json').read_text()); values=np.load(dest/'mean.npy',mmap_mode='r')
        assert new['method_version']==meta['method_version'], (new['method_version'],meta['method_version'])
        assert values.shape==shape and np.isfinite(values).all() and (values>=0).all()
        assert new['cell_ids']==cells and new['gene_ids']==genes and new['scale']==meta['scale']
        assert new['parameters']['test_used_for_fit'] is True and new['parameters']['fit_cells']==len(cells)
        assert new['parameters']['heldout_test_cells']==int(test.sum())
        receipt.update(status='completed',shape=list(shape),parameters=new['parameters'],method_version=new['method_version'],test_prediction_changed=bool(np.any(values[test]!=np.load(old/'mean.npy',mmap_mode='r')[test])))
    except Exception as e:
        receipt.update(status='failed',error=repr(e),traceback=traceback.format_exc()); raise
    finally:
        receipt['elapsed_seconds']=time.monotonic()-started
        (dest/'status.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':main()
