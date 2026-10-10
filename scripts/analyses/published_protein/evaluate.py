import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

import anndata as ad
import numpy as np
import pandas as pd

ROOT=Path.cwd(); OUT=ROOT/'artifacts/paper_evidence/review_round4/published_protein'; ROUND=OUT.parent
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import evaluate_protein_within_state as e
from masked_f1_units import COUNT_SCALE, NATIVE_SCALES, dense
spec=importlib.util.spec_from_file_location('s20',ROOT/'scripts/analyses/protein_s20_rows/evaluate.py')
s20=importlib.util.module_from_spec(spec); spec.loader.exec_module(s20)
UNITS=json.loads((ROUND/'published_downstream/units.json').read_text())

def main(method,block):
    assert os.environ.get('SLURM_JOB_ID')
    dest=OUT/'evaluation'/method/block; dest.mkdir(parents=True,exist_ok=True)
    try:
        frame=pd.read_parquet(ROUND/f'protein_s20_rows/{block}/pdl1_cells.parquet')
        assert len(frame)==(655 if block=='small' else 4083) and frame.cell_id.is_unique
        frame['target_baseline']=frame.target_baseline_clr
        unit=next(u for u in UNITS if u['analysis']=='protein' and u['key']==block)
        a=ad.read_h5ad(ROOT/unit['input'])
        rows=a.obs_names.get_indexer(frame.cell_id.astype(str)); assert (rows>=0).all()
        np.testing.assert_array_equal(rows,frame.cell_index.to_numpy(dtype=int))
        split=pd.read_parquet(ROOT/unit['splits']).set_index('cell_id').loc[frame.cell_id,'split']
        assert (split=='test').all()
        column=a.var_names.get_loc('CD274'); counts=dense(a.layers['corrupted_counts']).astype(np.float32)
        assert (counts[rows,column]==0).all()
        if method=='existing':
            names=list(e.RANKINGS)
            if block=='full': names+=['svd_all_cells','weighted_knn_all_cells','magic']
            names+=['p_nonzero']
            if block=='small':names+=['magic']
        else:
            contract=OUT/'fits'/method/block
            status=json.loads((contract/'status.json').read_text())
            if status['status']!='completed':raise RuntimeError('Fit failed: '+status.get('error','unknown'))
            meta=json.loads((contract/'metadata.json').read_text())
            assert meta['cell_ids']==a.obs_names.astype(str).tolist()
            assert meta['gene_ids']==a.var_names.astype(str).tolist()
            values=np.asarray(np.load(contract/'mean.npy',mmap_mode='r')[rows,column],dtype=np.float64)
            scale=meta['scale']
            if scale not in NATIVE_SCALES: values=COUNT_SCALE[scale](values,counts.sum(axis=1,dtype=np.float64)[rows])
            frame[method]=values.astype(np.float32)
            names=['safe_fusion',method]
        assert np.isfinite(frame[names].to_numpy()).all()
        frame.to_parquet(dest/'pdl1_cells.parquet',index=False)
        associations=[]; paired=[]; se_rows=[]
        for number,(scope,subset,resampling) in enumerate([
            ('all',frame,'target_clusters'),('all',frame,'cells_within_replicate'),
            ('control',frame.loc[frame.target==e.CONTROL],'cells_within_replicate')]):
            group_names,groups=np.unique(subset.target.astype(str),return_inverse=True)
            rankings=tuple(n for n in names if not(scope=='control' and n=='target_baseline'))
            job=e.Job(protein='PDL1',gene='CD274',scale='clr',cells='recorded_rna_zero',scope=scope,resampling=resampling,
                rankings=rankings,scores=subset[list(rankings)].to_numpy(dtype=float),response=subset.adt_clr.to_numpy(dtype=float),
                groups=groups,replicate=np.unique(subset.replicate.astype(str),return_inverse=True)[1],
                covariates={n:subset[n].to_numpy(dtype=float) for n in e.CONTINUOUS_COVARIATES},within=resampling=='target_clusters',
                covariate_sets=e.CONTROL_COVARIATE_SETS if scope=='control' else e.COVARIATE_SETS if resampling=='target_clusters' else {},
                seed=(1729,list(e.PROTEIN_TO_GENE).index('PDL1'),0,0,number),draws=2000,group_names=list(group_names))

            calls=[]; original=e.statistics
            def capture(*args):
                result=original(*args); calls.append(result); return result
            e.statistics=capture
            try: association,difference=e.run_job(job)
            finally:e.statistics=original
            associations.append(association); paired.append(difference)
            if number==0:
                assert len(calls)==2001
                samples=pd.DataFrame(calls[1:])
                for (stat,cov,ranking),estimate in calls[0].items():
                    if cov or stat not in ('pooled_spearman','within_target_spearman'):continue
                    draws=samples[(stat,cov,ranking)].to_numpy(dtype=float)
                    se_rows.append(dict(block=block,statistic=stat,ranking=ranking,estimate=estimate,se=float(np.nanstd(draws,ddof=1)),
                        ci_low=float(np.nanquantile(draws,.025)),ci_high=float(np.nanquantile(draws,.975)),defined_draws=int(np.isfinite(draws).sum())))
        association=pd.concat(associations,ignore_index=True); pair=pd.concat(paired,ignore_index=True)
        association.to_csv(dest/'association.csv',index=False); pair.to_csv(dest/'paired_differences.csv',index=False)
        pd.DataFrame(se_rows).to_csv(dest/'estimates.csv',index=False)
        saved=pd.read_csv(ROUND/f'protein_s20_rows/{block}/association.csv')
        if method!='existing': saved=saved[saved.ranking=='safe_fusion']
        checks=s20.compare(saved,association,'ranking',['estimate','ci_low','ci_high'])
        checks.to_csv(dest/'reproduction_association.csv',index=False)
        if method=='existing':
            check=s20.compare(pd.read_csv(ROUND/f'protein_s20_rows/{block}/paired_differences.csv'),pair,'comparator',['safe_fusion_minus_comparator','ci_low','ci_high'])
            check.to_csv(dest/'reproduction_paired.csv',index=False)
        (dest/'status.json').write_text(json.dumps(dict(status='completed',method=method,block=block,fit_cells=a.n_obs,genes=a.n_vars,
            evaluated_zeros=len(frame),control_cells=int((frame.target==e.CONTROL).sum()),target_groups=int(frame.target.nunique()),
            bootstrap_draws=2000,seed=1729,safe_fusion_reproduction=True,max_reproduction_difference=float(checks.absolute_difference.max()),
            job_id=os.environ['SLURM_JOB_ID']),indent=2)+'\n')
    except Exception as error:
        (dest/'status.json').write_text(json.dumps(dict(status='failed',method=method,block=block,error=repr(error),traceback=traceback.format_exc(),job_id=os.environ['SLURM_JOB_ID']),indent=2)+'\n')
        raise

if __name__=='__main__': main(*sys.argv[1:])
