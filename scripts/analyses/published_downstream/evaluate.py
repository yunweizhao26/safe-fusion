import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import shutil

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path.cwd()
OUT = ROOT/'artifacts/paper_evidence/review_round4/published_downstream'
ROUND = OUT.parent
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
from masked_f1_units import COUNT_SCALE, dense
from selector_attribution import exact_topk
from compute_matched_baseline_f1_curves import tie_broken_order
from safefusion_benchmark.contracts import write_output_contract
spec=importlib.util.spec_from_file_location('dense_run', ROOT/'scripts/analyses/fill_grid_dense/run.py')
frozen=importlib.util.module_from_spec(spec); spec.loader.exec_module(frozen)
UNITS=json.loads((OUT/'units.json').read_text())[:8]
OLD=frozen.METHODS
NEW=['saver','dca','scimpute','screcover','enimpute','scgpt']
PCTS=[0.25,0.35,0.5,0.7,1,1.4,2,2.5,3,4,5,7,10,15,20,30,50,70,100]

def evaluate(method,index):
    assert os.environ.get('SLURM_JOB_ID')
    u=UNITS[index]; key=u['key']; p=ROOT/u['source_dir']
    dest=OUT/'evaluation'/method/key
    dest.mkdir(parents=True,exist_ok=True)
    try:
        if method in NEW:
            source=OUT/'fits'/method/key
            status=json.loads((source/'status.json').read_text())
            if status['status']!='completed': raise RuntimeError('Fit failed: '+status.get('error','unknown'))
            source_meta=json.loads((source/'metadata.json').read_text())
            if source_meta['scale'] not in COUNT_SCALE:
                raise ValueError('Unsupported count-scale filling: '+source_meta['scale']+'; Table 1 ranks scGPT scores but defines no count-scale reconstruction. No substitute or conversion used.')
        a=ad.read_h5ad(p/'recorded.h5ad')
        recorded=dense(a.layers['corrupted_counts']).astype(np.float32)
        test=pd.read_parquet(p/'splits.parquet').set_index('cell_id').loc[a.obs_names,'split'].to_numpy()=='test'
        h=ad.read_h5ad(p/'hybrid.h5ad'); counts=dense(h.layers['corrupted_counts']).astype(np.float32)
        np.testing.assert_array_equal(recorded[test],counts[test])
        rows,cols=np.where((recorded==0)&test[:,None])
        pcts=PCTS if method in NEW or method=='svd' else PCTS[13:]
        extra=ROUND/'reviewer_extras/B_thinning'/key
        if method in OLD:
            if method in ['detection','transductive']: source=p/'transductive/safe_fusion'
            elif method=='svd': source=p/'inductive/svd_impute'
            elif method=='alra': source=extra/'alra'
            else: source=p/'transductive'/f'{method.removeprefix("selector_")}_inductive'
            source_meta=json.loads((source/'metadata.json').read_text())
            value=np.load(source/'mean.npy',mmap_mode='r')[rows,cols]
            if method in ['detection','transductive']:
                saved=np.load(p/'transductive/selector/detection.npz')
                np.testing.assert_array_equal(saved['rows'],rows); np.testing.assert_array_equal(saved['cols'],cols)
                score=saved['scores']
                if method=='detection': value=value*saved['detection']
            else: value=np.maximum(value,0); score=value
            if method.startswith('selector_'):
                saved=np.load(extra/method/'test_scores.npz')
                np.testing.assert_array_equal(saved['rows'],rows); np.testing.assert_array_equal(saved['cols'],cols)
                score=saved['score']
            oldroot=extra if method in ['alra','selector_scvi','selector_magic'] else p
            template=json.loads((oldroot/frozen.name(method,1)/'metadata.json').read_text())
        else:
            mean=np.load(source/'mean.npy',mmap_mode='r')

            value=np.maximum(COUNT_SCALE[source_meta['scale']](np.asarray(mean[rows,cols],dtype=np.float64),counts.sum(axis=1,dtype=np.float64)[rows]),0).astype(np.float32)
            score=value
            template=source_meta
        assert source_meta['cell_ids']==a.obs_names.astype(str).tolist()
        assert source_meta['gene_ids']==a.var_names.astype(str).tolist()
        assert np.isfinite(value).all() and np.isfinite(score).all()

        tie_seed=frozen.D[index].get('tie_seed',1729)
        order=tie_broken_order(score,tie_seed) if method in NEW else None
        checks=[]; frames=[]; baseline=None
        for start in range(0,len(pcts),8):
            fractions=pcts[start:start+8]
            work=dest/f'batch_{start//8:02d}'
            work.mkdir(exist_ok=True)
            method_args=[]
            for pct in fractions:
                name=frozen.name(method,pct); target=work/name
                k=max(1,round(pct/100*len(rows)))
                if method in NEW:
                    selected=np.zeros(len(rows),dtype=bool); selected[order[:k]]=True
                else: selected=exact_topk(score,k)
                result=recorded.copy(); result[rows[selected],cols[selected]]=value[selected]
                assert selected.sum()==k
                np.testing.assert_array_equal(result[~test],recorded[~test])
                np.testing.assert_array_equal(result[recorded>0],recorded[recorded>0])
                bitwise=None
                if method=='svd' and pct in [1,5,10]:
                    bitwise=np.array_equal(result,np.load(p/name/'mean.npy',mmap_mode='r')); assert bitwise
                if method=='saver':
                    prior=OUT/'evaluation_attempt1/saver'/key/'batch_00'/name/'mean.npy'
                    if prior.exists():
                        bitwise=np.array_equal(result,np.load(prior,mmap_mode='r')); assert bitwise
                meta=copy.deepcopy(template)
                meta.update(method=name,scale='counts')
                if method in NEW:

                    meta.setdefault('parameters',{}).update(transductive=True,test_used_for_fit=True,fit_cells=int(len(recorded)))
                meta.setdefault('parameters',{})['published_downstream']=dict(source_contract=str(source),fill_fraction=pct/100,tie_rule='Table 1 randomized ties' if method in NEW else 'dense grid exact_topk',tie_seed=tie_seed,all_input_cells=method in NEW)
                write_output_contract(target,result,meta)
                checks.append(dict(method=method,fill_pct=pct,candidates=len(rows),selected=int(selected.sum()),changed=int(np.sum(value[selected]!=0)),saved_fill_bitwise_equal=bitwise))
                method_args+=['--method',f'{name}={target}']
            args=['--truth',p/'truth.h5ad','--corrupted',p/'recorded.h5ad','--splits',p/'splits.parquet',*method_args,'--output-dir',work/'evaluation','--bootstrap',2000,'--seed',1729,'--allow-transductive']
            if key.startswith('pancreas'):
                script=ROOT/'scripts/analyses/transductive_downstream/pancreas_fold.py'; args+=['--coordinates',p/'empty_coordinates.parquet','--fold',key[-1]]
            elif key.startswith('colon'):
                script=Path('scripts/evaluate_colon_donor_biology.py'); args+=['--coordinates',p/'empty_coordinates.parquet','--marker-panel','source']
            elif key=='zebrafish': script=Path('scripts/evaluate_trajectory_preservation.py')
            else:
                script=Path('scripts/evaluate_interventional_grn.py'); args+=['--dataset','norman_crispra','--intervention','gain_of_function','--publication-doi','10.1126/science.aax4438']
            cmd=['.venv/bin/python',str(script),*map(str,args)]
            (work/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
            subprocess.run(cmd,check=True)
            filename='donor_metrics.parquet' if key.startswith(('colon','pancreas')) else 'unit_metrics.parquet'
            frame=pd.read_parquet(work/'evaluation'/filename)
            ref=frame[frame.method.isin(['corrupted_raw','reference_truth'])].reset_index(drop=True)
            if baseline is None: baseline=ref; frames.append(ref)
            else: pd.testing.assert_frame_equal(ref,baseline,check_exact=True)
            frames.append(frame[~frame.method.isin(['corrupted_raw','reference_truth'])])

            for pct in fractions: shutil.rmtree(work/frozen.name(method,pct))
            print(key,method,fractions,'evaluated',flush=True)
        pd.concat(frames,ignore_index=True).to_parquet(dest/filename,index=False)
        pd.DataFrame(checks).to_csv(dest/'fill_checks.csv',index=False)
        (dest/'status.json').write_text(json.dumps(dict(status='completed',method=method,unit=key,fractions=pcts,job_id=os.environ['SLURM_JOB_ID']),indent=2)+'\n')
    except Exception as error:
        (dest/'status.json').write_text(json.dumps(dict(status='failed',method=method,unit=key,error=repr(error),traceback=traceback.format_exc(),job_id=os.environ['SLURM_JOB_ID']),indent=2)+'\n')
        raise

if __name__=='__main__': evaluate(sys.argv[1],int(sys.argv[2]))
