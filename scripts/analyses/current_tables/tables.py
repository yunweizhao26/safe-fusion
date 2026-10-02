import json, os, sys, subprocess
from pathlib import Path
REPO=Path.cwd()
E=REPO/'artifacts/paper_evidence'
O=E/'review_round4/current_tables'
T=E/'review_round4/transductive_main'
D=E/'review_round4/transductive_downstream'
sys.path[:0]=[str(REPO/'scripts'),str(REPO/'src'),str(T/'code')]
import numpy as np
import pandas as pd
import anndata as ad
import evaluate_value_accuracy as va
import evaluate_inserted_value as iv
from common import units, fusion_root
from thinning import specs
TEACHERS=list(va.TEACHERS)

def link(src,dst):
    src,dst=Path(src),Path(dst)
    assert src.exists(),src
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.is_symlink(): assert dst.resolve()==src.resolve()
    elif not dst.exists(): dst.symlink_to(src.resolve())

def run(script,*args):
    cmd=[str(REPO/'.venv/bin/python'),str(REPO/script),*map(str,args)]
    with (O/'commands.jsonl').open('a') as f: f.write(json.dumps(cmd)+'\n')
    subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)

def parity(new,old,keys,path):
    a,b=pd.read_csv(new),pd.read_csv(old)
    assert len(a)==len(b),(new,len(a),len(b))
    merged=a.merge(b,on=keys,suffixes=('_new','_old'),validate='one_to_one')
    assert len(merged)==len(a)
    errors={}
    for col in a:
        if col not in keys and pd.api.types.is_numeric_dtype(a[col]):
            x,y=merged[col+'_new'].to_numpy(),merged[col+'_old'].to_numpy()
            assert np.allclose(x,y,atol=1e-10,rtol=1e-10,equal_nan=True),(new,col,np.nanmax(abs(x-y)))
            errors[col]=float(np.nanmax(abs(x-y))) if np.isfinite(x-y).any() else 0
    Path(path).write_text(json.dumps(dict(rows=len(a),max_errors=errors),indent=2)+'\n')

def original(part):
    out=O/'original'/part; out.mkdir(parents=True,exist_ok=True)
    if part=='masked':
        reference=E/'review_round2/norman_rebuilt/value_accuracy'
        summary=json.loads((reference/'summary.json').read_text())
        args=[]
        for key,s in summary['units'].items(): args+=['--unit',key,*[s[f] for f in va.UNIT_FIELDS]]
        run('scripts/evaluate_value_accuracy.py',*args,'--bootstrap',2000,'--seed',1729,'--output-dir',out)
        for f,keys in [('log_error.csv',['dataset','value']),('error_removed.csv',['dataset','value','fill_fraction']),('log_error_strata.csv',['dataset','value','stratum_kind','stratum'])]:
            parity(out/f,reference/f,keys,out/(f+'.parity.json'))
    else:
        run('scripts/evaluate_inserted_value.py',part,'--output-dir',out,'--draws',2000,'--seed',1729)
        f='recorded_zero_fills.csv' if part=='recorded' else 'thinning_positive_bias.csv'
        keys=['dataset','method','fill_fraction'] if part=='recorded' else ['dataset','design','positives','recorded_count']
        parity(out/f,E/'review_round2/inserted_value'/f,keys,out/'parity.json')

def model_view(u):
    src=fusion_root(u)/'transductive'/u.key
    dst=O/'models/masked'/u.key
    for name in TEACHERS+['safe_fusion']:
        link(src/{'magic_inductive':'magic','scvi_inductive':'scvi'}.get(name,name),dst/name)
    return dst

def fit_linear(index):
    jobs=[('masked',u,None,None) for u in units()]+[('thinning',u,key,root) for u,key,root,p in specs()]
    kind,u,key,root=jobs[index]
    if kind=='masked':
        dest=model_view(u); inp=u.corrupted; coords=u.coordinates
    else:
        dest=O/'models/thinning'/key
        src=T/'thinning'/key/'mask-trained'
        for name in TEACHERS+['safe_fusion']:
            link(src/{'magic_inductive':'magic','scvi_inductive':'scvi'}.get(name,name),dest/name)
        inp=root/'mask_trained'/key/'input/hybrid.h5ad'; coords=root/'mask_trained'/key/'input/coordinates.parquet'
    args=sum((['--teacher-contract',dest/n] for n in TEACHERS),[])
    run('scripts/run_leakage_safe_method.py','--method','safe_fusion','--input',inp,'--coordinates',coords,'--splits',u.splits,'--output',dest/'safe_fusion_linear','--value-model','linear','--transductive','--seed',1729,*args)

def masked():
    assert (O/'original/masked/log_error.csv.parity.json').exists()
    va.VALUES=tuple(x for x in va.VALUES if not x.startswith('autoencoder'))
    frames=[]
    for u in units():
        va.UNIT_KEYS[u.key]=({'Pancreas':'pancreas','Colon':'colon','CRISPRa':'norman_crispra'}[u.dataset],u.unit_column)
        spec=dict(input=u.corrupted,coordinates=u.coordinates,splits=u.splits,methods_root=model_view(u),selector=D/'masked'/u.key/'transductive/selector')

        va.FRACTIONS=(.01,.05,.10)
        frames.append(va.table_unit(u.key,spec))
    frame=pd.concat(frames,ignore_index=True); rng=np.random.default_rng(1729)
    tables=[[],[],[]]
    for dataset,part in frame.groupby('dataset',sort=True):
        n=part.bio_unit.nunique(); draws=rng.integers(0,n,size=(2000,n))
        for dest,rows in zip(tables,va.dataset_tables(dataset,part,draws)): dest.extend(rows)
    dest=O/'current/masked'; dest.mkdir(parents=True,exist_ok=True)
    for name,rows in zip(['log_error','error_removed','log_error_strata'],tables): pd.DataFrame(rows).to_csv(dest/(name+'.csv'),index=False)

if __name__=='__main__':
    task=sys.argv[1]
    if task=='original': original(sys.argv[2])
    elif task=='linear': fit_linear(int(os.environ['SLURM_ARRAY_TASK_ID']))
    elif task=='masked': masked()
    elif task=="report": run(O/"code/report.py"); run(O/"code/audit.py")
    elif task=="original_print": run(O/"code/original_print.py")
    else: raise ValueError(task)
