import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
ROOT=Path.cwd()
O=ROOT/'artifacts/paper_evidence/review_round4/table2_added'
R4=O.parent
R3=R4.parent/'review_round3/comparators'
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'scripts/standard_imputers'),str(ROOT/'src')]
import evaluate_knockdown_zero_analyses as kd
from disease_control_common import dense, stratified_draws
from masked_f1_units import COUNT_SCALE

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
sex=module('sex_bootstrap',ROOT/'scripts/analyses/label_baselines/sex/evaluate.py')
A=R4/'round2_extras/part_a'
VIEW=R4/'transductive_references/knockdown/evaluation_inputs'
ARGS=argparse.Namespace(deploy_root=str(VIEW/'deployment'),review_root=str(VIEW/'review'),norman_root=str(VIEW/'norman'),norman_benchmark=str(R4.parent/'review_round2/leakage_free/norman_crispra'))
UNITS=['adamson_crispri','norman_crispra']+[f'{t}_{i}' for t in ['pancreas','colon'] for i in range(3)]

def base(unit):
    return kd.screen_inputs(unit,ARGS).deploy if unit in UNITS[:2] else R4/'transductive_references/sex_zeros/deployment_rebuilt'/unit

def legacy_base(unit):
    if unit=='norman_crispra': return R4.parent/'review_round2/knockdown/norman_rebuilt/deployment'
    if unit.startswith('pancreas'): return R4.parent/'downstream_deployment/pancreas'/f'fold_{unit[-1]}'
    return R4.parent/'downstream_deployment'/unit

def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n')

def contract(unit,method):
    if method=='SAVER':
        return kd.screen_inputs(unit,ARGS).standard['saver'] if unit in UNITS[:2] else R4/'sex_zero_comparators/fits/saver'/unit
    if method=='scRecover':
        audit=json.loads((O/'audit'/f'{unit}.json').read_text())
        if audit['reuse']: return R3/'fits/screcover'/f'deploy_{unit}'
    return O/'fits'/method.lower()/unit

def get_scores(path,a,rows,cols,library,method):
    m=json.loads((path/'metadata.json').read_text())
    assert m['cell_ids']==list(a.obs_names) and m['gene_ids']==list(a.var_names)
    v=np.load(path/'mean.npy',mmap_mode='r')
    assert v.shape==a.shape
    s=np.asarray(v[rows,cols],dtype=np.float64)
    if method=='scGPT':
        assert m['scale']=='frozen_scgpt_masked_value_score' and m['seed']==1729
        assert m['parameters']['checkpoint_sha256']==json.loads((O/'audit/checkpoint_sha256.json').read_text())['best_model.pt']
        assert m['parameters']['max_observed_sequence_length']==1200
    else:
        s=np.maximum(COUNT_SCALE[m['scale']](s,library[rows]),0)
        if method=='SAVER': s=s.astype(np.float32).astype(float)
    assert np.isfinite(s).all()
    return s
