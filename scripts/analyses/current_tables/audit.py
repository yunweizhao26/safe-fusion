from tables import *
import hashlib
checks=[]
def record(label,condition,**details):
    checks.append(dict(check=label,passed=bool(condition),**details))
    assert condition,(label,details)
for p in (O/'original').rglob('*parity.json'):
    record(str(p.relative_to(O)),True,receipt=json.loads(p.read_text()))
printed=json.loads((O/'printed_value_checks.json').read_text())
record('all original printed cells reproduced',all(c['passed'] for c in printed),rows=len(printed),cells=sum(len(c['printed']) for c in printed))
heldouts={}
for u in units():
    src=fusion_root(u)/'selectors'/u.key/'safe_fusion_transductive/test_scores.npy'
    downstream=D/'masked'/u.key/'transductive/selector/detection.npz'
    score=np.load(src)
    current=np.load(downstream)['scores']
    record('S12/S13 selection score parity '+u.key,np.array_equal(score,current),candidates=len(score))
    record('S26 score parity '+u.key,json.loads((O/'current/rule/masked'/u.key/'score_parity.json').read_text())['exact'])
    data=ad.read_h5ad(u.corrupted)
    splits=pd.read_parquet(u.splits).set_index('cell_id').loc[data.obs_names.astype(str),'split'].to_numpy()
    g=set(data.obs.loc[splits=='test',u.unit_column].astype(str))
    if u.dataset in ['Pancreas','Colon']:
        old=heldouts.setdefault(u.dataset,set())
        record('disjoint held-out donors '+u.key,not (old&g),donors=len(g))
        old.update(g)
    for kind in ['masked']:
        for model in TEACHERS+['safe_fusion','safe_fusion_linear']:
            p=O/'models'/kind/u.key/model
            m=json.loads((p/'metadata.json').read_text())
            record('count scale/order '+u.key+'/'+model,m['scale']=='counts' and m['cell_ids']==data.obs_names.astype(str).tolist() and m['gene_ids']==data.var_names.astype(str).tolist())
    recorded=ad.read_h5ad(T/'recorded'/u.key/'input/recorded.h5ad')
    hybrid=ad.read_h5ad(T/'recorded'/u.key/'input/hybrid.h5ad')
    a,b=iv.dense(recorded.layers['corrupted_counts']),iv.dense(hybrid.layers['corrupted_counts'])
    record('unmasked held-out recorded counts '+u.key,np.array_equal(a[splits=='test'],b[splits=='test']))
record('pancreas donor total',len(heldouts['Pancreas'])==24,donors=len(heldouts['Pancreas']))
record('colon donor total',len(heldouts['Colon'])==34,donors=len(heldouts['Colon']))
for name,expected in [('masked/log_error.csv',21),('masked/error_removed.csv',63),('thinning/bias.csv',32),('rule/summary.csv',9)]:
    f=pd.read_csv(O/'current'/name)
    record('output rows '+name,len(f)==expected,rows=len(f))
fresh=pd.read_csv(O/'current/thinning/bias.csv')
saved=pd.read_csv(T/'value_accuracy/thinning.csv')
saved=saved[(saved.model=='transductive')&(saved.design=='mask-trained')]
for _,r in fresh[fresh.value=='safe_fusion'].iterrows():
    stat='bias_all' if r.scope=='all positives' else 'bias_filled_5pct'
    ref=saved[(saved.dataset==r.dataset)&(saved.statistic==stat)&(saved.value=='conditional')]
    record('saved boosted thinning parity '+r.dataset+'/'+stat,len(ref)==1 and np.allclose(r[['estimate','lower','upper']].to_numpy(float),ref.iloc[0][['estimate','lower','upper']].to_numpy(float),atol=1e-10,rtol=0))
fresh=pd.read_csv(O/'current/masked/log_error.csv')
saved=pd.read_csv(T/'value_accuracy/masked.csv')
for _,r in fresh[fresh.value=='safe_fusion'].iterrows():
    name={'colon':'Colon','pancreas':'Pancreas','norman_crispra':'CRISPRa'}[r.dataset]
    ref=saved[(saved.model=='transductive')&(saved.dataset==name)&(saved.statistic=='log_error')]
    record('saved boosted masked parity '+name,len(ref)==1 and np.isclose(r.log_error,ref.iloc[0].estimate,atol=1e-10,rtol=0))
source_names=['evaluate_value_accuracy.py','evaluate_inserted_value.py','evaluate_thinning_transfer.py','v2_value_evaluate.py','calibrated_selective_fill.py','run_leakage_safe_method.py','run_autoencoder_fusion.py','apply_fill_fraction.py','selector_attribution.py']
hashes={str(REPO/'scripts'/n):hashlib.sha256((REPO/'scripts'/n).read_bytes()).hexdigest() for n in source_names}
for n in source_names:
    record('source unchanged '+n,(REPO/'scripts'/n).read_bytes()==(O/'source_snapshot'/n).read_bytes())
hashes.update({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [E/'review_round2/leakage_free/units_manifest.json',E/'review_round3/colon_crossfit/units_manifest.json',O/'original_supplementary_results.tex']})
(O/'audit.json').write_text(json.dumps(dict(passed=True,checks=checks,sha256=hashes,limitations=['Autoencoder runners cannot accept transductive inputs unchanged','No Dixit transductive deployment chain'],execution_notes=json.loads((O/'execution_notes.json').read_text()),dependency_recovery=(O/'dependency_recovery.txt').read_text()),indent=2)+'\n')
print('AUDIT PASSED',len(checks),'checks')
