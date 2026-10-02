import json
from pathlib import Path
O=(Path.cwd() / 'artifacts/paper_evidence/review_round4/round2_extras/part_c')
E=Path('artifacts/paper_evidence'); R=E/'review_round4'; C=E/'review_round3/comparators'
base=json.loads((R/'transductive_comparators/units_manifest.json').read_text())
comparators={u['key']:u for p in [C/'units_manifest.json',C/'colon_crossfit_units.json'] for u in json.loads(p.read_text())}
rows=[]
for seed in range(1729,1734):
    for u in base:
        key=u['key']; r=R/'transductive_main/rebuilt_replicates'/f'{key}_s{seed}'
        v=u.copy() if seed==1729 else json.loads((r/'unit.json').read_text())[0]
        f=E/('review_round3/colon_crossfit/fusion_value' if u['dataset']=='Colon' else 'review_round2/fusion_value')
        dest=O/f'{key}_s{seed}'
        v['seed']=seed; v['base_key']=key; v['destination']=str(dest)
        v['reference']=str((f/'selectors'/key if seed==1729 else r/'scores'/v['key'])/'safe_fusion_transductive/test_scores.npy')
        v['extras']={}
        for method,slug in [('SVD (all-cell)','svd_impute'),('Weighted kNN (all-cell)','graph_smooth')]:
            v['extras'][method]=u['contracts'][method.replace('all-cell','transductive')] if seed==1729 else str(r/'transductive'/slug)
        for method in ['EnImpute','DCA']:
            v['extras'][method]=comparators[key]['contracts'][method] if seed==1729 else str(dest/'fits16'/method.lower())
        v['extras']['scVI probability']=str(f/'nonzero_probability/scvi'/key) if seed==1729 else str(dest/'probability')
        v['selectors']={m:str(C/'stacked'/key/'enimpute/test_scores.npz') if m=='EnImpute' and seed==1729 else str(f/'selectors'/key/f'{m.lower()}_stacked/test_scores.npy') if seed==1729 else str(dest/'selectors'/m.lower()/'test_scores.npz') for m in ['EnImpute','scVI','MAGIC']}
        if seed!=1729: v['selectors']['EnImpute']=str(dest/'selectors/enimpute16/test_scores.npz')
        for path in [v['corrupted'],v['coordinates'],v['splits'],v['reference']]: assert Path(path).exists(),path
        if seed==1729:
            for path in [*v['selectors'].values(),*[str(Path(p)/'mean.npy') for p in v['extras'].values()]]: assert Path(path).exists(),path
        rows.append(v)
(O/'units.json').write_text(json.dumps(rows,indent=2)+'\n')
print(len(rows),'units')
