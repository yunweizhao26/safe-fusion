import json,hashlib,re
from pathlib import Path
import numpy as np
import pandas as pd
O=(Path.cwd() / 'artifacts/paper_evidence/review_round4/round2_extras')
hashes=json.loads((O/'source_sha256.json').read_text())
current={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in hashes}
changed=[p for p,h in hashes.items() if current[p]!=h]
presentation={'scripts/plot_overview_f1.py','safe_fusion_overleaf/bioinformatics/6_results.tex'}
assert not set(changed)-presentation,changed
drift={p:dict(initial_sha256=hashes[p],final_sha256=current[p],mtime_ns=Path(p).stat().st_mtime_ns) for p in changed}
(O/'external_source_drift.json').write_text(json.dumps(drift,indent=2)+'\n')
A=json.loads((O/'part_a/verification.json').read_text()); assert A['all_65_source_values_reproduced']
a=pd.read_csv(O/'part_a/table2_aurocs.csv'); assert len(a)==65
for c in ['fill_order','fill_lower','fill_upper','continuous','continuous_lower','continuous_upper','tied_pair_share','unfilled_zero_share']:
    assert np.isfinite(a[c]).all() and a[c].between(0,1).all(),c
B=json.loads((O/'part_b/verification.json').read_text()); assert B['unchanged_records_verbatim']
assert json.loads((O/'part_b/table1_parity.json').read_text())['exact_at_saved_precision']
C=json.loads((O/'part_c/verification.json').read_text()); assert C['complete'] and not C['missing'],C
assert len(C['reference_parity'])==15
c=pd.read_csv(O/'part_c/across_seeds.csv'); p=pd.read_csv(O/'part_c/paired_per_seed.csv')
assert len(c)==24 and len(p)==120
for r in c.itertuples():
    group=p[(p.dataset==r.dataset)&(p.method==r.method)].sort_values('seed')
    assert group.seed.tolist()==list(range(1729,1734))
    np.testing.assert_allclose(group.difference.mean(),r.mean_difference,atol=1e-12,rtol=0)
    np.testing.assert_allclose(group.difference.std(ddof=1),r.seed_sd,atol=1e-12,rtol=0)
    assert int((group.difference>0).sum())==r.seeds_higher
for u in json.loads((O/'part_c/units.json').read_text()):
    for method in ['DCA','EnImpute']:
        m=json.loads((Path(u['extras'][method])/'metadata.json').read_text())
        assert m['seed']==u['seed'] and m['parameters']['ncores']==16,(u['key'],method)
table=(Path(__import__('os').environ['SAFE_FUSION_MANUSCRIPT_DIR']) / '6_results.tex').read_text().split('\\label{tab:knockdown}')[1].split('\\bottomrule')[0]
current_matches=0
for method,g in a.groupby('method',sort=False):
    line=next(line for line in table.splitlines() if line.startswith(method+' &'))
    observed=re.findall(r'\d+\.\d+',line)
    expected=[]
    for dataset,adjustment in [('adamson_crispri','none'),('adamson_crispri','depth_strata'),('norman_crispra','none'),('pancreas','none'),('colon','none')]:
        row=g[(g.dataset==dataset)&(g.adjustment==adjustment)].iloc[0]
        expected += [f'{row[k]:.2f}' for k in ['continuous','continuous_lower','continuous_upper']]
    current_matches += sum(x==y for x,y in zip(observed,expected))
report=dict(result_checks_passed=True,source_hashes_unchanged=len(hashes)-len(changed),all_source_hashes_unchanged=not changed,external_presentation_drift=changed,current_table2_continuous_numbers_matching=current_matches,current_table2_continuous_numbers_expected=195,part_a_source_cells=65,part_a_caption_gate_passed=False,part_b_unchanged_rows=18000,part_c_complete_comparisons=24,part_c_seed_comparisons=120,reference_estimates_and_intervals_reproduced=15)
(O/'final_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(report)
