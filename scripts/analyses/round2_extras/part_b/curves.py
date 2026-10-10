import sys,json,csv
from pathlib import Path
import numpy as np
import pandas as pd
sys.path[:0]=['scripts','src']
from masked_f1_units import load_units_manifest,load_unit,count_scale_values,CURVE_BUDGETS
from compute_matched_baseline_f1_curves import tie_broken_order,ranked_curve,pool_curves
O=(Path.cwd() / 'artifacts/paper_evidence/review_round4/round2_extras/part_b')
R=O.parent.parent
methods=['SVD','Weighted kNN','ALRA']
curves={}
for u in load_units_manifest(R/'transductive_comparators/units_manifest.json'):
    d=load_unit(u)
    rr,cc=np.where((d.counts==0)&(d.split=='test')[:,None])
    for method in methods:
        score,_=count_scale_values(u.contracts[method+' (transductive)'],d,rr,cc)
        curves.setdefault((u.dataset,method),[]).append(ranked_curve(tie_broken_order(score,u.tie_seed),d.masked[rr,cc],CURVE_BUDGETS))
    print(u.key,flush=True)
pooled={k:pool_curves(v) for k,v in curves.items()}
source=R/'transductive_main/selector_f1_fillrate_transductive_1000_points.csv'

lines=source.read_text().splitlines(keepends=True)
header=next(csv.reader([lines[0]]))
import io
output=[lines[0]]; n=0
for line in lines[1:]:
    row=dict(zip(header,next(csv.reader([line]))))
    key=(row['dataset'],row['method'])
    if key in pooled:
        index=int(row['coverage_index']); row.update(pooled[key][index])
        buf=io.StringIO(); csv.DictWriter(buf,fieldnames=header,lineterminator='\n').writerow(row)
        output.append(buf.getvalue()); n+=1
    else: output.append(line)
assert n==9000
out=O/'selector_f1_fillrate_allcell_1000_points.csv'
out.write_text(''.join(output))
new=pd.read_csv(out); old=pd.read_csv(source)
pd.testing.assert_frame_equal(new[~new.method.isin(methods)],old[~old.method.isin(methods)],check_exact=True)
comparators=['Weighted kNN','SVD','ALRA','SAVER','MAGIC','scVI','scGPT']
(O/'verification.json').write_text(json.dumps(dict(replaced_rows=n,unchanged_rows=len(old)-n,unchanged_records_verbatim=True,grid_points=1000,comparators=comparators),indent=2)+'\n')
import runpy
runpy.run_path(str(O/'verify_points.py'),run_name='__main__')
