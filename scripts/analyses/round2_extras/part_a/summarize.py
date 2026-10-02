import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
O=(Path.cwd() / 'artifacts/paper_evidence/review_round4/round2_extras/part_a')
source=pd.read_csv(O/'source_gate.csv')
frames=[pd.read_csv(O/f'{x}_summary.csv') for x in ['adamson_crispri','norman_crispra','pancreas','colon']]
f=pd.concat(frames,ignore_index=True)
checks=[]; rows=[]
for col,(dataset,adj) in enumerate([('adamson_crispri','none'),('adamson_crispri','depth_strata'),('norman_crispra','none'),('pancreas','none'),('colon','none')]):
    for method,g in f[(f.dataset==dataset)&(f.adjustment==adj)].groupby('method',sort=False):
        fill=g[g.score_type=='fill_order'].iloc[0]; cont=g[g.score_type=='continuous'].iloc[0]
        saved=source[(source.method==method)&(source.column==col)].iloc[0]
        reproduced=fill.estimate if saved.source_score=='fill_order' else cont.estimate
        np.testing.assert_allclose(reproduced,saved.source_value,atol=1e-12,rtol=0)
        checks.append(dict(method=method,column=col,source_type=saved.source_score,reproduced=reproduced,paper=saved.paper,source_exact_atol=1e-12,fill_order_matches_printed=f'{fill.estimate:.2f}'==f'{saved.paper:.2f}'))
        rows.append(dict(method=method,dataset=dataset,adjustment=adj,paper=saved.paper,fill_order=fill.estimate,fill_lower=fill.lower,fill_upper=fill.upper,continuous=cont.estimate,continuous_lower=cont.lower,continuous_upper=cont.upper,tied_pair_share=fill.tied_pair_share,unfilled_zero_share=fill.unfilled_zero_share,n_units=fill.n_units,fill_grid=fill.fill_grid))
pd.DataFrame(rows).to_csv(O/'table2_aurocs.csv',index=False)
pd.DataFrame(checks).to_csv(O/'reproduction.csv',index=False)
(O/'verification.json').write_text(json.dumps(dict(all_65_source_values_reproduced=True,fill_order_printed_matches=sum(x['fill_order_matches_printed'] for x in checks),cells=65,all_fill_order_points_reproduce_table2=all(x['fill_order_matches_printed'] for x in checks),reason='The manuscript mixes fill-order screen AUROCs and continuous-score tissue AUROCs.'),indent=2)+'\n')
print((O/'verification.json').read_text())
