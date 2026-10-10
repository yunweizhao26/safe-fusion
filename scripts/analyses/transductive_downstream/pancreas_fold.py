import sys
from pathlib import Path
sys.path[:0]=[str(Path.cwd()/'scripts'),str(Path.cwd()/'src')]
import evaluate_pancreas_crossfit_biology as pan
import pandas as pd
at=sys.argv.index('--fold'); fold=int(sys.argv[at+1]); del sys.argv[at:at+2]
source=Path('scripts/evaluate_colon_donor_biology.py')
code=source.read_text().replace('PANELS["colon"]','PANELS["pancreas"]').replace('obs["disease"]','obs["condition"]')
code=code.replace('selected_markers = sorted(set(panel) | INFLAMMATION_MARKERS)','selected_markers = sorted(set(panel))')
ns=dict(__name__='fold_adapter',__file__=str(source.resolve()))
exec(compile(code,str(source),'exec'),ns)
for name in ['canonical_marker_metrics','development_marker_metrics','macro_f1']:
 ns[name]=getattr(pan,name)
ns['centroid_predictions']=lambda matrix,labels,train,test,seed:pan.centroid_predictions(matrix,labels,train,test,seed+fold)

ns['main']()
out=Path(sys.argv[sys.argv.index('--output-dir')+1])
frame=pd.read_parquet(out/'donor_metrics.parquet').rename(columns={'disease':'condition'}).assign(fold=fold)
frame.to_parquet(out/'donor_metrics.parquet',index=False)
donors=frame[['donor','condition']].drop_duplicates().sort_values('donor')
draws=pan.stratified_bootstrap_indices(donors,2000,1729)
summ,paired=pan.summarize_unit_metrics(frame,draws)
summ.to_parquet(out/'bootstrap_summary.parquet',index=False); paired.to_parquet(out/'paired_comparisons.parquet',index=False)
