from pipeline import *
design=sys.argv[1]
args=[]
for d in D:
 if d['design']==design and d['key'].split('_')[0] in ['pancreas','colon']:
  key=d['key']; tissue,fold=key.rsplit('_',1)
  args+=['--unit',f'{tissue}/fold_{fold}={d["dir"]}']
for name in NAMES:
 for pct in PCTS: args+=['--method',f'{name}_{pct}pct={{unit}}/{name}_{pct}pct']
for scale in ['counts','log_cp10k']:
 run('scripts/r3_downstream_correlation.py',*args,'--label-column','cell_type','--scale',scale,'--fdr',.05,'--bootstrap',2000,'--seed',1729,'--output-dir',R/'correlation'/design/scale)
