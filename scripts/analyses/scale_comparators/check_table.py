#!/usr/bin/env python3

from pathlib import Path
import json
import pandas as pd
ROOT=(Path.cwd() / 'artifacts/paper_evidence/review_round4/scale_comparators')
REPO=ROOT.parents[3]
data=pd.read_csv(REPO/'artifacts/paper_evidence/review_round3/scale/results/paired_differences.csv')
text=(Path(__import__('os').environ['SAFE_FUSION_MANUSCRIPT_DIR']) / 'supplementary_results.tex').read_text()
table=text.split(r'\label{tab:s_scale}',1)[1].split(r'\end{table}',1)[0]
methods={'SVD':'SVD','weighted kNN':'Weighted kNN','ALRA':'ALRA','SAVER':'SAVER','MAGIC':'MAGIC','scVI':'scVI','scVI probability':'scVI probability'}
checked=0
for label,method in methods.items():
    line=next(line for line in table.splitlines() if line.startswith('Minus '+label+' &'))
    values=[v.strip().removesuffix('\\\\').strip().replace('$-$','-') for v in line.split('&')[1:]]
    for n,printed in zip((25000,50000,100000,200000),values,strict=True):
        r=data[(data.cells==n)&(data.reference=='Safe Fusion')&(data.method==method)&(data.statistic=='mean_f1_1_to_10')].iloc[0]
        expected=f'{100*r.difference:.2f} [{100*r.lower:.2f}, {100*r.upper:.2f}]'
        assert printed==expected,(n,method,printed,expected)
        checked+=1
(ROOT/'S28_table_PASS.json').write_text(json.dumps({'entries_checked':checked,'exact_printed_match':True},indent=2)+'\n')
print(checked,'S28 entries match')
