import csv,hashlib,json,subprocess
from pathlib import Path
O=Path.cwd()/'artifacts/paper_evidence/review_round4/table2_added'
def records(p):
    with p.open() as f:return list(csv.DictReader(f))
endpoints=[('adamson_crispri','none','CRISPRi unadjusted'),('adamson_crispri','depth_strata','CRISPRi library strata'),('norman_crispra','none','CRISPRa'),('pancreas','none','Pancreas sex RPS4Y1'),('colon','none','Colon sex RPS4Y1')]
results=records(O/'evaluation/added_estimates.csv') if (O/'evaluation/added_estimates.csv').exists() else []
lookup={(r['method'],r['dataset'],r['adjustment']):r for r in results}
methods=['scRecover','scGPT']
reference=json.loads((O/'evaluation/reference_checks.json').read_text()) if (O/'evaluation/reference_checks.json').exists() else []
assert reference and all(c['passed'] for c in reference),'Reference gate must pass before rows are released'
main=[];supp=[]
for method in methods:
    a={'Method':method};b={'Method':method}
    for dataset,adj,column in endpoints:
        r=lookup.get((method,dataset,adj))
        ci=f"{float(r['estimate']):.2f} [{float(r['lower']):.2f}, {float(r['upper']):.2f}]" if r else 'Unavailable'
        b[column]=ci
        if adj=='none': a[column]=f"{float(r['estimate']):.2f} ± {float(r['se']):.2f}" if r else 'Unavailable'
    main.append(a);supp.append(b)
for file,rows in [('table2_rows.csv',main),('s17_rows.csv',supp)]:
    with (O/file).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
assert len(results)==12 and len(lookup)==12
checks=json.loads((O/'evaluation/added_checks.json').read_text())
assert checks and all(c['passed'] for c in checks)
print('Wrote Table 2 and S17 rows; reference and added checks passed')
