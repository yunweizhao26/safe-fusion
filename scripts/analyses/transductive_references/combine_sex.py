from pathlib import Path
import json
import pandas as pd
O=Path('artifacts/paper_evidence/review_round4/transductive_references/sex_zeros')
for tissue in ('pancreas','colon'):
 a=O/'recorded_transductive'/tissue; b=O/'recorded_inductive'/tissue; out=O/'evaluation'/tissue;out.mkdir(parents=True,exist_ok=True)
 for name in ('zero_counts.csv','donor_sex.csv','gene_availability.csv'):
  x=pd.read_csv(a/name);y=pd.read_csv(b/name);pd.testing.assert_frame_equal(x,y);x.to_csv(out/name,index=False)
 for name in ('auroc.csv','fill_rates.csv'): pd.concat([pd.read_csv(a/name),pd.read_csv(b/name)],ignore_index=True).to_csv(out/name,index=False)
 audit=json.loads((a/'audit.json').read_text());audit['inductive_cohort_matches']=True
 (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print('Matched sex-zero tables assembled; cohort checks passed.')
