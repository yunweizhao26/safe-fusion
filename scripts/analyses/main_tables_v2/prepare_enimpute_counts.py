
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path.cwd()
O=(Path.cwd() / 'artifacts/paper_evidence/review_round4/main_tables_v2')
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from masked_f1_units import Unit,load_units_manifest,load_unit,count_scale_values
from compute_matched_baseline_f1_curves import tie_broken_order

specs=json.loads((O.parent/'round2_extras/part_c/units.json').read_text())
entries=[]
for spec in specs:
    if spec['seed'] != 1729:
        continue
    row={key:value for key,value in spec.items() if key in Unit.__dataclass_fields__}
    row['contracts']={'EnImpute':spec['extras']['EnImpute']}
    entries.append(row)
assert len(entries)==7
manifest=O/'enimpute_units.json'
manifest.write_text(json.dumps(entries,indent=2)+'\n')
rows=[];sources={}
for unit in load_units_manifest(manifest):
    data=load_unit(unit)
    rr,cc=np.where((data.counts==0)&(data.split=='test')[:,None])
    labels=data.masked[rr,cc]
    names,codes=np.unique(np.char.add(unit.key+':',data.unit_labels[rr].astype(str)),return_inverse=True)
    scores,_=count_scale_values(unit.contracts['EnImpute'],data,rr,cc)

    order=tie_broken_order(scores.astype(np.float64),unit.tie_seed)
    positive=np.bincount(codes[labels],minlength=len(names))
    for fraction in np.arange(1,11)/100:
        chosen=order[:max(1,round(fraction*len(order)))]
        selected=np.bincount(codes[chosen],minlength=len(names))
        hits=np.bincount(codes[chosen[labels[chosen]]],minlength=len(names))
        rows.extend(dict(dataset=unit.dataset,method='EnImpute',group=unit.key,unit=name,fraction=fraction,n_selected=s,n_true_positive=h,n_masked_positives=p) for name,s,h,p in zip(names,selected,hits,positive))
    for path in [unit.corrupted,unit.splits,unit.coordinates,unit.contracts['EnImpute']/'mean.npy',unit.contracts['EnImpute']/'metadata.json']:
        sha=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):
                sha.update(block)
        sources[str(path.relative_to(ROOT))]=sha.hexdigest()
    print('Frozen EnImpute counts:',unit.key,flush=True)
pd.DataFrame(rows).to_parquet(O/'enimpute_unit_counts.parquet',index=False)
(O/'enimpute_input_sha256.json').write_text(json.dumps(sources,indent=2)+'\n')
