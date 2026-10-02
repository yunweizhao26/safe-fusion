from pathlib import Path
import json
ROOT=Path.cwd()
SRC=ROOT/'artifacts/paper_evidence/review_round4/sle_transductive'
O=SRC.with_name('sle_donor_labels')
units=json.loads((SRC/'units.json').read_text())
def link(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.is_symlink() and not dst.exists(): dst.symlink_to(src.resolve())
for variant in ('verification','verification_detection','conditional','conditional_detection'):
    case=O/variant
    for s in ('main','sex','cite'):
        for p in (SRC/'conditional'/s).iterdir():
            if p.is_file(): link(p,case/s/p.name)
    for name in units:
        source=SRC/'conditional'/name
        new=case/name
        new.mkdir(parents=True,exist_ok=True)
        for p in source.parent.iterdir():
            if p.is_file(): link(p,new.parent/p.name)
        for p in source.iterdir():
            if p.is_file(): link(p,new/p.name)
        link(source/'baselines',new/'baselines')
        for m in ('svd_impute','graph_smooth'): link(source/'methods'/m,new/'methods'/m)
        if variant.startswith('verification'):
            link(source/'selector',new/'selector')
            link(source/'methods/safe_fusion',new/'methods/safe_fusion')
        elif variant=='conditional_detection':
            link(O/'conditional'/name/'selector',new/'selector')
            link(O/'conditional'/name/'methods/safe_fusion',new/'methods/safe_fusion')
        else:
            for m in ('gene_median','svd_impute','magic'): link(source/'teachers'/m,new/'teachers'/m)
(O/'units.json').write_text(json.dumps(units,indent=2)+'\n')
