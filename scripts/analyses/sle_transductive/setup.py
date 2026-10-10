from pathlib import Path
import json
ROOT = Path.cwd()
OLD = ROOT / 'artifacts/paper_evidence/sle_treg_case'
OUT = ROOT / 'artifacts/paper_evidence/review_round4/sle_transductive'
(OUT/'logs').mkdir(parents=True,exist_ok=True)
UNITS = [f'main/fold_{k}/{a}' for k in range(3) for a in ('masked','deploy')] + [f'{s}/fold_{k}/deploy' for s in ('sex','cite') for k in range(3)]
def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists(): dst.symlink_to(src.resolve())
for variant in ('verification','conditional','conditional_detection'):
    case = OUT / variant
    for s in ('main','sex','cite'):
        for p in (OLD/s).iterdir():
            if p.is_file(): link(p, case/s/p.name)
    for unit in UNITS:
        old, new = OLD/unit, case/unit
        new.mkdir(parents=True, exist_ok=True)
        for p in old.parent.iterdir():
            if p.is_file(): link(p, new.parent/p.name)
        for p in old.iterdir():
            if p.is_file(): link(p, new/p.name)
        link(old/'baselines', new/'baselines')
        for m in ('svd_impute','graph_smooth'):
            link(old/'methods'/m, new/'methods'/m)
        if variant == 'verification':
            link(old/'methods/safe_fusion', new/'methods/safe_fusion')
            link(old/'selector', new/'selector')
        elif variant == 'conditional_detection':
            link(OUT/'conditional'/unit/'selector', new/'selector')
            link(OUT/'conditional'/unit/'methods/safe_fusion',new/'methods/safe_fusion')
(OUT/'units.json').write_text(json.dumps(UNITS, indent=2)+'\n')
