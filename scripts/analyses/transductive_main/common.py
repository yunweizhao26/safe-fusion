from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / 'scripts'), str(REPO / 'src')]
E = REPO / 'artifacts/paper_evidence'
OUT = E / 'review_round4/transductive_main'
LF = E / 'review_round2/leakage_free'
CC = E / 'review_round3/colon_crossfit'
FV = E / 'review_round2/fusion_value'

def units():
    from masked_f1_units import load_units_manifest
    return [u for u in load_units_manifest(LF / 'units_manifest.json') if u.dataset != 'Colon'] + load_units_manifest(CC / 'units_manifest.json')

def fusion_root(u):
    return CC / 'fusion_value' if u.dataset == 'Colon' else FV

def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() and not target.is_symlink():
        target.symlink_to(source.resolve())
