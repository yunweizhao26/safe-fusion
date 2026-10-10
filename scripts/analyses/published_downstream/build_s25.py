import argparse, csv, sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=Path, nargs=2, default=[
    Path('artifacts/paper_evidence/review_round4/published_downstream/fill_grid_published_long.csv'),
    Path('artifacts/paper_evidence/review_round4/published_downstream_allcells/fill_grid_allcells_long.csv'),
])
files = parser.parse_args().data
rows = [r for f in files for r in csv.DictReader(open(f))]
ENDPOINTS = [("colon", "Marker AUPRC"), ("pancreas", "Marker AUPRC"), ("zebrafish", "Dynamics"),
             ("colon", "Reference mapping F1"), ("pancreas", "Reference mapping F1"),
             ("zebrafish", "Stage error"), ("norman_crispra", "Edge AUPRC")]
METHODS = [("detection", "Safe Fusion, weighted"), ("transductive", "Safe Fusion, fused"), None,
           ("magic", "MAGIC"), ("scvi", "scVI"), ("alra_all_cells", "ALRA"), ("saver", "SAVER"),
           ("dca", "DCA"), ("scimpute", "scImpute"), ("screcover", "scRecover"), ("enimpute", "EnImpute"), None,
           ("svd_all_cells", "SVD"), ("selector_scvi", "Selector on scVI"), ("selector_magic", "Selector on MAGIC")]
FRACTIONS = [1.0, 10.0, 100.0]
idx = {(r["dataset"], r["endpoint"], r["method"], float(r["fill_pct"])): r for r in rows}

def cell(key):
    r = idx.get(key)
    if r is None or r["gap_closed_pct"] in ("", "NA", "nan"):
        return None
    g, lo, hi = float(r["gap_closed_pct"]), float(r["gap_ci_low_pct"]), float(r["gap_ci_high_pct"])
    star = "*" if lo > 0 or hi < 0 else ""
    return g, f"{g:.1f}".replace("-", "$-$") + star

out = []
for frac in FRACTIONS:
    best = {}
    for ep in ENDPOINTS:
        vals = [cell((*ep, m[0], frac)) for m in METHODS if m]
        best[ep] = max(round(v[0], 1) for v in vals if v)
    out.append(r"\midrule")
    out.append(r"\multicolumn{8}{@{}l}{\emph{Fill fraction " + f"{frac:g}" + r"\%}} \\")
    for m in METHODS:
        if m is None:
            continue
        cells = []
        for ep in ENDPOINTS:
            v = cell((*ep, m[0], frac))
            if v is None:
                sys.exit(f"missing {ep} {m} {frac}")
            cells.append(r"\textbf{" + v[1] + "}" if round(v[0], 1) == best[ep] else v[1])
        out.append(m[1] + " & " + " & ".join(cells) + r" \\")
print("\n".join(out[1:]))
