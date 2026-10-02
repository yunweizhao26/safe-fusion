import json
from pathlib import Path

out = (Path.cwd() / 'artifacts/paper_evidence/review_round4/label_baselines/masked')
sets = ['adamson_crispri', 'papalexi_eccite', 'norman_crispra', 'Pancreas', 'Colon']
names = {'adamson_crispri': 'Adamson', 'papalexi_eccite': 'Papalexi', 'norman_crispra': 'Norman', 'Pancreas': 'Pancreas', 'Colon': 'Colon'}
methods = {'safe_fusion_labels': 'Safe Fusion with labels', 'weighted_knn_labels': 'Weighted kNN with labels', 'scvi_labels': 'scVI with labels', 'per_label_expected_count': 'Per-label expected count'}

def formatted(values, scale):
    x, lo, hi = [v * scale for v in values]
    return f'{x:.4f} [{lo:.4f}, {hi:.4f}]'

reports = {ds: json.loads((out / ds / 'report.json').read_text()) for ds in sets}
lines = ['## A. Masked recovery', '', 'Entries are estimates [95% paired unit-bootstrap intervals], 2,000 draws, seed 1729. Mean masked F1 averages fill fractions 1% through 10%; AP pools all candidate zeros. F1 is a percentage; AP is on the 0–1 scale.', '', '| Dataset | Ranking | Mean masked F1 (%) | AP |', '|---|---|---:|---:|']
for ds, report in reports.items():
    for method, label in methods.items():
        if method in report['unavailable']:
            f1 = ap = 'Unavailable: no fitting-label support'
        else:
            f1 = formatted(report['mean_masked_f1'][method], 100)
            ap = formatted(report['average_precision'][method], 1)
        lines.append(f'| {names[ds]} | {label} | {f1} | {ap} |')
lines += ['', 'Positive differences favor Safe Fusion with labels. F1 differences are percentage points; AP differences use the 0–1 scale.', '', '| Dataset | Safe Fusion minus | Mean F1 difference (pp) | AP difference |', '|---|---|---:|---:|']
for ds, report in reports.items():
    for method, label in methods.items():
        if method == 'safe_fusion_labels':
            continue
        if method in report['unavailable']:
            f1 = ap = 'Unavailable'
        else:
            f1 = formatted(report['safe_fusion_minus'][method]['mean_masked_f1'], 100)
            ap = formatted(report['safe_fusion_minus'][method]['average_precision'], 1)
        lines.append(f'| {names[ds]} | {label} | {f1} | {ap} |')
lines += ['', 'Safe Fusion F1 reproduces the original round-4 per-unit counts exactly for all five benchmarks. The missing continuous selector scores were exported through fixed-setting selector replay, requiring exact calibration reports (except timing) and exact equality of all ten saved fill matrices. Existing selector source and its attribution helper match the original round-4 snapshots byte for byte. The weighted AP kernel agrees with scikit-learn on tied weighted and duplicated-unit test cases; each full-data AP also agrees with scikit-learn.', '', 'The per-label rule uses only the original selector fitting split and masked counts. Held-out pancreas and colon donor labels have no fitting cells, so this strict rule has no defined tissue score. No all-cell means or global fallback are substituted. `masked/<dataset>/label_support.csv` lists every tested label, fitting count mass and support status. Screen teacher rankings retain the published comparator contracts; Safe Fusion retains its round-4 transductive teachers. Tissue donor-aware teachers are shared with Safe Fusion and retain their original transductive masked-count information.', '', 'Paths: `masked/<dataset>/report.json`, `masked_f1_unit_counts.csv`, `label_support.csv`, `units_manifest.json`, and `bootstrap_draws.npz`; `masked/replay/<unit>/verified.json` and exported scores; `masked/selfcheck.json`; `masked/design.md`; Slurm IDs in `masked/jobs.txt`.']
lines += ['', '### Supplementary tissue rule using the actual transductive teacher-fitting population', '', 'The tissue donor-aware teachers fit all supplied masked cells. This explicitly named variant uses that same population for per-label count shares. It supplements the strict selector-fitting rule above, whose held-out donors lack support; it does not replace that rule. The bootstrap draws and Safe Fusion values are identical to the main analysis.', '', '| Tissue | Mean masked F1 (%) | AP | SF minus rule F1 (pp) | SF minus rule AP |', '|---|---:|---:|---:|---:|']
for ds in ('Pancreas', 'Colon'):
    report = json.loads((out / 'supplementary' / ds / 'report.json').read_text())
    lines.append(f"| {ds} | {formatted(report['mean_masked_f1'], 100)} | {formatted(report['average_precision'], 1)} | {formatted(report['safe_fusion_minus']['mean_masked_f1'], 100)} | {formatted(report['safe_fusion_minus']['average_precision'], 1)} |")
lines += ['', 'Supplementary outputs: `masked/supplementary/<tissue>/{report.json,masked_f1_unit_counts.csv,label_support.csv,bootstrap_draws.npz}`. The support audit records all masked fitting cells, including the held-out cells seen by these transductive teachers.']
(out / 'summary.md').write_text('\n'.join(lines) + '\n')
