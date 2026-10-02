from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

OUT = Path('artifacts/paper_evidence/review_round4/label_baselines/screens')
R4 = Path('artifacts/paper_evidence/review_round4/transductive_references/knockdown/evaluation_matched')
R3 = Path('artifacts/paper_evidence/review_round3/comparators/evaluation/known_zeros/knockdown')
checks = []
for original, rerun in [(R4, OUT/'verification_transductive'), (R3, OUT/'verification_inductive_comparators/knockdown')]:
    for filename in ['auroc.csv', 'effects.csv'] + (['targets.csv', 'fills.csv', 'mixscape.csv', 'per_target_table.csv'] if original == R4 else ['summary.csv']):
        a, b = pd.read_csv(original/filename), pd.read_csv(rerun/filename)
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12, atol=1e-12)
        checks.append(dict(source=str(original/filename), rerun=str(rerun/filename), rows=len(a), match=True))

assert json.loads((R4/'report.json').read_text()) == json.loads((OUT/'verification_transductive/report.json').read_text())
checks.append(dict(source=str(R4/'report.json'), match=True))
trans = OUT/'verification_transductive'
ind = OUT/'verification_inductive_comparators/knockdown'
methods = {
'Safe Fusion': ('safe_fusion', trans), 'Safe Fusion, inductive': ('safe_fusion', ind),
'SVD': ('svd', trans), 'Weighted kNN': ('weighted_knn', trans), 'ALRA': ('alra', trans),
'SAVER': ('saver', trans), 'MAGIC': ('magic_standard', trans), 'scVI': ('scvi_standard', trans),
'DCA': ('DCA', ind), 'DCA, zero-inflated': ('DCA (ZINB)', ind), 'scImpute': ('scImpute', ind),
'scRecover': ('scRecover', ind), 'EnImpute': ('EnImpute', ind), 'scVI, zero-inflated': ('scVI (ZINB)', ind),
'scImpute dropout probability': ('scImpute P(dropout)', ind), 'scRecover dropout probability': ('scRecover P(dropout)', ind),
'Safe Fusion, labels': ('safe_fusion_condition', trans), 'Weighted kNN teacher, labels': ('knn_condition', trans),
'scVI teacher, labels': ('scvi_condition', trans),
}
loaded = {path: {kind: pd.read_csv(path/f'{kind}.csv') for kind in ['auroc','effects']} for path in [trans, ind]}
paper = (Path(__import__('os').environ['SAFE_FUSION_MANUSCRIPT_DIR']) / 'supplementary_results.tex').read_text().split(r'\label{tab:s_knockdown}')[1].split(r'\end{table}')[0]
cell_rows = []
for line in paper.splitlines():
    if ' & ' not in line: continue
    parts = [p.strip() for p in line.removesuffix(r'\\').split('&')]
    if parts[0] not in methods: continue
    method, path = methods[parts[0]]
    assert len(parts) == 9
    for i, raw in enumerate(parts[1:]):
        expected = re.sub(r'\\textbf\{([^}]+)\}', r'\1', raw).replace('$','').strip()
        if expected == '--': continue
        dataset = ['adamson_crispri','papalexi_eccite','norman_crispra'][i//2] if i < 6 else ['adamson_crispri','norman_crispra'][i-6]
        if i < 6:
            part = loaded[path]['auroc']
            part = part[(part.dataset == dataset)&(part.method == method)&(part.score_type == 'fill_order')]
            metric = ['none','depth_strata'][i%2]
        else:
            part = loaded[path]['effects']
            part = part[(part.dataset == dataset)&(part.method == method)&(part.fill_pct == 10)]
            metric = 'shift'
        assert len(part) == {'adamson_crispri':26,'papalexi_eccite':6,'norman_crispra':41}[dataset]
        value = float(part[metric].mean())

        actual = f'{round(value,4):.2f}'
        cell_rows.append(dict(method=parts[0],dataset=dataset,metric=metric,paper=expected,recomputed=value,
                              displayed_recomputed=actual,match=actual == expected,source=str(path)))
pd.DataFrame(cell_rows).to_csv(OUT/'s16_paper_cells.csv',index=False)
failures = [row for row in cell_rows if not row['match']]
receipt = dict(status='paper_mismatch' if failures else 'passed',saved_evaluators_match=True,
               paper_matches=len(cell_rows)-len(failures),paper_mismatches=failures,checks=checks,
               paper_cells=len(cell_rows),numeric_tolerance=1e-12,
               paper_rounding='4 decimals in evaluator report, then 2 decimals',draws=2000,seed=1729)
(OUT/'verification_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2),flush=True)
assert not failures, f'{len(failures)} manuscript cells do not reproduce'
