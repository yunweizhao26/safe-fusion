import json
from pathlib import Path

out = (Path.cwd() / 'artifacts/paper_evidence/review_round4/label_baselines')
screen = json.loads((out / 'screens/verification_receipt.json').read_text())
gate = json.loads((out / 'verification_passed.json').read_text())
assert gate['passed'] and screen['saved_evaluators_match']
differences = [row for row in screen['paper_mismatches']
               if float(row['displayed_recomputed']) != float(row['paper'])]
parts = [
    '# Label-aware baselines and colon sex-label sensitivity',
    'Output: `artifacts/paper_evidence/review_round4/label_baselines/`. '
    'All paths below are relative to this directory unless stated otherwise.',
    'Safe Fusion with labels exceeds both label-aware teachers on masked F1 and AP in all five datasets. '
    'It also exceeds the strict fitting-only per-label rule in all three screens. '
    'The separately labelled transductive donor rule also trails Safe Fusion on both metrics in pancreas and colon. '
    'Every corresponding paired 95% interval excludes zero. '
    'The teacher’s higher biological-proxy AUROC therefore does not imply better masked-entry recovery in these benchmarks. '
    'Colon donor-label AUROCs remain high on the 25 agreeing donors (RPS4Y1 0.848; XIST 0.928), '
    'but fall to approximately 0.605 for both genes when metadata sex labels all 34 donors; both metadata-based intervals include 0.5.',
    '## Verification and scope',
    'The saved S16 evaluator tables and bootstrap summaries reproduce within 1e-12. '
    'All 48 S18 estimates and interval pairs reproduce within 1e-12, and their printed values match exactly. '
    f'However, {len(differences)} S16 AUROC cells do not match direct two-decimal rounding of the saved evaluator results. '
    'Exact reproduction of S16 as printed therefore failed. The discrepancies below were audited before new analyses were released. '
    'The analyses retain the original numerical definitions, settings and seeds; no paper file was changed.',
    '| S16 row | Screen | Adjustment | Recomputed AUROC | Rounded | Printed |\n'
    '|---|---|---|---|---|---|\n' + '\n'.join(
        f"| {r['method']} | {r['dataset']} | {r['metric']} | {r['recomputed']:.9f} | {r['displayed_recomputed']} | {r['paper']} |"
        for r in differences),
    'The initial verification job stopped on the first printed mismatch. A complete audit then confirmed all saved numerical outputs. '
    'The release receipt explicitly records the manuscript mismatch instead of declaring exact paper reproduction. '
    'The Adamson label-aware Safe Fusion effect shift is -0.002223 and displays as -0.00; the paper displays 0.00. '
    'Evidence: `screens/verification_receipt.json`, `screens/s16_paper_cells.csv`, '
    '`sex/{pancreas,colon}/gate.json`, and `verification_passed.json`.',
    'All intervals use 2,000 paired bootstrap draws and seed 1729 with the original endpoint-specific unit order and seed derivation. '
    'F1 measures recovery of masked positive entries, averaged over the ten fill fractions 1%–10%; AP ranks all held-out zero candidates. '
    'These metrics do not measure numerical count-amplitude reconstruction. '
    'The expected-count detection ranking is 1-exp(-L_c a_g,l), with gene shares estimated only from masked fitting counts. '
    'It is undefined for held-out donor labels because those donors have no fitting cells. '
    'That strict baseline is not silently replaced. A separately labelled transductive per-label baseline '
    'uses the existing donor teachers’ actual fitting population: all cells in the supplied observed-count input. '
    'Its masked benchmark input remains masked, and its deployment input is the same hybrid matrix used by the teachers.',
]
for path in ['masked/summary.md', 'screens/summary.md', 'sex/summary.md']:
    parts.append((out / path).read_text().strip())
parts += [
    '## Complete reproduction',
    'Run from the repository root in a restored workspace containing the input artifacts and these scripts, '
    'with generated outputs under this new directory absent. The selector replay refuses to overwrite an existing replay directory. '
    'The script submits verification first, gates all new analyses on saved-result reproduction, and waits in 600-second loops. '
    'Per-unit selector commands are also preserved in `masked/replay/<unit>/command.json` and `command.sh`.',
    '```bash\nbash artifacts/paper_evidence/review_round4/label_baselines/reproduce.sh\n```',
    '## Execution, failures and remaining limits',
    'All workloads ran through Slurm on partition `cs`, account `torch_pr_634_general`. '
    'Only new `lb-*` job chains were managed. Existing `scmp-*`, `cx-*` and `mm3-*` jobs were not modified. '
    'The cluster rejected dependency updates after the first verification failed; only our pending dependent jobs were cancelled and resubmitted. '
    'The original submission that referenced an expired completed array dependency was rejected and replaced. '
    '`jobs.txt` and `complete_job_accounting.txt` record execution, CPU allocation, elapsed time and available memory/CPU measurements.',
    'The fitting-only donor baseline remains unavailable in A and B. Pancreas XIST retains the original fold-0-only coverage. '
    'Four S16 printed AUROC discrepancies remain unresolved in the manuscript, which was read only. '
    'No missing baseline was assigned a zero score, and no full-cohort pancreas XIST result was inferred. '
    'All other requested available-data analyses are complete. '
    'A strictly selector-fitting-only donor estimate would require a new unseen-label rule. '
    'The separately reported transductive estimate answers the comparison using the existing donor-teacher fitting population and is not that strict estimator.',
    'Source checksums: `source_sha256.txt` and each analysis directory’s provenance files. '
    'The score-export replays use the existing selector with only the additive `--score-gene` option. '
    'Their verification receipts require exact agreement with the saved calibration results, excluding timing fields, and all ten fill matrices. '
    'No commit or push was made. Existing paper files, documentation and root README were not edited.',
]
(out / 'README.md').write_text('\n\n'.join(parts) + '\n')
