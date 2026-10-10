import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
import pandas as pd
import evaluate_protein_within_state as e


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--benchmark-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--include-magic', action='store_true')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    panel = ROOT / 'artifacts/paper_evidence/papalexi_crossmodal/audit/matched_rna_adt_panel.parquet'
    if args.include_magic:
        e.VALUE_RANKINGS = {**e.VALUE_RANKINGS, 'magic': 'magic_standard'}
        e.RNA_RANKINGS = (*e.RNA_RANKINGS, 'magic')
        e.RANKINGS = (*e.RANKINGS, 'magic')
    inputs = e.load_inputs(args.prepared, args.benchmark_root, panel)
    global_dir = args.benchmark_root / 'global_fill_selector'
    all_scores = pd.read_parquet(global_dir / 'selected_gene_scores.parquet')
    control_scores = all_scores
    frames = {}
    for protein, gene in e.PROTEIN_TO_GENE.items():
        for cells, loader in [('recorded_rna_zero', e.recorded_zero_frame), ('detected_rna', e.detected_rna_frame)]:
            frame = loader(inputs, protein, gene)
            frame['cell_zero_score'] = e.cell_zero_score(frame, control_scores, gene)
            frames[(protein, cells)] = frame
    jobs = [j for j in e.make_jobs(frames, 2000, 1729)
            if j.protein == 'PDL1' and j.scale == 'clr'
            and j.cells == 'recorded_rna_zero' and j.scope in ('all', 'control')]
    assert len(jobs) == 3
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(e.run_job, jobs))
    association = pd.concat([r[0] for r in results], ignore_index=True)
    paired = pd.concat([r[1] for r in results], ignore_index=True)
    fill, check = e.global_fill(inputs, frames, all_scores, global_dir, args.benchmark_root)
    assert check['max_abs_score_difference_from_production_selector'] <= 1e-7, check
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    association.to_csv(out / 'association.csv', index=False)
    paired.to_csv(out / 'paired_differences.csv', index=False)
    e.pdl1_pooled_table(association, paired).to_csv(out / 'pdl1_pooled_rankings.csv', index=False)
    e.by_target_table(frames).query('gene == "CD274" and scale == "clr"').to_csv(out / 'within_target_by_target.csv', index=False)
    e.target_table(inputs, frames).to_csv(out / 'cd274_zero_targets.csv', index=False)
    fill.query('gene == "CD274"').to_csv(out / 'global_fill.csv', index=False)
    frames[('PDL1', 'recorded_rna_zero')].to_parquet(out / 'pdl1_cells.parquet', index=False)
    report = {'n_cd274_zeros': len(frames[('PDL1', 'recorded_rna_zero')]),
              'n_control_zeros': int((frames[('PDL1', 'recorded_rna_zero')]['target'] == 'NT').sum()),
              'n_targets_including_control': int(frames[('PDL1', 'recorded_rna_zero')]['target'].nunique()),
              'benchmark_split_counts': inputs['cells']['split'].value_counts().to_dict(),
              'bootstrap': 2000, 'seed': 1729, 'ifng_genes': list(e.IFNG_RESPONSE_GENES),
              'global_fill_check': check, 'unchanged_functions': ['load_inputs', 'make_jobs', 'run_job', 'global_fill']}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
