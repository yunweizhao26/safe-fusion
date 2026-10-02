import argparse
import hashlib
import json
from pathlib import Path


def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        assert target.resolve() == source.resolve(), target
    elif target.exists():
        raise FileExistsError(target)
    else:
        target.symlink_to(source.resolve())


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text() == text, path
    else:
        with path.open('x') as stream:
            stream.write(text)


def setup(name, reference_root=None):
    root = Path.cwd()
    evidence = root / 'artifacts/paper_evidence'
    out = evidence / 'review_round4' / name
    source = root / 'scripts/analyses' / name
    assert source.is_dir(), source
    flat = {'transductive_comparators', 'current_tables', 'fill_grid', 'scale_caps', 'sle_donor_labels'}
    for path in source.rglob('*'):
        if path.is_file() and path.suffix in ('.py', '.sh', '.R', '.s', '.cpp'):
            relative = path.relative_to(source)
            target = out / ('code' if name in flat else '') / relative
            link(path, target)
    for directory in ['logs', 'tmp', 'runtime']:
        (out / directory).mkdir(parents=True, exist_ok=True)
    if name == 'label_baselines':
        for directory in ['masked', 'screens', 'sex/pancreas', 'sex/colon']:
            (out / directory).mkdir(parents=True, exist_ok=True)
        write(out / 'sex/design.md', 'Donor-label rankings use the original masked and hybrid inputs, donor bootstrap, and expression-derived sex labels.\n')
    if name == 'reviewer_extras':
        for directory in ['A_calibration', 'B_thinning', 'C_scale/logs']:
            (out / directory).mkdir(parents=True, exist_ok=True)
    if name == 'current_tables':
        old = root / 'scripts/analyses/transductive_main'
        for path in old.glob('*.py'):
            link(path, evidence / 'review_round4/transductive_main/code' / path.name)
        for filename in ['apply_fill_fraction.py', 'calibrated_selective_fill.py', 'evaluate_inserted_value.py', 'evaluate_thinning_transfer.py', 'evaluate_value_accuracy.py', 'run_autoencoder_fusion.py', 'run_leakage_safe_method.py', 'selector_attribution.py', 'v2_value_evaluate.py']:
            write(out / 'source_snapshot' / filename, (root / 'scripts' / filename).read_text())
        write(out / 'execution_notes.json', json.dumps({'run': 'release reproduction'}) + '\n')
        write(out / 'dependency_recovery.txt', 'No dependency recovery recorded at initialization.\n')
    if name == 'transductive_comparators':
        units = json.loads((evidence / 'review_round2/leakage_free/units_manifest.json').read_text())
        units = [u for u in units if u['dataset'] != 'Colon']
        units += json.loads((evidence / 'review_round3/colon_crossfit/units_manifest.json').read_text())
        assert len(units) == 7
        for unit in units:
            base = evidence / ('review_round3/colon_crossfit/fusion_value' if unit['dataset'] == 'Colon' else 'review_round2/fusion_value') / 'transductive' / unit['key']
            unit['contracts'].update({
                'SVD (transductive)': str((base / 'svd_impute').relative_to(root)),
                'Weighted kNN (transductive)': str((base / 'graph_smooth').relative_to(root)),
                'ALRA (transductive)': str((out / 'alra' / unit['key']).relative_to(root)),
            })
        write(out / 'units_manifest.json', json.dumps(units, indent=2) + '\n')
    if name == 'scale_caps':
        for genes in [2000, 5000]:
            base = out / f'genes_{genes}'
            original = evidence / ('review_round3/scale' if genes == 2000 else 'review_round4/reviewer_extras/C_scale') / 'cells_200000'
            write(base / 'source.txt', str(original) + '\n')
            for entry in ['baselines', 'masked', 'methods', 'splits.parquet']:
                link(original / entry, base / 'evaluation/cells_200000' / entry)
            for variant in ['default', 'large']:
                link(base / variant / 'selector', base / 'evaluation/cells_200000' / f'{variant}_selector')
    if reference_root is not None:
        required = {
            'current_tables': ['original_supplementary_results.tex'],
            'round2_extras': ['part_a/table2_original_snapshot.csv'],
        }
        for relative in required.get(name, []):
            original = reference_root / name / relative
            assert original.is_file(), original
            link(original, out / relative)
    scripts = sorted(p for p in (root / 'scripts').rglob('*') if p.is_file() and p.suffix in ('.py', '.sh', '.s', '.R', '.cpp'))
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts}
    write(out / 'source_sha256.json', json.dumps(hashes, indent=2) + '\n')
    checksum_text = ''.join(f'{digest}  {path}\n' for path, digest in hashes.items())
    write(out / 'source_sha256.txt', checksum_text)
    if name == 'scale_caps':
        write(out / 'code/source_sha256.json', json.dumps(hashes, indent=2) + '\n')
    if name == 'sle_donor_labels':
        write(out / 'runtime/source_sha256_before.txt', checksum_text)
    print(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('analysis', choices=['transductive_comparators', 'label_baselines', 'round2_extras', 'current_tables', 'fill_grid', 'reviewer_extras', 'scale_caps', 'scale_comparators', 'sle_donor_labels'])
    parser.add_argument('--reference-root', type=Path)
    args = parser.parse_args()
    setup(args.analysis, args.reference_root)
