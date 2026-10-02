import json
import os
from pathlib import Path
import tempfile

from setup_revision import setup


root = Path(__file__).resolve().parents[1]
previous = Path.cwd()
with tempfile.TemporaryDirectory() as directory:
    os.chdir(directory)
    Path('scripts').symlink_to(root)
    evidence = Path('artifacts/paper_evidence')
    units = []
    for dataset, prefix, count in [('Pancreas', 'pancreas', 3), ('CRISPRa', 'norman', 1), ('Colon', 'colon', 3)]:
        units.extend({'dataset': dataset, 'key': f'{prefix}_{i}', 'contracts': {}} for i in range(count))
    for name, rows in [('review_round2/leakage_free', units[:4]), ('review_round3/colon_crossfit', units[4:])]:
        path = evidence / name / 'units_manifest.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(rows))
    names = ['transductive_comparators', 'label_baselines', 'round2_extras', 'current_tables', 'fill_grid', 'reviewer_extras', 'scale_caps', 'scale_comparators', 'sle_donor_labels']
    for name in names:
        setup(name)
        setup(name)
    out = evidence / 'review_round4'
    manifest = json.loads((out / 'transductive_comparators/units_manifest.json').read_text())
    assert len(manifest) == 7
    assert all(len(unit['contracts']) == 3 for unit in manifest)
    assert all(not Path(value).is_absolute() for unit in manifest for value in unit['contracts'].values())
    assert (out / 'current_tables/code/selector.py').resolve() == root / 'analyses/current_tables/selector.py'
    assert (out / 'scale_caps/genes_5000/evaluation/cells_200000/large_selector').is_symlink()
    assert (out / 'sle_donor_labels/runtime/source_sha256_before.txt').stat().st_size > 0
    path = out / 'transductive_comparators/units_manifest.json'
    path.write_text('[]')
    try:
        setup('transductive_comparators')
    except AssertionError:
        pass
    else:
        raise AssertionError('Existing manifest was overwritten')
    os.chdir(previous)
print('Setup checks passed: nine analyses, repeatability, paths, and collision rejection.')
