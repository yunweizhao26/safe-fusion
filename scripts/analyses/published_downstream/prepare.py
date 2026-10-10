import json
from pathlib import Path

root = Path.cwd()
round4 = root / 'artifacts/paper_evidence/review_round4'
units = []
keys = [f'pancreas_{i}' for i in range(3)] + ['norman_crispra'] + [f'colon_{i}' for i in range(3)] + ['zebrafish']
for key in keys + ['small', 'full']:
    analysis = 'downstream' if key in keys else 'protein'
    base = round4 / ('transductive_downstream/thinning/' + key if analysis == 'downstream' else
                     'transductive_references/protein_cs' if key == 'small' else 'protein_full/full')
    unit = dict(analysis=analysis, key=key,
                input=str((base / ('hybrid.h5ad' if analysis == 'downstream' else 'corrupted.h5ad')).relative_to(root)),
                splits=str((base / 'splits.parquet').relative_to(root)),
                coordinates=str((base / 'coordinates.parquet').relative_to(root)),
                source_dir=str(base.relative_to(root)),
                out=str((round4 / ('published_' + analysis)).relative_to(root)))
    for name in ['input', 'splits', 'coordinates']:
        assert (root / unit[name]).is_file(), unit[name]
    units.append(unit)
for name in ['published_downstream', 'published_downstream_allcells', 'published_protein']:
    out = round4 / name
    for directory in ['logs', 'tmp']:
        (out / directory).mkdir(parents=True, exist_ok=True)
    if name != 'published_protein':
        text = json.dumps(units if name == 'published_downstream' else units[:8], indent=2) + '\n'
        path = out / 'units.json'
        if path.exists():
            assert json.loads(path.read_text()) == json.loads(text), path
        else:
            path.write_text(text)
print('Prepared ten published-method inputs and eight all-cell refit inputs')
