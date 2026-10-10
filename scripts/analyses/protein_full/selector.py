import ast
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
import selector_attribution as a
import calibrated_selective_fill as production
original = a.stratified_fit_indices
records = []

def fit_indices(labels, maximum, seed):
    if int(np.asarray(labels).sum()) <= maximum // 2:
        return original(labels, maximum, seed)
    rng = np.random.default_rng(seed)
    half = maximum // 2
    keep = np.sort(np.concatenate([
        rng.choice(np.flatnonzero(labels == 1), size=half, replace=False),
        rng.choice(np.flatnonzero(labels == 0), size=maximum - half, replace=False),
    ]))
    records.append(dict(available_positive=int(labels.sum()), available_negative=int((labels == 0).sum()),
                        selected_positive=int(labels[keep].sum()), selected_negative=int((labels[keep] == 0).sum()),
                        maximum=maximum, seed=seed, indices_sha256=hashlib.sha256(keep.tobytes()).hexdigest()))
    return keep

def selftest():

    from types import SimpleNamespace
    tree = ast.parse((ROOT / 'scripts/scale_selector.py').read_text())
    block = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == 'int(fit_labels.sum()) > args.max_fit_rows // 2')
    for positives in (5, 8, 12):
        y = np.r_[np.ones(positives, dtype=np.int8), np.zeros(20, dtype=np.int8)]
        env = dict(np=np, fit_labels=y.copy(), fit_rows=np.arange(len(y)), fit_cols=np.zeros(len(y)),
                   args=SimpleNamespace(max_fit_rows=8, seed=1729), sampling='')
        exec(compile(ast.Module(body=[block], type_ignores=[]), '<S8 source block>', 'exec'), env)
        got = fit_indices(y, 8, 1729)
        np.testing.assert_array_equal(got, env['fit_rows'])
        assert len(got) == len(np.unique(got)) == 8 and y[got].sum() == 4
    for positives in (2, 4):
        y = np.r_[np.ones(positives), np.zeros(20)]
        np.testing.assert_array_equal(fit_indices(y, 8, 1729), original(y, 8, 1729))
    print('PASS: exact S8 sampling indices, half-cap boundary, deterministic sampling.')

if __name__ == '__main__':
    if '--selftest' in sys.argv:
        selftest()
    else:
        if '--scale-balanced' in sys.argv:
            sys.argv.remove('--scale-balanced')
            a.stratified_fit_indices = fit_indices
        out = Path(sys.argv[sys.argv.index('--output-dir') + 1])
        production.main()
        assert len(records) == 1 and records[0]['available_positive'] == 2059109
        assert records[0]['selected_positive'] == records[0]['selected_negative'] == 1000000
        (out / 'sampling.json').write_text(json.dumps({'source': 'scripts/scale_selector.py', 'calls': records}, indent=2) + '\n')
