import ast
from pathlib import Path
import subprocess

root = (Path.cwd() / 'artifacts/paper_evidence/review_round4/reviewer_extras/C_scale')
original = ast.parse(Path('scripts/scale_evaluate.py').read_text())
local = ast.parse((root / 'code/scale_evaluate.py').read_text())
functions = lambda tree: {n.name: ast.dump(n, include_attributes=False) for n in tree.body
                          if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name != 'main'}
assert functions(original) == functions(local)
source = (root / 'code/stage.sh').read_text()
assert 'SIZES=(200000)' in source and '--variable-genes 5000' in source
assert 'SEED=1729' in source and '--extension-block-cells 1024' in source
assert 'torch_pr_634_courant' not in source
launcher = (root / 'code/submit.sh').read_text()
assert 'magic_teacher=$(submit teachers 3 192 12 ' in launcher
assert '-A torch_pr_634_general' in launcher and '--gres=gpu:l40s:1' in launcher
for path in (root / 'code').glob('*.sh'):
    subprocess.run(['bash', '-n', str(path)], check=True)
for path in (root / 'code').glob('*.py'):
    ast.parse(path.read_text())
print('Numerical functions unchanged; shell syntax, seed, 5000-gene scope, account, and MAGIC limit verified.')

if '--numerical' in __import__('sys').argv:
    import sys
    sys.path.insert(0, str(Path('scripts').resolve()))
    import numpy as np
    from scipy import sparse
    from scale_selector import zero_entries, is_positive
    counts = sparse.csr_matrix([[0, 2, 0], [0, 0, 1], [3, 0, 0]])
    cells = np.array([0, 2])
    parts = list(zero_entries(counts, cells, 1))
    rows = np.concatenate([r for r, _ in parts])
    cols = np.concatenate([c for _, c in parts])
    rr, cc = np.where(counts.toarray()[cells] == 0)
    assert np.array_equal(rows, cells[rr]) and np.array_equal(cols, cc)
    assert np.array_equal(is_positive(rows, cols, np.array([2, 7]), 3), [False, True, True, False])
    print('Fallback candidates and labels equal dense canonical enumeration.')
