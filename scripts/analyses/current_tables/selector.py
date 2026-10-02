import sys
from pathlib import Path
root=Path.cwd()
sys.path[:0]=[str(root/'scripts'),str(root/'src')]
source=root/'scripts/calibrated_selective_fill.py'
code=source.read_text()
needle='        test_detection = detection_probability(test_probability, rho)'
assert code.count(needle)==1
code=code.replace(needle,needle+'\n        np.savez(output_root / "detection.npz", rows=test_rows, cols=test_cols, scores=test_score, probability=test_probability, detection=test_detection, labels=test_labels)')
exec(compile(code,str(source),'exec'),dict(__name__='__main__',__file__=str(source)))
