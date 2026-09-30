import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path.cwd()/'scripts'))
import calibrated_selective_fill as selector
original = selector.cross_fitted_calibration
out = Path(sys.argv[sys.argv.index('--output-dir')+1])
def capture(*args, **kwargs):
    result = original(*args, **kwargs)
    iso = result[0]
    np.savez(out/'isotonic_map.npz', score=iso.X_thresholds_, probability=iso.y_thresholds_)
    return result
selector.cross_fitted_calibration = capture
selector.main()
