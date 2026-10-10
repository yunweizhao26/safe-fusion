import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path.cwd()/'scripts'))
import sle_common as sc
from calibrated_selective_fill import detection_probability
from sklearn.isotonic import IsotonicRegression
root = Path('artifacts/paper_evidence/review_round4/sle_transductive').resolve()
unit_name = json.loads((root/'units.json').read_text())[int(sys.argv[1])]
s, fold, app = unit_name.split('/')
sc.CASE = root/'conditional'
u = sc.load_unit(s, int(fold.split('_')[1]), app, sc.CASE)
fills = sc.load_fills(u, 'Safe Fusion')
score = sc.selector_scores(u, fills['rows'], fills['cols'])
with np.load(u.directory/'selector/isotonic_map.npz') as iso:
    calibration = IsotonicRegression(y_min=0., y_max=1., out_of_bounds='clip')
    calibration.X_thresholds_ = iso['score']
    calibration.y_thresholds_ = iso['probability']
    calibration.X_min_, calibration.X_max_ = iso['score'][[0,-1]]
    calibration._build_f(calibration.X_thresholds_, calibration.y_thresholds_)
    probability = calibration.predict(score).astype(np.float32)
detection = detection_probability(probability, .1).astype(np.float32)
assert np.isfinite(detection).all() and (detection >= 0).all() and (detection <= 1).all()
original = fills['value']
fills['value'] = (original * detection).astype(np.float32)
assert (fills['value'] <= original).all()
out = root/'conditional_detection'/unit_name/'fills'
out.mkdir(parents=True, exist_ok=True)
np.savez(out/'Safe_Fusion.npz', **fills)
np.save(out/'detection_probability.npy', detection)
for p in (u.directory/'fills').iterdir():
    if p.name not in ('Safe_Fusion.npz','report.json') and not (out/p.name).exists(): (out/p.name).symlink_to(p)
(out/'report.json').write_text(json.dumps({'value':'conditional_detection','rho':.1,'calibration':'production five-fold fitting-cell isotonic','ranking_and_budgets':'unchanged conditional selector', 'zero_values':int((fills['value']==0).sum()), 'candidates':len(detection)},indent=2)+'\n')
