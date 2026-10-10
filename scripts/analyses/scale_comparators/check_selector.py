#!/usr/bin/env python3

from pathlib import Path
import json
import numpy as np
ROOT=(Path.cwd() / 'artifacts/paper_evidence/review_round4/scale_comparators')
with np.load(ROOT/'checks/selector_scvi/test_scores.npz') as standard, np.load(ROOT/'cells_25000/selector_scvi/test_candidates.npz') as block:
    for key in ('rows','cols','labels'):
        np.testing.assert_array_equal(standard[key],block[key])
    scores=np.load(ROOT/'cells_25000/selector_scvi/test_score.npy')
    np.testing.assert_array_equal(scores,standard['score'])
    result={'candidates':len(scores),'max_score_error':0,'exact':True}
(ROOT/'checks/selector_PASS.json').write_text(json.dumps(result,indent=2)+'\n')
print(result)
