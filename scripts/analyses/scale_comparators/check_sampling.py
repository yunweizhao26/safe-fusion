#!/usr/bin/env python3

import json
from pathlib import Path
ROOT=(Path.cwd() / 'artifacts/paper_evidence/review_round4/scale_comparators')
OLD=ROOT.parents[3]/'artifacts/paper_evidence/review_round3/scale'
fields=['architecture','fit_split','fit_cells','test_cells','fit_candidates','fit_masked_positives',
        'fit_sampling','fit_joined_candidates','fit_sample_candidates','fit_sample_positives']
checks=[]
for n in (25000,50000,100000,200000):
    reference=json.loads((OLD/f'cells_{n}/selector/selector_report.json').read_text())
    for method in ('scvi','magic'):
        new=json.loads((ROOT/f'cells_{n}/selector_{method}/selector_report.json').read_text())
        shared=[key for key in fields if key in reference]
        for key in shared:
            assert reference[key]==new[key],(n,method,key)
        for key in ('n_zeros','n_masked_positives'):
            assert reference['test'][key]==new['test'][key],(n,method,key)
        for key in ('kind','fit_rows','model_parameters'):
            assert reference['models']['mlp'][key]==new['models']['mlp'][key],(n,method,key)
        assert not new['test_labels_used_for_fit']
        checks.append({'cells':n,'method':method,'fields_checked':shared,'model_parameters_identical':True,
                       'model_fit_rows_identical':True,'test_candidate_counts_identical':True,
                       'fields_not_recorded_in_S28':[key for key in fields if key not in reference]})
(ROOT/'selector_sampling_PASS.json').write_text(json.dumps(checks,indent=2)+'\n')
print('All eight selectors match S28 model parameters, fit rows and recorded sampling counts.')
