import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path.cwd()/'scripts'))
import selector_attribution as attribution
import calibrated_selective_fill as production
original=attribution.stratified_fit_indices
records=[]

def fit_indices(labels,maximum,seed):
    labels=np.asarray(labels)
    positive=np.flatnonzero(labels==1); negative=np.flatnonzero(labels==0)
    if len(labels)>maximum and len(positive)>=maximum:
        half=maximum//2
        if half<1 or len(negative)<maximum-half:
            raise ValueError('Overflow fallback requires both classes and enough recorded zeros')
        rng=np.random.default_rng(seed)
        indices=np.sort(np.r_[rng.choice(positive,half,replace=False),rng.choice(negative,maximum-half,replace=False)])
        records.append({'available_positive':len(positive),'available_negative':len(negative),'selected_positive':half,'selected_negative':maximum-half,'maximum':maximum,'seed':seed})
        return indices
    return original(labels,maximum,seed)

def selftest():
    y=np.r_[np.ones(12,dtype=int),np.zeros(20,dtype=int)]
    got=fit_indices(y,8,1729);assert len(got)==8 and y[got].sum()==4 and len(np.unique(got))==8
    assert np.array_equal(got,fit_indices(y,8,1729))
    for y,cap in [(np.r_[np.ones(3),np.zeros(20)],8),(np.r_[np.ones(3),np.zeros(4)],8)]:
        assert np.array_equal(fit_indices(y,cap,1729),original(y,cap,1729))
    print('Overflow fallback preserves both classes; ordinary production sampling unchanged.')

if __name__=='__main__':
    if '--selftest' in sys.argv:selftest()
    else:
        if '--positive-overflow-balanced' in sys.argv:
            sys.argv.remove('--positive-overflow-balanced')
            attribution.stratified_fit_indices=fit_indices
        out=Path(sys.argv[sys.argv.index('--output-dir')+1])
        production.main()
        (out/'overflow_sampling.json').write_text(json.dumps({'source':'scripts/analyses/scale_caps/scale_selector.py','changes':'only the otherwise-single-class positive-overflow branch','calls':records},indent=2)+'\n')
