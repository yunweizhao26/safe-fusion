from common import *
import json
import anndata as ad
import numpy as np
import pandas as pd
import evaluate_thinning_transfer as t

def specs():
    result = []
    for u in units():
        levels = ['050','025'] if u.dataset == 'Colon' else ['050']
        for level in levels:
            if u.dataset == 'CRISPRa':
                key = 'norman_thinning_050'
                root = E/'review_round2/thinning_transfer'
            else:
                key = f'{u.dataset.lower()}_thinning_{level}_{u.key[-1]}'
                root = CC/'thinning' if u.dataset == 'Colon' else OUT/'pancreas_rebuilt_thinning'
            result.append((u, key, root, float(int(level))/100))
    return result

def evaluate():
    grouped = {}
    for u,key,root,p in specs():
        unit = dict(key=key, splits=u.splits, thin_root=root/'thinning_trained'/key,
                    thin_stacked=root/'thinning_trained/stacked'/key, mask_root=root/'mask_trained'/key)
        dataset = dict(comparators=root/'thinning_trained/comparators'/key)
        truth = ad.read_h5ad(u.truth)
        data = ad.read_h5ad(root/'data'/key/'corrupted.h5ad')
        assert np.array_equal(data.obs_names,truth.obs_names) and np.array_equal(data.var_names,truth.var_names)
        args = (t.dense(truth.layers['counts']).astype(np.float32), t.dense(data.layers['corrupted_counts']).astype(np.float32),
                data.obs_names.astype(str).to_numpy(), truth.obs[u.unit_column].astype(str).to_numpy())
        original = t.unit_table(dataset,unit,*args)
        previous = t.SELECTORS
        t.SELECTORS = {'Safe Fusion':None}
        newunit = dict(unit, thin_root=OUT/'thinning'/key/'thinning-trained',mask_root=OUT/'thinning'/key/'mask-trained')
        new = t.unit_table(dataset,newunit,*args)
        t.SELECTORS = previous
        for field in ['counts','count_one_hits','strata','average_precision']:
            for design in t.DESIGNS:
                name = f'Safe Fusion, {design}'
                original[field][name.replace('Safe Fusion','Transductive Safe Fusion')] = new[field][name]
        grouped.setdefault(f'{u.dataset}, {int(p*100)}%',[]).append(original)
    dest = OUT/'thinning_evaluation'
    dest.mkdir(exist_ok=True)
    rows, diffs, parity = [], [], []
    for dataset,tables in grouped.items():
        result = t.analyze(tables,2000,7)

        reference_root = (CC/'thinning' if dataset.startswith('Colon') else
                          OUT/'pancreas_rebuilt_thinning' if dataset.startswith('Pancreas') else E/'review_round2/thinning_transfer')
        old = pd.read_csv(reference_root/'evaluation/transfer_summary.csv')
        label = ('norman' if dataset.startswith('CRISPRa') else dataset.split(',')[0].lower())+'_thinning_'+('025' if '25%' in dataset else '050')
        old = old[old.dataset == label]
        fresh = pd.DataFrame(result['rows'])
        joined = old.merge(fresh,on='method',suffixes=('_old','_new'))
        cols = ['mean_f1_1_10','average_precision','f1_1','f1_5','f1_10']
        error = max(float(np.max(np.abs(joined[c+'_old']-joined[c+'_new']))) for c in cols)
        assert len(joined) == len(old) and error < 1e-8, (dataset,error)
        parity.append(dict(dataset=dataset,rows=len(old),max_difference=error))

        n = sum(len(x['units']) for x in tables)
        draws = np.random.default_rng(7).integers(0,n,size=(2000,n))
        positive = np.concatenate([x['positives'] for x in tables])
        for row in result['rows']:
            samples=[]
            for b in t.FRACTIONS:
                s = np.concatenate([x['counts'][row['method']][b][0] for x in tables])
                h = np.concatenate([x['counts'][row['method']][b][1] for x in tables])
                samples.append(200*h[draws].sum(1)/(s[draws].sum(1)+positive[draws].sum(1)))
            low,high = np.quantile(np.mean(samples,axis=0),[.025,.975])
            rows.append(dict(dataset=dataset,**row,mean_f1_low=low,mean_f1_high=high))

        for table in tables:
            for field in ['counts','count_one_hits','strata','average_precision']:
                for design in t.DESIGNS:
                    name = f'Safe Fusion, {design}'
                    table[field][name.replace('Safe Fusion','Inductive Safe Fusion')] = table[field].pop(name)
                    table[field][name] = table[field].pop(name.replace('Safe Fusion','Transductive Safe Fusion'))
        trans = t.analyze(tables,2000,7)
        diffs += [dict(dataset=dataset,**r) for r in trans['comparisons']]
    pd.DataFrame(rows).to_csv(dest/'absolute.csv',index=False)
    pd.DataFrame(diffs).to_csv(dest/'paired_differences.csv',index=False)
    (dest/'parity.json').write_text(json.dumps(parity,indent=2)+'\n')

if __name__ == '__main__':
    evaluate()
