from pathlib import Path
import argparse
import gc
import json
import sys
sys.path[:0] = ['scripts', 'scripts/standard_imputers', 'src']
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import evaluate_knockdown_zero_analyses as kd
from selector_attribution import exact_topk

OUT = Path('artifacts/paper_evidence/review_round4/label_baselines/screens')
VIEW = Path('artifacts/paper_evidence/review_round4/transductive_references/knockdown/evaluation_inputs')
DATASETS = ('adamson_crispri','papalexi_eccite','norman_crispra')
LABELS = {'knn_condition':'Weighted kNN teacher, labels','scvi_condition':'scVI teacher, labels',
          'label_detection':'Per-label expected-count detection','safe_fusion_condition':'Safe Fusion, labels'}

def teacher_ranking(name, value, screen):

    rows, cols = np.where((screen.recorded == 0) & screen.test[:,None])
    scores = np.maximum(value[rows,cols],0.)
    order = np.full(screen.recorded.shape,np.inf,dtype=np.float32)
    for pct in reversed(kd.FRACTIONS):
        chosen = exact_topk(scores,max(1,int(round(pct/100*len(scores))))) & (scores > 0)
        order[rows[chosen],cols[chosen]] = pct
    return kd.Method(name,order,value,value,None)

def expected_detection(counts, fitting, labels, library):

    probabilities = np.zeros(counts.shape, dtype=np.float64)
    details = []
    for label in sorted(set(labels)):
        fit = fitting & (labels == label)
        if not fit.any():
            raise ValueError(f'No fitting cells for label {label}')
        totals = np.asarray(counts[fit].sum(axis=0, dtype=np.float64)).ravel().astype(np.float64)
        if not totals.sum() > 0:
            raise ValueError(f'Zero fitting library for label {label}')
        members = labels == label
        probabilities[members] = -np.expm1(-library[members,None] * (totals / totals.sum())[None,:])
        details.append(dict(label=str(label),fitting_cells=int(fit.sum()),fitting_counts=float(totals.sum())))
    assert np.isfinite(probabilities).all()
    return probabilities, details

def selfcheck():
    counts = np.array([[2.,1.],[0.,99.],[1.,3.],[99.,0.]])
    fitting = np.array([True,False,True,False])
    labels = np.array(['a','a','b','b'])
    library = np.array([3.,6.,4.,8.])
    p, _ = expected_detection(counts,fitting,labels,library)
    np.testing.assert_allclose(p[1],-np.expm1(-np.array([4.,2.])))
    np.testing.assert_allclose(p[3],-np.expm1(-np.array([2.,6.])))
    changed = counts.copy(); changed[~fitting] = 999
    np.testing.assert_array_equal(p,expected_detection(changed,fitting,labels,library)[0])


def main():
    selfcheck()
    verification = json.loads((OUT/'verification_receipt.json').read_text())
    assert verification['saved_evaluators_match']
    assert json.loads((OUT.parent/'verification_passed.json').read_text())['passed'], 'Combined evaluator gate failed'
    args = argparse.Namespace(deploy_root=str(VIEW/'deployment'), review_root=str(VIEW/'review'),
        norman_root=str(VIEW/'norman'), norman_benchmark='artifacts/paper_evidence/review_round2/leakage_free/norman_crispra')
    rows, provenance = [], []
    for dataset in DATASETS:
        inputs = kd.screen_inputs(dataset,args)
        screen = kd.load_screen(dataset,inputs.deploy,inputs.splits,inputs.prepared)
        targets = kd.eligible_targets(screen,0.2)
        assert len(targets) == dict(zip(DATASETS,[26,6,41]))[dataset]
        library = screen.recorded.sum(axis=1,dtype=np.float64)
        columns = np.array([item.g for item in targets])
        orders = kd.first_fill_order(screen,inputs.deploy)
        methods = {'safe_fusion_condition': kd.Method('safe_fusion_condition',orders['safe_fusion_condition'],None,None,None)}
        for name in ['knn_condition','scvi_condition']:
            value = kd.count_scale_contract(inputs.deploy/'methods'/kd.TEACHER_VALUES[name],screen,library)
            methods[name] = teacher_ranking(name,value,screen)
        hybrid = ad.read_h5ad(inputs.deploy/'hybrid.h5ad')
        assert hybrid.obs_names.astype(str).tolist() == screen.cell_ids.tolist()
        assert np.array_equal(hybrid.obs['target'].astype(str).to_numpy(),screen.target)
        prepared_axis = ad.read_h5ad(inputs.prepared, backed='r')
        assert hybrid.var_names.astype(str).tolist() == prepared_axis.var_names.astype(str).tolist()
        prepared_axis.file.close()
        for name in ['knn_condition','scvi_condition']:
            metadata = json.loads((inputs.deploy/'methods'/kd.TEACHER_VALUES[name]/'metadata.json').read_text())
            assert metadata['gene_ids'] == hybrid.var_names.astype(str).tolist()
        fitting = screen.split == 'development'
        assert fitting.any() and not np.any(fitting & screen.test)
        probability, detail = expected_detection(hybrid.layers['corrupted_counts'],fitting,screen.target,library)
        methods['label_detection'] = teacher_ranking('label_detection',probability,screen)
        provenance.append(dict(dataset=dataset,prepared=str(inputs.prepared),splits=str(inputs.splits),deployment=str(inputs.deploy),
            fitting_layer='hybrid.h5ad:corrupted_counts',fitting_split='development',library='recorded counts over gene panel',
            per_label=detail,target_genes=[t.gene for t in targets],
            teacher_first_fill_matches_saved={m:bool(np.array_equal(methods[m].order,orders[m])) for m in ['knn_condition','scvi_condition']}))
        assert all(provenance[-1]['teacher_first_fill_matches_saved'].values()), provenance[-1]['teacher_first_fill_matches_saved']
        for item in targets:
            y = kd.likely_dropout_labels(screen,item)
            for name,method in methods.items():
                order = method.order[item.cells,item.g]
                score = np.where(np.isfinite(order),11-order,0.)
                strata,pairs = kd.stratified_auroc(y,score,np.log(library[item.cells]),5)
                rows.append(dict(dataset=dataset,gene=item.gene,method=name,none=kd.auroc(y,score),depth_strata=strata,
                                 strata_pair_share=pairs,n_control_zeros=len(item.control_zero),n_perturbed_zeros=len(item.perturbed_zero)))
        print(json.dumps({'dataset':dataset,'targets':len(targets),'teacher_first_fill_matches_saved':provenance[-1]['teacher_first_fill_matches_saved']}),flush=True)
        del screen,hybrid,methods,orders,probability,value
        gc.collect()
    per_target = pd.DataFrame(rows)
    per_target.to_csv(OUT/'baseline_per_target.csv',index=False)
    summary = []
    for dataset in DATASETS:
        data = per_target[per_target.dataset == dataset]
        genes = data.gene.drop_duplicates().tolist()
        draws = np.random.default_rng([1729,kd.SCOPES.index(dataset)]).integers(0,len(genes),size=(2000,len(genes)))
        reference = data[data.method == 'safe_fusion_condition'].set_index('gene').loc[genes]
        for name in LABELS:
            part = data[data.method == name].set_index('gene').loc[genes]
            for metric in ['none','depth_strata']:
                est,low,high = kd.interval(part[metric].to_numpy(),draws)
                diff,dlow,dhigh = kd.interval((reference[metric]-part[metric]).to_numpy(),draws)
                summary.append(dict(dataset=dataset,method=name,adjustment=metric,n_targets=len(genes),value=est,ci_low=low,ci_high=high,
                                    safe_fusion_labels_minus_method=diff,difference_ci_low=dlow,difference_ci_high=dhigh))
    result = pd.DataFrame(summary)
    result.to_csv(OUT/'baseline_summary.csv',index=False)
    (OUT/'baseline_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    lines=['## Screen verification and known-zero baselines','',
        f"The unchanged evaluators reproduce all saved target tables within 1e-12, the saved bootstrap report exactly, and all comparator summaries. They reproduce {verification['paper_matches']} of {verification['paper_cells']} printed cell strings in Supplementary Table S16. Four AUROC cells differ after direct rounding; the fifth discrepancy is only -0.00 versus 0.00 for the Adamson label-aware Safe Fusion effect shift. All available effect shifts agree numerically. The baseline release gate verifies computational reproduction and does not claim exact manuscript equality.", '',
        'The four AUROC discrepancies (recomputed; paper) are DCA zero-inflated Norman within-quintile 0.574731 (0.57; 0.58), scVI zero-inflated Papalexi within-quintile 0.534820 (0.53; 0.54), scImpute dropout probability Papalexi unadjusted 0.504664 (0.50; 0.51), and Safe Fusion with labels Papalexi unadjusted 0.554925 (0.55; 0.56). These are consistent with an intermediate rounding to three decimals before two, but no documented intermediate rounding was established. The manuscript remains unchanged.', '',
        'Known-zero eligibility retains 26 Adamson, 6 Papalexi and 41 Norman targets. Positive zeros are control zeros for Adamson/Papalexi and activated-cell zeros for Norman. Each ranking uses global held-out recorded-zero budgets from 1% through 10%, the original teacher exact_topk deterministic tie convention, and the existing first-fill score (11 minus first fill percent; zero if unfilled). Teacher first-fill arrays are required to match every saved entry exactly. Library adjustment uses the existing pair-weighted within-quintile AUROC. Each screen resamples targets with 2,000 paired draws and the original screen-specific seed [1729, screen index].', '',
        'The detection baseline pools masked development counts by perturbation target, divides each gene count total by its label total, and computes 1-exp(-L*a) using the held-out recorded library. This adds no fitting, tuning, masking or seed changes. Teacher rankings use saved label-aware count-scale values; Safe Fusion uses its saved label-aware fills.', '',
        '| Screen | Ranking | Unadjusted AUROC [95% CI] | Within quintiles [95% CI] |', '|---|---|---|---|']
    def fmt(r,prefix='value'):
        return f"{r.value:.4f} [{r.ci_low:.4f}, {r.ci_high:.4f}]"
    for dataset in DATASETS:
        for name,label in LABELS.items():
            part = result[(result.dataset==dataset)&(result.method==name)].set_index('adjustment')
            lines.append(f'| {dataset} | {label} | {fmt(part.loc["none"])} | {fmt(part.loc["depth_strata"])} |')
    lines += ['', 'Paired differences are Safe Fusion with labels minus the listed ranking.', '',
              '| Screen | Ranking | Unadjusted difference [95% CI] | Within-quintile difference [95% CI] |','|---|---|---|---|']
    for dataset in DATASETS:
        for name,label in LABELS.items():
            if name=='safe_fusion_condition':continue
            part = result[(result.dataset==dataset)&(result.method==name)].set_index('adjustment')
            def delta(row):return f'{row.safe_fusion_labels_minus_method:.4f} [{row.difference_ci_low:.4f}, {row.difference_ci_high:.4f}]'
            lines.append(f'| {dataset} | {label} | {delta(part.loc["none"])} | {delta(part.loc["depth_strata"])} |')
    lines += ['', 'Artifacts: `screens/verification_receipt.json`, `screens/s16_paper_cells.csv`, `screens/verification_transductive/`, `screens/verification_inductive_comparators/`, `screens/baseline_per_target.csv`, `screens/baseline_summary.csv`, `screens/baseline_provenance.json`. All paths are relative to the requested output directory.', '',
              'Reproduce from the repository root (new output paths are required for an independent replay):', '', '```bash',
              'sbatch -A torch_pr_634_general -p cs -c 2 --mem=64G --time=04:00:00 -o artifacts/paper_evidence/review_round4/label_baselines/screens/verify_%j.log artifacts/paper_evidence/review_round4/label_baselines/screens/verify.sh',
              'sbatch -A torch_pr_634_general -p cs -c 2 --mem=64G --time=02:00:00 -o artifacts/paper_evidence/review_round4/label_baselines/screens/baselines_%j.log artifacts/paper_evidence/review_round4/label_baselines/screens/baselines.sh','```','']
    (OUT/'summary.md').write_text('\n'.join(lines))
    print(result.to_string(index=False),flush=True)

if __name__ == '__main__': main()
