
import run


def collect(stage, pcts):
    pd, np = run.pd, run.np
    old = pd.read_csv(run.BASE / 'summary/thinning_unit_values.csv', dtype={'unit': str})
    frames, checks = [], []
    for dataset, endpoints in run.SPECS.items():
        donors = dataset in ['colon', 'pancreas']
        keys = [f'{dataset}_{i}' for i in range(3)] if donors else [dataset]
        unitcol = 'donor' if donors else 'unit'
        filename = 'donor_metrics.parquet' if donors else 'unit_metrics.parquet'
        for key in keys:
            assert (run.OUT / stage / key / 'DONE').exists()
        frame = pd.concat([pd.read_parquet(run.OUT / stage / key / 'evaluation' / filename) for key in keys])
        for endpoint, metric in endpoints:
            f = frame[frame.metric == metric].copy()
            f[unitcol] = f[unitcol].astype(str)
            pivot = f.pivot(index=unitcol, columns='method', values='value')
            ref = old[(old.dataset == dataset) & (old.endpoint == endpoint) & (old.matrix == 'corrupted_raw')].set_index('unit').reindex(pivot.index)
            assert ref.unthinned.notna().all()
            for column, savedcol in [('corrupted_raw', 'value'), ('reference_truth', 'thinned_truth_pipeline')]:
                delta = float(np.nanmax(abs(pivot[column].to_numpy() - ref[savedcol].to_numpy())))
                checks.append(dict(dataset=dataset, endpoint=endpoint, reference=column,
                    max_difference=delta, original_reference_check_passed=delta < 1e-12))

                assert delta < 1e-12 or (dataset == 'zebrafish' and endpoint == 'Stage error' and delta <= 1e-7), (dataset, endpoint, column, delta)
            for name in ['corrupted_raw'] + [run.name(m, p) for m in run.METHODS for p in pcts]:
                frames.append(pd.DataFrame(dict(dataset=dataset, endpoint=endpoint, unit=pivot.index,
                    stratum=ref.stratum.to_numpy(), matrix=name, value=pivot[name].to_numpy(),
                    unthinned=ref.unthinned.to_numpy(), thinned_truth_pipeline=ref.thinned_truth_pipeline.to_numpy())))
    values = pd.concat(frames, ignore_index=True)
    values.to_csv(run.OUT / f'{stage}_unit_values.csv', index=False)
    pd.DataFrame(checks).to_csv(run.OUT / f'{stage}_reference_parity.csv', index=False)
    return values
