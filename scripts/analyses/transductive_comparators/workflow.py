from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

OUT = (Path.cwd() / 'artifacts/paper_evidence/review_round4/transductive_comparators')
REPO = OUT.parents[3]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src")]
MAIN = OUT.parent / "transductive_main"
DATASETS = ("Pancreas", "Colon", "CRISPRa")
COMPARATORS = ("SVD", "Weighted kNN", "ALRA")
UNITS = json.loads((OUT / "units_manifest.json").read_text())


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def require_parity():
    for dataset in DATASETS:
        assert json.loads((OUT / "parity" / dataset / "parity.json").read_text())["exact"]


def check_parity(destination, source):
    import pandas as pd
    checks = []
    for name, keys in [
        ("absolute.csv", ["dataset", "method", "statistic"]),
        ("paired_differences.csv", ["dataset", "reference", "comparator", "statistic"]),
    ]:
        new = pd.read_csv(destination / name).set_index(keys).sort_index()
        old = pd.read_csv(source / name).set_index(keys).sort_index().loc[new.index]
        pd.testing.assert_frame_equal(new, old, check_exact=True)
        checks.append({"file": name, "rows": len(new), "maximum_absolute_difference": 0})
    new = pd.read_parquet(destination / "unit_counts.parquet")
    old = pd.read_parquet(source / "unit_counts.parquet")
    old = old.loc[old.method.isin(new.method.unique())]
    keys = ["dataset", "method", "group", "unit", "fraction"]
    pd.testing.assert_frame_equal(new.set_index(keys).sort_index(), old.set_index(keys).sort_index(), check_exact=True)
    checks.append({"file": "unit_counts.parquet", "rows": len(new), "maximum_absolute_difference": 0})
    write_json(destination / "parity.json", {"exact": True, "source": str(source), "checks": checks})


def evaluate(stage, index):
    import numpy as np
    import fusion_value_bootstrap as bootstrap
    if stage == "evaluate":
        require_parity()
    dataset = DATASETS[index]
    destination = OUT / stage / dataset
    destination.mkdir(parents=True, exist_ok=False)
    methods = {"Safe Fusion": "selector_fusion", "Safe Fusion (transductive)": "selector_transductive"}
    methods.update({name: "table_1" for name in COMPARATORS})
    if stage == "evaluate":
        methods.update({name + " (transductive)": "transductive_comparator" for name in COMPARATORS})
    bootstrap.method_families = lambda: methods
    captured = {}
    original = bootstrap.statistics_from_sums

    def capture(sums):
        result = original(sums)
        method = list(methods)[len(captured)]
        captured[method] = result
        return result

    bootstrap.statistics_from_sums = capture
    sys.argv = ["fusion_value_bootstrap.py", "--units-manifest", str(OUT / "units_manifest.json"),
                "--unit-keys", *[u["key"] for u in UNITS if u["dataset"] == dataset],
                "--selector-root", str(MAIN / "masked/selector_overlay"),
                "--output-dir", str(destination), "--draws", "2000", "--seed", "1729"]
    bootstrap.main()
    np.savez(destination / "bootstrap_statistics.npz",
             **{f"{method}|{stat}": values for method, stats in captured.items() for stat, values in stats.items()})
    if stage == "parity":
        check_parity(destination, MAIN / "masked" / dataset)
    else:

        with np.load(OUT / "parity" / dataset / "bootstrap_statistics.npz") as old:
            for name in old.files:
                method, stat = name.split("|")
                np.testing.assert_array_equal(old[name], captured[method][stat])
        write_json(destination / "parity.json", {"exact": True, "all_2001_draw_values_unchanged": True})


def fit_alra(index):
    import numpy as np
    from masked_f1_units import load_units_manifest, load_unit, contract_metadata
    from threadpoolctl import threadpool_info
    require_parity()
    unit = load_units_manifest(OUT / "units_manifest.json")[index]
    destination = OUT / "alra" / unit.key
    destination.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(OUT / "code/run_alra_baseline.py"),
               "--corrupted", str(unit.corrupted), "--splits", str(unit.splits),
               "--output", str(destination), "--components", "100", "--power-iterations", "2",
               "--quantile-prob", "0.001", "--seed", "1729", "--transductive"]
    write_json(destination / "command.json", command)
    pools = threadpool_info()
    assert all(pool["num_threads"] == 8 for pool in pools if pool["user_api"] == "blas"), pools
    subprocess.run(command, check=True)
    data = load_unit(unit)
    metadata = contract_metadata(destination, data)
    mean = np.load(destination / "mean.npy", mmap_mode="r")
    assert mean.shape == data.counts.shape and mean.dtype == np.float32
    assert np.isfinite(mean).all() and (mean >= 0).all()
    assert metadata["scale"] == "counts" and metadata["seed"] == 1729
    assert metadata["parameters"]["fit_cells"] == len(data.counts)
    assert metadata["parameters"]["test_used_for_fit"] is True
    write_json(destination / "validation.json", {
        "valid": True, "shape": list(mean.shape), "parameters": metadata["parameters"],
        "threadpools": pools, "job_id": os.environ.get("SLURM_JOB_ID"),
        "sha256": {str(path): hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
                   for path in [unit.corrupted, unit.splits, destination / "mean.npy"]},
    })


if __name__ == "__main__":
    stage = sys.argv[1]
    index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    if stage in ("parity", "evaluate"):
        evaluate(stage, index)
    elif stage == "alra":
        fit_alra(index)
    else:
        raise ValueError(stage)
