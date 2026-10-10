from pathlib import Path
import hashlib
import json
import subprocess

import numpy as np
import pandas as pd

from workflow import OUT, REPO, MAIN, UNITS, DATASETS, COMPARATORS, require_parity, write_json

require_parity()
for path, digest in json.loads((OUT / "source_sha256.json").read_text()).items():
    assert hashlib.file_digest((REPO / path).open("rb"), "sha256").hexdigest() == digest, path

provenance = []
for unit in UNITS:
    validation = json.loads((OUT / "alra" / unit["key"] / "validation.json").read_text())
    assert validation["valid"]
    for method in COMPARATORS:
        path = REPO / unit["contracts"][method + " (transductive)"]
        metadata = json.loads((path / "metadata.json").read_text())
        parameters = metadata["parameters"]
        assert metadata["scale"] == "counts" and parameters["test_used_for_fit"]
        assert metadata["seed"] == 1729
        assert parameters["fit_cells"] == validation["shape"][0]
        if method != "ALRA":
            assert parameters["transductive"] and not parameters["test_labels_used_for_fit"]
            assert parameters["cross_fitting"] == "none: training and held-out cells enter the teachers in the same way"
            assert metadata["input_sha256"] == validation["sha256"][str(REPO / unit["corrupted"])]
            assert metadata["coordinates_sha256"] == hashlib.file_digest((REPO / unit["coordinates"]).open("rb"), "sha256").hexdigest()
        values = np.load(path / "mean.npy", mmap_mode="r")
        assert list(values.shape) == validation["shape"]
        assert np.isfinite(values).all() and (values >= 0).all()
        provenance.append({"unit": unit["key"], "method": method, "path": str(path),
                           "parameters": parameters,
                           "mean_sha256": hashlib.file_digest((path / "mean.npy").open("rb"), "sha256").hexdigest()})

absolute = pd.concat([pd.read_csv(OUT / "evaluate" / d / "absolute.csv") for d in DATASETS])
paired = pd.concat([pd.read_csv(OUT / "evaluate" / d / "paired_differences.csv") for d in DATASETS])
for dataset, n in zip(DATASETS, (24, 34, 66)):
    assert (absolute.loc[absolute.dataset == dataset, "n_units"] == n).all()
    assert json.loads((OUT / "evaluate" / dataset / "parity.json").read_text())["exact"]
absolute.to_csv(OUT / "absolute.csv", index=False, float_format="%.4f")
paired.to_csv(OUT / "paired_differences.csv", index=False, float_format="%.4f")
write_json(OUT / "audit.json", {"complete": True, "source_hashes_unchanged": True,
                               "exact_inductive_parity": True, "contracts": provenance})

def cell(table, statistic, suffix):
    row = table.loc[table.statistic == statistic].iloc[0]
    return f'{row["estimate_"+suffix]:.4f} [{row["lower_"+suffix]:.4f}, {row["upper_"+suffix]:.4f}]'

stats = ("mean_f1_1_to_10", "mean_f1_1_to_2", "average_precision")
lines = [
    "# Mode-matched comparators for Table 1", "",
    "We use the Table 1 rebuilt pancreas folds, colon_0–2 covering all 34 donors, and rebuilt Norman CRISPRa, with unchanged masks, held-out candidates, units, settings, and seeds.",
    "Transductive SVD and weighted kNN reuse the all-cell teachers used by transductive Safe Fusion, and transductive ALRA fits the paper runner to all masked cells with eight BLAS threads and the original automatic rank rule.",
    "Every comparator ranks by its count-scale value with the original random tie seed, and the unchanged evaluator computes paired 95% percentile intervals using 2,000 biological-unit draws with seed 1729.", "",
    "All differences below are **transductive Safe Fusion minus the comparator**, in percentage points. The comparator's fitting mode is explicit. F1 1%–10% averages ten integer fractions. F1 1%–2% averages eleven fractions at 0.1% steps, as in Supplementary Table S5. AP uses the paper's rank-based definition, including random tie breaking and fold pooling. Positive differences favor Safe Fusion.", "",
    "| Dataset | Comparator | Mode | Δ F1 1%–10% [95% CI] | Δ F1 1%–2% [95% CI] | Δ AP [95% CI] |",
    "| --- | --- | --- | --- | --- | --- |",
]
for dataset in DATASETS:
    for method in COMPARATORS:
        for mode in ("Inductive", "Transductive"):
            name = method + (" (transductive)" if mode == "Transductive" else "")
            table = paired.loc[(paired.dataset == dataset) & (paired.reference == "Safe Fusion (transductive)") & (paired.comparator == name)]
            lines.append(f"| {dataset} | {method} | {mode} | " + " | ".join(cell(table, stat, "pp") for stat in stats) + " |")
lines += ["", "Absolute transductive Safe Fusion performance, in percent:", "",
          "| Dataset | Biological units | F1 1%–10% [95% CI] | F1 1%–2% [95% CI] | AP [95% CI] |",
          "| --- | --- | --- | --- | --- |"]
for dataset, n in zip(DATASETS, (24, 34, 66)):
    table = absolute.loc[(absolute.dataset == dataset) & (absolute.method == "Safe Fusion (transductive)")]
    lines.append(f"| {dataset} | {n} | " + " | ".join(cell(table, stat, "percent") for stat in stats) + " |")

lines += ["", "CRISPRa retains the source evaluator's 66 bootstrap groups: 65 activation targets plus the `none` control group. The manuscript's 65-target description excludes this control group; no group was removed here.", "",
          "Validation: before fitting ALRA, the evaluator exactly reproduced all three existing inductive comparator rows, both Safe Fusion references, every saved estimate and interval, and all corresponding biological-unit counts. Each dataset passed 15 absolute rows and 24 paired-difference rows at the source CSV's four-decimal precision, with zero difference. The final evaluation also reproduced every one of the 2,001 stored statistic values per existing method exactly, including the observed value and all bootstrap draws.", "",
          "All paths below are relative to `artifacts/paper_evidence/review_round4/transductive_comparators/`:", "",
          "- `absolute.csv`, `paired_differences.csv`: all three statistics, both Safe Fusion references, and both comparator modes.",
          "- `parity/<dataset>/`: independent reproduction of existing rows, unit counts, bootstrap values, and `parity.json`.",
          "- `evaluate/<dataset>/`: complete mode-matched evaluation, `summary_table.csv`, unit counts, bootstrap values, and repeat-parity checks.",
          "- `alra/<unit>/`: count matrices, fit metadata, exact command, thread and output validation, and input/output hashes.",
          "- `units_manifest.json`: original masks, split roles, tie seeds, inductive contracts, and transductive comparator paths.",
          "- `source_sha256.json`, `audit.json`: source immutability checks and reused teacher provenance with matrix hashes.",
          "- `code/run_alra_baseline.py`, `code/alra_additive.patch`: output-local paper runner with additive `--transductive`; all numerical settings and the automatic rank rule are unchanged.",
          "- `code/workflow.py`, `code/run.sh`, `code/report.py`, `code/submit.sh`: evaluator wrapper and Slurm reproduction.",
          "- `code/wait.sh`: ten-minute polling loop and final Slurm accounting capture.",
          "- `jobs.tsv`, `jobs_final.tsv`, `logs/`: Slurm job IDs, resource/accounting records, and logs.", "",
          "Source references: `../transductive_main/masked/<dataset>/` and its selector overlay. Reused teachers are in `artifacts/paper_evidence/review_round2/fusion_value/transductive/` for pancreas/Norman and `artifacts/paper_evidence/review_round3/colon_crossfit/fusion_value/transductive/` for colon. These are the teachers consumed by the Table 1 transductive Safe Fusion run.", "",
          "Reproduce from `the repository root` after restoring these input artifacts. Completed output directories are exclusive. The exact submitted dependency chain is in `code/submit.sh`; this command creates a fresh sibling copy and submits the same work:", "",
          "```bash",
          "O=artifacts/paper_evidence/review_round4/transductive_comparators",
          "N=artifacts/paper_evidence/review_round4/transductive_comparators_reproduce",
          "mkdir \"$N\"",
          "cp -r \"$O/code\" \"$O/units_manifest.json\" \"$O/source_sha256.json\" \"$N/\"",
          "sed -i 's@review_round4/transductive_comparators@review_round4/transductive_comparators_reproduce@g' \"$N/units_manifest.json\" \"$N/code/run.sh\" \"$N/code/submit.sh\" \"$N/code/wait.sh\"",
          "O=$N",
          "bash \"$O/code/submit.sh\"",
          "bash \"$O/code/wait.sh\"",
          "```", "",
          "The submission script runs parity first, seven ALRA fits after successful parity, three dataset evaluations after successful fits, and the report after successful evaluation. Every workload uses account `torch_pr_634_general`, partition `cs`, and eight CPUs. No tuning or model refitting beyond the requested ALRA fits occurred.", "",
          "Failures: none. All requested fits, evaluations, parity checks, and artifact audits completed. No work remains. Existing jobs named `scmp-*`, `cx-*`, or `mm3-*` were untouched. No manuscript, documentation, repository README, original output, commit, or remote branch was changed.", ""]
with (OUT / "README.md").open("x") as stream:
    stream.write("\n".join(lines))
print("\n".join(lines))
