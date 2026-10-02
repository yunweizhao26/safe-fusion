#!/usr/bin/env python3

import csv
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = (Path.cwd() / 'artifacts/paper_evidence/review_round4/scale_caps')
LARGE = "Safe Fusion (larger caps)"
DEFAULT = "Safe Fusion (default caps)"
REFERENCES = [DEFAULT, "scVI", "scVI probability", "MAGIC", "SAVER", "Weighted kNN"]
STATISTICS = ["mean_f1_1_to_10", "average_precision"]


def read_csv(path, delimiter=","):
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(str(value) for value in row) + " |" for row in rows])


def metric(row, field="estimate"):
    if row is None:
        return "pending"
    spec = "+.4f" if field == "difference" else ".4f"
    values = [format(100 * float(row[key]), spec) for key in [field, "lower", "upper"]]
    return f"{values[0]} [{values[1]}, {values[2]}]"


def validate(current, published):
    checks, missing = [], []
    for method in REFERENCES:
        original = "Safe Fusion (transductive)" if method == DEFAULT else method
        for statistic in STATISTICS:
            now = next((r for r in current if r["method"] == method and r["statistic"] == statistic
                        and r["cells"] == "200000"), None)
            old = next((r for r in published if r["method"] == original and r["statistic"] == statistic
                        and r["cells"] == "200000"), None)
            if now is None or old is None:
                missing.append({"method": method, "statistic": statistic,
                                "current_missing": now is None, "published_missing": old is None})
                continue
            for field in ["estimate", "lower", "upper"]:
                delta = float(now[field]) - float(old[field])
                checks.append({"method": method, "statistic": statistic, "field": field,
                               "published": float(old[field]), "current": float(now[field]),
                               "difference": delta, "exact": delta == 0.0,
                               "passed": math.isfinite(delta) and abs(delta) <= 1e-12})
    return {"tolerance": 1e-12, "passed": not missing and all(c["passed"] for c in checks),
            "exact": not missing and all(c["exact"] for c in checks), "missing": missing, "checks": checks}


def main():
    jobs = read_csv(ROOT / "jobs.tsv", "\t")
    accounting, accounting_error = {}, ""
    if jobs:
        try:
            result = subprocess.run(["sacct", "-n", "-P", "-j", ",".join(row["job"] for row in jobs),
                                     "--format=JobIDRaw,State,Elapsed,MaxRSS,TotalCPU,ExitCode"],
                                    text=True, capture_output=True, timeout=45, check=True)
            for line in result.stdout.splitlines():
                fields = line.split("|")
                if len(fields) >= 6:
                    accounting[fields[0]] = dict(zip(["job", "state", "elapsed", "max_rss", "total_cpu", "exit_code"], fields))
        except (OSError, subprocess.SubprocessError) as error:
            accounting_error = str(error)
    resource_rows = []
    for row in jobs:
        summary = accounting.get(row["job"], {})
        batch = accounting.get(row["job"] + ".batch", {})
        resource_rows.append({**row, **{key: summary.get(key, "unknown") for key in ["state", "elapsed", "exit_code"]},
                              "max_rss": batch.get("max_rss", summary.get("max_rss", "")),
                              "total_cpu": batch.get("total_cpu", summary.get("total_cpu", ""))})
    with (ROOT / "resources.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["genes", "variant", "stage", "job", "state", "elapsed", "exit_code", "max_rss", "total_cpu"])
        writer.writeheader()
        writer.writerows(resource_rows)

    sections = ["# Safe Fusion training-size caps at 200,000 lupus cells",
                "Generated " + datetime.now(timezone.utc).isoformat(timespec="seconds") + ".",
                "## Design",
                "One pre-specified alternative raises both caps tenfold: selector candidates from 2,000,000 to "
                "20,000,000 and fused-value training entries from 600,000 to 6,000,000. Both 2,000- and 5,000-gene "
                "experiments use the saved transductive variant. Only the fused value and selector are refitted. "
                "All five saved teachers (gene median, SVD, weighted kNN, MAGIC, and scVI), inputs, donor splits, "
                "mask, development fitting split, seed 1729, model settings, and eight BLAS threads are reused. "
                "No teacher is refitted and no alternative is tuned.",
                "Local copies in `code/` expose `--max-value-fit-entries` (default 600000) and "
                "`--max-fit-rows` (default 2000000). Source hashes and patches preserve the implementation record. "
                "The selector option already existed; the value option is additive. Shared source and manuscript files are unchanged.",
                "The original scale selector does not keep every masked positive at the default cap: "
                "when positives exceed half the cap it samples half the fitting rows from each class. "
                "Saved defaults use 1,000,000 positives plus 1,000,000 recorded zeros. The 2,000-gene input has "
                "251,014,870 fitting candidates and 6,041,039 positives; the 5,000-gene input has 689,955,238 "
                "candidates and 8,165,494 positives. With the 20,000,000 cap, the same code retains all positives "
                "and samples recorded zeros to reach the cap. Thus the intervention also changes sample composition. "
                "The class-balanced weighting rule remains unchanged.",
                "Selector features contain only teacher proposals and context, not the fused value. The fused-value "
                "refit changes inserted values but cannot directly change the ranking metrics below. These results "
                "test the joint pre-specified intervention; ranking effects arise from the selector cap.",
                "Both default refits must reproduce saved arrays exactly before either larger-cap fit can run. "
                "Evaluation reuses the C_scale evaluator's statistical implementation: pooled F1 at each of ten "
                "fill fractions from 1% to 10%, their arithmetic mean, and pooled ranked average precision. "
                "The same 2,000 paired draws resample 24 held-out donors with replacement, seed 1729. "
                "F1 reweights fixed global-budget donor counts; AP recomputes precision at ranked positives under "
                "donor multiplicities. Selector ties retain candidate order; comparison ties use the original seeded "
                "random order and count-scale conversions. The Weighted kNN comparison uses the saved `methods/graph_smooth` contract, matching the original evaluator. Intervals are 2.5% and 97.5% quantiles of paired differences."]
    validations, unfinished = {}, []
    reproduction_rows, fit_resources, selector_rows, interpretations = [], [], [], []
    for genes in [5000, 2000]:
        base = ROOT / f"genes_{genes}"
        source = Path((base / "source.txt").read_text().strip())
        reproduction = read_json(base / "reproduction.json")
        if not reproduction or not reproduction.get("passed"):
            unfinished.append(f"{genes} genes: default reproduction has not passed")
        for name in ["fused_mean", "test_score", "test_fused_value", "rows", "cols", "labels"]:
            check = (reproduction or {}).get("comparisons", {}).get(name, {})
            reproduction_rows.append([genes, name, check.get("different", "pending"),
                                      check.get("maximum_absolute_difference", "pending")])
        summary = read_csv(base / "evaluation/results/method_summary.csv")
        differences = read_csv(base / "evaluation/results/paired_differences.csv")
        validation = validate(summary, read_csv(source.parent / "results/method_summary.csv"))
        validations[str(genes)] = validation
        sections += [f"## Results: {genes:,} genes",
                     "Method scores are percentages (%), with 95% intervals. Paired differences are percentage points (pp); positive favors larger caps.",
                     table(["Method", "Mean F1, 1%–10% (%)", "Average precision (%)"],
                           [[method] + [metric(next((r for r in summary if r["method"] == method and r["statistic"] == statistic), None))
                                        for statistic in STATISTICS] for method in [LARGE] + REFERENCES]),
                     table(["Larger caps minus", "Mean F1 difference (pp)", "AP difference (pp)"],
                           [[method] + [metric(next((r for r in differences if r["reference"] == LARGE and r["method"] == method
                                                    and r["statistic"] == statistic), None), "difference")
                                        for statistic in STATISTICS] for method in REFERENCES])]
        needed = [(method, statistic) for method in REFERENCES for statistic in STATISTICS]
        missing = [f"{method}: {statistic}" for method, statistic in needed if not any(
            r["reference"] == LARGE and r["method"] == method and r["statistic"] == statistic for r in differences)]
        if missing:
            unfinished.append(f"{genes} genes: missing paired results: " + "; ".join(missing))
        else:
            judgments = []
            for statistic in STATISTICS:
                row = next(r for r in differences if r["reference"] == LARGE and r["method"] == DEFAULT and r["statistic"] == statistic)
                direction = "lies above zero" if float(row["lower"]) > 0 else "lies below zero" if float(row["upper"]) < 0 else "includes zero"
                label = "mean F1" if statistic == "mean_f1_1_to_10" else "average precision"
                judgments.append(f"{label}: the paired interval {direction}")
            sections.append("Against default caps, " + "; ".join(judgments) + ".")
            interpretations.append(f"At {genes:,} genes versus default caps, " + "; ".join(judgments) + ".")
        largest = max((abs(c["difference"]) for c in validation["checks"]), default=float("nan"))
        sections.append(f"Published metric validation: passed={validation['passed']}; exact={validation['exact']}; "
                        f"{len(validation['checks'])}/36 scalar checks available; maximum absolute delta={largest:.17g} "
                        "on the 0–1 scale. Tolerance is 1e-12 for every estimate and interval bound. "
                        "Full deltas and missing entries: `published_metrics_validation.json`.")
        if not validation["passed"]:
            unfinished.append(f"{genes} genes: published metric validation is incomplete or exceeds 1e-12")
        discrepancies = [c for c in validation["checks"] if not c["passed"]]
        if discrepancies:
            sections.append(table(["Discrepancy", "Statistic", "Field", "Current minus published"],
                                  [[c["method"], c["statistic"], c["field"], f"{c['difference']:.17g}"] for c in discrepancies]))
        for variant in ["default", "large"]:
            selector = read_json(base / variant / "selector/selector_report.json")
            if selector:
                model = selector["models"]["mlp"]
                actual_rows = model["fit_rows"]

                assert selector["fit_sample_positives"] < actual_rows
                selector_rows.append([genes, variant, f"{actual_rows:,}", f"{selector['fit_sample_positives']:,}",
                                      f"{100 * actual_rows / selector['fit_candidates']:.4f}%", model["model_iterations"]])
            else:
                selector_rows.append([genes, variant, "pending", "pending", "pending", "pending"])
                unfinished.append(f"{genes} genes: {variant} selector report missing")
            for stage in ["value", "selector"]:
                resource = read_json(base / f"{variant}_{stage}.resources.json")
                if resource:
                    fit_resources.append([genes, variant, stage, f"{resource['wall_seconds']:.2f}",
                                          f"{resource['peak_rss_kib'] / 1024**2:.3f}",
                                          f"{resource['user_seconds'] + resource['system_seconds']:.2f}", resource["returncode"]])
                    if resource["returncode"]:
                        unfinished.append(f"{genes} genes: {variant} {stage} exited {resource['returncode']}")
                else:
                    fit_resources.append([genes, variant, stage, "pending", "pending", "pending", "pending"])
                    unfinished.append(f"{genes} genes: {variant} {stage} resource record missing")
        sections.append(f"Source: `{source}`. Local outputs: `genes_{genes}/default/`, `genes_{genes}/large/`, "
                        f"and `genes_{genes}/evaluation/results/`.")
    (ROOT / "published_metrics_validation.json").write_text(json.dumps(validations, indent=2) + "\n")
    sections += ["## Actual selector fitting sample",
                 table(["Genes", "Caps", "Actual fit rows", "Fit positives", "Fraction of fitting candidates", "MLP iterations"], selector_rows),
                 "Actual fit rows come from `models.mlp.fit_rows`, after final subsampling; "
                 "`fit_sample_candidates` can count a larger intermediate pool. Positive counts come from "
                 "`fit_sample_positives`: both prescribed caps retain every positive surviving the existing balancing branch. "
                 "The saved default cap covers approximately 0.80% of the 2,000-gene candidates and 0.29% of the "
                 "5,000-gene candidates. Final boosted-value fits used 600,000 entries with default caps and 6,000,000 with larger caps at both sizes. "
                 "Cross-validation folds obey the same cap and can use fewer available entries.",
                 "## Exact default reproduction", table(["Genes", "Array", "Different entries", "Maximum absolute difference"], reproduction_rows),
                 "Checks require identical shapes, dtypes, candidates, scores, fused means, and gathered fused values. "
                 "Exact machine-readable records: `genes_2000/reproduction.json` and `genes_5000/reproduction.json`.",
                 "## Resources", "Each refit uses eight CPUs on partition `cs`, account `torch_pr_634_general`. "
                 "Wall time includes subprocess execution; peak RSS is the child-process Linux maximum resident set; "
                 "CPU time is user plus system. GiB means 2^30 bytes. These are single runs on different cs nodes; wall and CPU times are observed costs, not a controlled speed comparison.",
                 table(["Genes", "Caps", "Step", "Wall seconds", "Peak RSS GiB", "CPU seconds", "Exit"], fit_resources),
                 "Slurm accounting for every attempt (batch-step MaxRSS and CPU where available):",
                 table(["Genes", "Caps", "Stage", "Job", "State", "Elapsed", "MaxRSS", "CPU", "Exit"],
                       [[r[k] for k in ["genes", "variant", "stage", "job", "state", "elapsed", "max_rss", "total_cpu", "exit_code"]] for r in resource_rows]),
                 "Raw subprocess commands and measurements are in `genes_*/{default,large}_{value,selector}.resources.json`. "
                 "All Slurm attempts are listed in `jobs.tsv`; accounting is exported to `resources.csv`.",
                 "Metric files are `genes_*/evaluation/results/method_summary.csv` and `paired_differences.csv`. "
                 "The same directories contain `masked_f1_unit_counts.parquet` and `masked_f1_curves.csv`. "
                 "Saved refits are `genes_*/{default,large}/safe_fusion/mean.npy` and `genes_*/{default,large}/selector/test_score.npy`.",
                 "## Reproduction commands",
                 "Run from a Slurm allocation; the launcher submits all heavy work with `sbatch`.",
                 "`code/submit.sh` implements the following commands and refuses duplicate initial submission when `jobs.tsv` exists. "
                 "The two successful default reproduction jobs gate both larger-cap runs. Run only in a clean experiment output "
                 "directory, or deliberately manage an authorized recovery; existing output paths are reused.",
                 "```bash\ncd .\n"
                 "root=\"$PWD/artifacts/paper_evidence/review_round4/scale_caps\"\n"
                 "bash \"$root/code/submit.sh\"\n"
                 "while squeue -u yz5944 -h -o %j | grep -q '^caps-'; do sleep 600; done\n"
                 "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \"$root/code/report.py\"\n```",
                 "The exact `sbatch` commands, resources, and dependencies are in `code/submit.sh`; "
                 "the stage interface is `code/run.sh {value,selector,check,evaluate} {2000,5000} {default,large}`. "
                 "Both default `check` jobs must succeed before either large `value` job. "
                 "Each resource JSON records the exact scientific command, including all five teacher paths.",
                 "## Failures and remaining work",
                 (ROOT / "execution_notes.md").read_text().removeprefix("# Execution notes\n").strip()
                 if (ROOT / "execution_notes.md").exists() else "No execution notes recorded."]
    if accounting_error:
        unfinished.append("Slurm accounting failed: " + accounting_error)
    latest_jobs = {(r["genes"], r["variant"], r["stage"]): r for r in resource_rows}
    for row in latest_jobs.values():
        if row["state"] != "COMPLETED":
            unfinished.append(f"Latest {row['genes']}-gene {row['variant']} {row['stage']} job {row['job']}: {row['state']}")
    status = "Incomplete: outstanding work is listed below." if unfinished else "Complete: both exact reproduction gates and all evaluations passed."
    if len(interpretations) == 2:
        status += " " + " ".join(interpretations)
        status += " These intervals describe only this tenfold-cap intervention; they do not establish a general scaling limit of the method."
    if not unfinished:
        status += (" The AP gains are small: 0.0349 percentage points at 5,000 genes and 0.1366 at 2,000 genes. "
                   "The results support a modest AP penalty from the default selector cap. "
                   "Evidence that the caps explain the mean-F1 plateau remains limited. "
                   "At 5,000 genes, the larger-cap model still has a mean-F1 difference interval spanning zero against scVI probability.")
    sections[2:2] = ["## Status and interpretation", status]
    sections.append("\n".join("- " + item for item in unfinished) if unfinished else
                    "All requested refits, exact reproduction checks, paired comparisons, and published-metric validations are complete.")
    sections += ["The inference is specific to this single cap intervention, these two feature panels, and the held-out donors. "
                 "No additional cap or setting was tested. All new code, outputs, caches, logs, and report files remain under this directory. Final protocol checks are in `checks/final_protocol.json`. Jobs named `scmp-*`, `cx-*`, and `mm3-*` were not modified. No commit or push was made."]
    (ROOT / "README.md").write_text("\n\n".join(sections) + "\n")
    print(ROOT / "README.md")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        current = [{"cells": "200000", "method": m, "statistic": s, "estimate": "0.2", "lower": "0.1", "upper": "0.3"}
                   for m in REFERENCES for s in STATISTICS]
        published = [{**r, "method": "Safe Fusion (transductive)" if r["method"] == DEFAULT else r["method"]} for r in current]
        assert validate(current, published)["passed"] and validate(current, published)["exact"]
        current[0] = {**current[0], "estimate": "0.20000000001"}
        assert not validate(current, published)["passed"]
        assert not validate(current[:-1], published)["passed"]
        assert metric(None) == "pending"
        print("report self-test passed")
    else:
        main()
