#!/usr/bin/env python3
"""Summarize wall time and peak memory of pipeline steps from Slurm accounting.

Reads a tab-separated job list (default ``scripts/runtime_jobs.tsv``) with
columns ``step``, ``dataset`` and ``jobs``, where ``jobs`` is a comma-separated
list of the Slurm job or array-task IDs that ran the step for the dataset.
The wall time of a step for a dataset is the summed Elapsed time of its tasks,
and its peak memory is the largest MaxRSS over the job steps of its tasks. For
each step, the table reports the range of wall times over datasets and the
largest peak memory. Every listed task must have state COMPLETED.

Accounting records come from ``sacct`` or, with ``--sacct-file``, from a saved
pipe-delimited sacct dump with a header line, such as the
``sacct_records.txt`` that this script writes next to the tables. MaxRSS
suffixes K, M, G and T are binary multiples (1K = 1024 bytes), and a number
without a suffix is in bytes (``sacct --noconvert``). Memory is reported in GB
of 10^9 bytes. Minutes and GB are rounded half up to one decimal.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

FIELDS = ("JobID", "JobName", "State", "Elapsed", "MaxRSS", "AllocCPUS", "AllocTRES", "Start")
REQUIRED = ("JobID", "State", "Elapsed", "MaxRSS", "AllocCPUS")
RSS_MULTIPLIER = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
GPU = re.compile(r"gres/gpu(?::[^=,]+)?=(\d+)")
BYTES_PER_GB = Decimal(10**9)
ONE_DECIMAL = Decimal("0.1")


def read_job_list(path: Path) -> list[dict]:
    lines = [line for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
    rows = list(csv.DictReader(lines, delimiter="\t"))
    if not rows or set(rows[0]) != {"step", "dataset", "jobs"}:
        raise ValueError(f"{path} needs the tab-separated columns step, dataset, jobs")
    for row in rows:
        row["jobs"] = [job.strip() for job in row["jobs"].split(",") if job.strip()]
    return rows


def query_sacct(job_ids: list[str]) -> str:
    command = ["sacct", "-j", ",".join(job_ids), "-P", f"--format={','.join(FIELDS)}"]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def parse_records(text: str) -> tuple[list[str], list[dict]]:
    lines = [line for line in text.splitlines() if line.strip()]
    header = lines[0].split("|")
    missing = [field for field in REQUIRED if field not in header]
    if missing:
        raise ValueError(f"sacct records lack the columns {missing}")
    return header, [dict(zip(header, line.split("|"))) for line in lines[1:]]


def elapsed_seconds(text: str) -> Decimal:
    days, _, clock = text.rpartition("-")
    parts = [Decimal(part) for part in clock.split(":")]
    parts = [Decimal(0)] * (3 - len(parts)) + parts
    hours, minutes, seconds = parts
    return Decimal(int(days or 0)) * 86400 + hours * 3600 + minutes * 60 + seconds


def rss_bytes(text: str) -> Decimal | None:
    text = text.strip()
    if not text:
        return None
    suffix = text[-1].upper() if text[-1].isalpha() else ""
    return Decimal(text[: len(text) - len(suffix)]) * RSS_MULTIPLIER[suffix]


def one_decimal(value: Decimal) -> str:
    return str(value.quantize(ONE_DECIMAL, rounding=ROUND_HALF_UP))


def value_range(low: Decimal, high: Decimal) -> str:
    low_text, high_text = one_decimal(low), one_decimal(high)
    return low_text if low_text == high_text else f"{low_text} to {high_text}"


def summarize(job_rows: list[dict], records: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    allocations: dict[str, dict] = {}
    job_steps: dict[str, list[dict]] = {}
    for record in records:
        task, _, job_step = record["JobID"].partition(".")
        if job_step:
            job_steps.setdefault(task, []).append(record)
        elif task in allocations:
            raise ValueError(f"duplicate sacct records for {task}")
        else:
            allocations[task] = record

    listed = [job for row in job_rows for job in row["jobs"]]
    absent = [job for job in listed if job not in allocations]
    if absent:
        raise ValueError(f"no sacct record for {absent}")
    incomplete = {job: allocations[job]["State"] for job in listed if allocations[job]["State"] != "COMPLETED"}
    if incomplete:
        raise ValueError(f"listed tasks did not complete: {incomplete}")

    used = []
    by_dataset = []
    for row in job_rows:
        tasks = [allocations[job] for job in row["jobs"]]
        task_steps = [record for job in row["jobs"] for record in job_steps.get(job, [])]
        used.extend(tasks + task_steps)
        rss = [value for value in map(rss_bytes, (r["MaxRSS"] for r in tasks + task_steps)) if value is not None]
        if not rss:
            raise ValueError(f"no MaxRSS recorded for {row['step']} on {row['dataset']}")
        wall = sum(elapsed_seconds(task["Elapsed"]) for task in tasks)
        gpus = {int(match.group(1)) for task in tasks for match in GPU.finditer(task.get("AllocTRES", ""))}
        by_dataset.append({
            "step": row["step"],
            "dataset": row["dataset"],
            "jobs": ",".join(row["jobs"]),
            "wall_time_s": wall,
            "wall_time_min": wall / 60,
            "peak_rss_bytes": max(rss),
            "peak_memory_gb": max(rss) / BYTES_PER_GB,
            "alloc_cpus": ";".join(sorted({task["AllocCPUS"] for task in tasks})),
            "alloc_gpus": ";".join(str(n) for n in sorted(gpus)) or "0",
        })

    table = []
    for step in dict.fromkeys(row["step"] for row in by_dataset):
        rows = [row for row in by_dataset if row["step"] == step]
        low = min(row["wall_time_min"] for row in rows)
        high = max(row["wall_time_min"] for row in rows)
        peak = max(row["peak_memory_gb"] for row in rows)
        table.append({
            "step": step,
            "n_datasets": len(rows),
            "wall_time_min_low": low,
            "wall_time_min_high": high,
            "peak_memory_gb": peak,
            "wall_time_min_text": value_range(low, high),
            "peak_memory_gb_text": one_decimal(peak),
            "alloc_cpus": ";".join(sorted({c for row in rows for c in row["alloc_cpus"].split(";")})),
            "alloc_gpus": ";".join(sorted({g for row in rows for g in row["alloc_gpus"].split(";")})),
        })
    return table, by_dataset, used


def csv_cell(value):
    if isinstance(value, Decimal) and value != value.to_integral_value():
        return f"{value:.4f}"
    return value


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_cell(value) for key, value in row.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", default="scripts/runtime_jobs.tsv",
                        help="tab-separated step, dataset, jobs list (default: %(default)s)")
    parser.add_argument("--sacct-file", help="saved pipe-delimited sacct dump with header; queries sacct when omitted")
    parser.add_argument("--output-dir", default="artifacts/paper_evidence/runtime",
                        help="directory for runtime_table.csv, runtime_by_dataset.csv and sacct_records.txt")
    args = parser.parse_args()

    job_rows = read_job_list(Path(args.jobs))
    if args.sacct_file:
        text = Path(args.sacct_file).read_text()
    else:
        text = query_sacct(list(dict.fromkeys(job for row in job_rows for job in row["jobs"])))
    header, records = parse_records(text)
    table, by_dataset, used = summarize(job_rows, records)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "runtime_table.csv", table)
    write_csv(output / "runtime_by_dataset.csv", by_dataset)
    (output / "sacct_records.txt").write_text(
        "\n".join(["|".join(header)] + ["|".join(record.get(field, "") for field in header) for record in used]) + "\n")

    width = max(len(row["step"]) for row in table)
    print(f"{'Step':<{width}}  {'Wall time (min)':>15}  {'Peak memory (GB)':>16}  CPUs  GPUs  Datasets")
    for row in table:
        print(f"{row['step']:<{width}}  {row['wall_time_min_text']:>15}  {row['peak_memory_gb_text']:>16}  "
              f"{row['alloc_cpus']:>4}  {row['alloc_gpus']:>4}  {row['n_datasets']:>8}")
    print(f"-> {output / 'runtime_table.csv'}")


if __name__ == "__main__":
    main()
