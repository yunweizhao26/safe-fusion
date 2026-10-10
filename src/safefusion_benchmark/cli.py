from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .config import config_hash, git_commit, load_config, run_id
from .contracts import validate_output_contract
from .corruption import corrupt_counts
from .evaluation import evaluate_core, evaluate_downstream
from .hashing import sha256_file
from .io import counts_layer, dense, make_synthetic_counts, read_parquet, write_json, write_parquet
from .methods import write_builtin_contract
from .provenance import execute_recorded
from .publication import evaluate_publication_gate
from .schema import validate_dataset, validate_perturbation_obs
from .perturbation import audit_eccite_library
from .splits import split_biological_units


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--environment", default="workflow/envs/core.yaml")


def _record(args: argparse.Namespace, rule: str, inputs: list[str], outputs: list[str], action, method=None, suppress_failure=False):
    config = load_config(args.config)
    return execute_recorded(
        rule,
        args.provenance,
        [args.config, *inputs],
        outputs,
        action,
        seed=int(config["study"]["seed"]),
        environment=args.environment,
        method=method,
        suppress_failure=suppress_failure,
    )


def cmd_run_id(args: argparse.Namespace) -> None:
    print(run_id(load_config(args.config)))


def cmd_manifest(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    payload = {
        "schema_version": 1,
        "run_id": run_id(config),
        "git_commit": git_commit(),
        "config_sha256": config_hash(config),
        "config": config,
        "locked_test_policy": "test biological units are never used for fitting or selection",
    }
    _record(args, "run_manifest", [], [args.output], lambda: write_json(args.output, payload))


def cmd_fetch(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    spec = config["datasets"][args.dataset]

    def action() -> None:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if spec["source"] == "synthetic":
            adata = make_synthetic_counts(int(spec["n_cells"]), int(spec["n_genes"]), int(spec["n_units"]), int(config["study"]["seed"]))
            adata.write_h5ad(destination)
        else:
            source = Path(spec["source"])
            if spec.get("expected_sha256") in {None, "", "REQUIRED"}:
                raise ValueError(f"{args.dataset}: expected_sha256 must be locked before production use")
            if not source.exists():
                raise FileNotFoundError(f"Missing source for {args.dataset}: {source}")
            observed = sha256_file(source)
            if observed != spec["expected_sha256"]:
                raise ValueError(f"{args.dataset}: source checksum mismatch ({observed})")
            shutil.copyfile(source, destination)

    inputs = [] if spec["source"] == "synthetic" else [spec["source"]]
    _record(args, "fetch", inputs, [args.output], action)


def cmd_validate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    spec = config["datasets"][args.dataset]

    def action() -> None:
        adata = ad.read_h5ad(args.input)
        report = validate_dataset(adata, spec["biological_unit"])
        if "perturbation" in spec["role"]:
            validate_perturbation_obs(adata.obs)
        if spec["accession"] == "GSE153056":
            report["guide_library"] = audit_eccite_library(adata.obs, int(spec["expected_guides"]), int(spec["expected_targets"]), int(spec["expected_nt_guides"]))
        report.update({"dataset": args.dataset, "accession": spec["accession"], "input_sha256": sha256_file(args.input)})
        write_json(args.output, report)

    _record(args, "validate", [args.input], [args.output], action)


def cmd_preprocess(args: argparse.Namespace) -> None:
    def action() -> None:
        adata = ad.read_h5ad(args.input)
        raw = counts_layer(adata).astype(np.int32)
        adata.X = raw.copy()
        adata.layers["counts"] = raw.copy()
        library = np.asarray(raw.sum(axis=1)).ravel()
        scale = np.divide(1e4, library, out=np.zeros_like(library, dtype=float), where=library > 0)
        normalized = sparse.diags(scale) @ raw
        normalized.data = np.log1p(normalized.data)
        adata.layers["log1p_cpm"] = normalized.astype(np.float32)
        adata.uns["preprocessing"] = {"raw_unchanged": True, "normalized_layer": "log1p_cpm", "normalization_target": 10000}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(args.output)

    _record(args, "preprocess", [args.input], [args.output], action)


def cmd_split(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    dataset = config["datasets"][args.dataset]
    split = config["split"]

    def action() -> None:
        adata = ad.read_h5ad(args.input, backed="r")
        frame = split_biological_units(adata.obs, dataset["biological_unit"], float(split["development_fraction"]), float(split["validation_fraction"]), int(config["study"]["seed"]))
        write_parquet(args.output, frame)

    _record(args, "split", [args.input], [args.output], action)


def cmd_corrupt(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    dataset = config["datasets"][args.dataset]
    spec = config["corruptions"][args.corruption]

    def action() -> None:
        adata = ad.read_h5ad(args.input)
        splits = read_parquet(args.splits).set_index("cell_id").loc[adata.obs_names.astype(str)]
        counts = dense(counts_layer(adata)).astype(np.int32)
        corrupted, coordinates = corrupt_counts(
            counts,
            adata.obs_names.astype(str).to_numpy(),
            adata.var_names.astype(str).to_numpy(),
            adata.obs[dataset["biological_unit"]].astype(str).to_numpy(),
            splits["split"].to_numpy(),
            spec,
            int(config["study"]["seed"]),
        )
        output = adata.copy()
        output.X = sparse.csr_matrix(corrupted)
        for layer in list(output.layers):
            del output.layers[layer]
        output.layers["corrupted_counts"] = sparse.csr_matrix(corrupted)
        output.uns["corruption"] = {"name": args.corruption, "spec": spec, "coordinates_sha256_pending": True}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        output.write_h5ad(args.output)
        write_parquet(args.coordinates, coordinates)

    _record(args, "corrupt", [args.input, args.splits], [args.output, args.coordinates], action)


def cmd_run_method(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    method_spec = config["methods"][args.method]

    def action() -> None:
        try:
            if method_spec["runner"] == "builtin":
                adata = ad.read_h5ad(args.input)
                if method_spec.get("smoke_only") and "generated" not in adata.uns:
                    raise RuntimeError(f"{args.method} builtin is a contract smoke implementation and is forbidden on production data; configure the frozen external runner")
                counts = dense(adata.layers["corrupted_counts"] if "corrupted_counts" in adata.layers else adata.X).astype(np.int32)
                splits = read_parquet(args.splits).set_index("cell_id").loc[adata.obs_names.astype(str)]
                training_mask = splits["split"].isin(["development", "validation"]).to_numpy()
                write_builtin_contract(
                    args.method,
                    counts,
                    adata.obs_names.astype(str).tolist(),
                    adata.var_names.astype(str).tolist(),
                    args.output,
                    args.input,
                    args.coordinates,
                    int(config["study"]["seed"]),
                    training_mask,
                )
            else:
                command = method_spec.get("command")
                if not isinstance(command, list) or not command:
                    raise RuntimeError(f"External runner for {args.method} requires a command list in the site config")
                substitutions = {"input": args.input, "coordinates": args.coordinates, "splits": args.splits, "output": args.output, "seed": str(config["study"]["seed"])}
                resolved = [str(token).format(**substitutions) for token in command]
                subprocess.run(resolved, check=True)
        except Exception as exc:
            write_json(Path(args.output) / "failure.json", {"method": args.method, "type": type(exc).__name__, "reason": str(exc)})
            raise

    _record(args, "run_method", [args.input, args.coordinates, args.splits], [args.output], action, method={"name": args.method, **method_spec}, suppress_failure=True)


def cmd_standardize(args: argparse.Namespace) -> None:
    def action() -> None:
        failure_path = Path(args.input) / "failure.json"
        if failure_path.exists():
            destination = Path(args.output)
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(failure_path, destination / "failure.json")
            write_json(destination / "contract_validation.json", {"valid": False, "failure": json.loads(failure_path.read_text())})
            return
        adata = ad.read_h5ad(args.input_data, backed="r")
        report = validate_output_contract(args.input, adata.obs_names.astype(str).tolist(), adata.var_names.astype(str).tolist())
        source, destination = Path(args.input), Path(args.output)
        destination.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            if item.is_file():
                shutil.copyfile(item, destination / item.name)
        write_json(destination / "contract_validation.json", report)

    _record(args, "standardize", [args.input, args.input_data], [args.output], action)


def cmd_core_eval(args: argparse.Namespace) -> None:
    config = load_config(args.config)

    def action() -> None:
        failure_path = Path(args.contract) / "failure.json"
        if failure_path.exists():
            failure = json.loads(failure_path.read_text())
            write_parquet(args.output, pd.DataFrame([{"dataset": args.dataset, "corruption": args.corruption, "method": args.method, "scope": "overall", "biological_unit": "all", "metric": "method_failure", "value": np.nan, "fill_budget": np.nan, "status": failure["reason"]}]))
            return
        coordinate_columns = [
            "cell_id", "gene_id", "cell_index", "gene_index", "original_value",
            "corrupted_value", "biological_unit", "split",
        ]
        locked_coordinates = pd.read_parquet(
            args.coordinates,
            columns=coordinate_columns,
        )
        frame = evaluate_core(ad.read_h5ad(args.truth), locked_coordinates, args.contract, ad.read_h5ad(args.corrupted), args.method, args.dataset, args.corruption, list(config["study"]["fill_budgets"]))
        write_parquet(args.output, frame)

    _record(args, "core_evaluation", [args.truth, args.corrupted, args.coordinates, args.contract], [args.output], action)


def cmd_downstream(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    def action() -> None:
        failure_path = Path(args.contract) / "failure.json"
        if failure_path.exists():
            failure = json.loads(failure_path.read_text())
            write_parquet(args.output, pd.DataFrame([{"dataset": args.dataset, "corruption": args.corruption, "method": args.method, "task": "method_failure", "metric": "method_failure", "value": np.nan, "status": failure["reason"]}]))
            return
        frame = evaluate_downstream(ad.read_h5ad(args.truth), read_parquet(args.splits), args.contract, args.method, args.dataset, args.corruption, config["datasets"][args.dataset]["biological_unit"], int(config["study"]["seed"]))
        write_parquet(args.output, frame)

    _record(args, "downstream_evaluation", [args.truth, args.splits, args.contract], [args.output], action)


def cmd_perturb(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    role = config["datasets"][args.dataset]["role"]

    def action() -> None:
        failure_path = Path(args.contract) / "failure.json"
        if failure_path.exists():
            status = json.loads(failure_path.read_text())["reason"]
        else:
            status = "not_applicable" if "perturbation" not in role else "requires_dataset_specific_confirmatory_runner"
        write_parquet(args.output, pd.DataFrame([{"dataset": args.dataset, "corruption": args.corruption, "method": args.method, "task": "perturbation_preservation", "status": status, "value": np.nan}]))

    _record(args, "perturbation_evaluation", [args.contract], [args.output], action)


def cmd_aggregate(args: argparse.Namespace) -> None:
    config = load_config(args.config)

    def action() -> None:
        core_frames = [read_parquet(path).assign(analysis_family="core") for path in args.core]
        downstream_frames = [read_parquet(path).assign(analysis_family="downstream") for path in args.downstream]
        perturb_frames = [read_parquet(path).assign(analysis_family="perturbation") for path in args.perturb]
        consolidated = pd.concat([*core_frames, *downstream_frames, *perturb_frames], ignore_index=True, sort=False)
        write_parquet(args.output, consolidated)
        failures = []
        for provenance_path in args.method_provenance:
            payload = json.loads(Path(provenance_path).read_text(encoding="utf-8"))
            if payload.get("failure"):
                failures.append({"provenance": provenance_path, **payload["failure"], "runtime_seconds": payload.get("runtime_seconds"), "peak_cpu_memory_mb": payload.get("peak_cpu_memory_mb"), "peak_gpu_memory_mb": payload.get("peak_gpu_memory_mb")})
        write_parquet(args.failures, pd.DataFrame(failures, columns=["provenance", "type", "message", "runtime_seconds", "peak_cpu_memory_mb", "peak_gpu_memory_mb"]))
        gate = evaluate_publication_gate(consolidated, config)
        gate.update({
            "required_relative_improvement": config["study"]["minimum_relative_improvement"],
            "independent_real_datasets_required": 2,
            "consolidated_metrics_sha256": sha256_file(args.output),
        })
        write_json(args.gate, gate)

    _record(args, "aggregate", [*args.core, *args.downstream, *args.perturb, *args.method_provenance], [args.output, args.failures, args.gate], action)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="safefusion-benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)
    runid = subparsers.add_parser("run-id"); runid.add_argument("--config", required=True); runid.set_defaults(func=cmd_run_id)
    manifest = subparsers.add_parser("manifest"); _common(manifest); manifest.add_argument("--output", required=True); manifest.set_defaults(func=cmd_manifest)
    fetch = subparsers.add_parser("fetch"); _common(fetch); fetch.add_argument("--dataset", required=True); fetch.add_argument("--output", required=True); fetch.set_defaults(func=cmd_fetch)
    validate = subparsers.add_parser("validate"); _common(validate); validate.add_argument("--dataset", required=True); validate.add_argument("--input", required=True); validate.add_argument("--output", required=True); validate.set_defaults(func=cmd_validate)
    preprocess = subparsers.add_parser("preprocess"); _common(preprocess); preprocess.add_argument("--input", required=True); preprocess.add_argument("--output", required=True); preprocess.set_defaults(func=cmd_preprocess)
    split = subparsers.add_parser("split"); _common(split); split.add_argument("--dataset", required=True); split.add_argument("--input", required=True); split.add_argument("--output", required=True); split.set_defaults(func=cmd_split)
    corrupt = subparsers.add_parser("corrupt"); _common(corrupt); corrupt.add_argument("--dataset", required=True); corrupt.add_argument("--corruption", required=True); corrupt.add_argument("--input", required=True); corrupt.add_argument("--splits", required=True); corrupt.add_argument("--output", required=True); corrupt.add_argument("--coordinates", required=True); corrupt.set_defaults(func=cmd_corrupt)
    run = subparsers.add_parser("run-method"); _common(run); run.add_argument("--method", required=True); run.add_argument("--input", required=True); run.add_argument("--coordinates", required=True); run.add_argument("--splits", required=True); run.add_argument("--output", required=True); run.set_defaults(func=cmd_run_method)
    standard = subparsers.add_parser("standardize"); _common(standard); standard.add_argument("--input", required=True); standard.add_argument("--input-data", required=True); standard.add_argument("--output", required=True); standard.set_defaults(func=cmd_standardize)
    core = subparsers.add_parser("core-eval"); _common(core); core.add_argument("--dataset", required=True); core.add_argument("--corruption", required=True); core.add_argument("--method", required=True); core.add_argument("--truth", required=True); core.add_argument("--corrupted", required=True); core.add_argument("--coordinates", required=True); core.add_argument("--contract", required=True); core.add_argument("--output", required=True); core.set_defaults(func=cmd_core_eval)
    downstream = subparsers.add_parser("downstream"); _common(downstream); downstream.add_argument("--dataset", required=True); downstream.add_argument("--corruption", required=True); downstream.add_argument("--method", required=True); downstream.add_argument("--truth", required=True); downstream.add_argument("--splits", required=True); downstream.add_argument("--contract", required=True); downstream.add_argument("--output", required=True); downstream.set_defaults(func=cmd_downstream)
    perturb = subparsers.add_parser("perturb"); _common(perturb); perturb.add_argument("--dataset", required=True); perturb.add_argument("--corruption", required=True); perturb.add_argument("--method", required=True); perturb.add_argument("--contract", required=True); perturb.add_argument("--output", required=True); perturb.set_defaults(func=cmd_perturb)
    aggregate = subparsers.add_parser("aggregate"); _common(aggregate); aggregate.add_argument("--core", nargs="*", default=[]); aggregate.add_argument("--downstream", nargs="*", default=[]); aggregate.add_argument("--perturb", nargs="*", default=[]); aggregate.add_argument("--method-provenance", nargs="*", default=[]); aggregate.add_argument("--output", required=True); aggregate.add_argument("--failures", required=True); aggregate.add_argument("--gate", required=True); aggregate.set_defaults(func=cmd_aggregate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
