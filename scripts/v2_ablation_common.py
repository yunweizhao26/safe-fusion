#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "scripts"))

from masked_f1_units import EVIDENCE, Unit, load_units_manifest

MANIFESTS = {
    EVIDENCE / "review_round2" / "leakage_free" / "units_manifest.json": EVIDENCE / "review_round2" / "fusion_value" / "selectors",
    EVIDENCE / "review_round3" / "colon_crossfit" / "units_manifest.json": EVIDENCE / "review_round3" / "colon_crossfit" / "fusion_value" / "selectors",
}
ABLATION_ROOT = EVIDENCE / "review_round3" / "value_v2_ablations" / "ablations"

UNIT_KEYS = ("pancreas_0", "pancreas_1", "pancreas_2", "colon", "norman_crispra", "colon_0", "colon_1", "colon_2")

DATASET_NAMES = {"colon": "Colon, locked split"}

TEACHERS = ("gene_median", "svd_impute", "graph_smooth", "magic_inductive", "scvi_inductive")
PRODUCTION_HIDDEN = (128, 64, 32)
HIDDEN_SIZES = ((64, 32), PRODUCTION_HIDDEN, (256, 128, 64))
PRODUCTION_BOOSTING = (63, 100)
BOOSTING_GRID = tuple((leaves, minimum) for leaves in (31, 63) for minimum in (50, 100, 200))
MASK_RATES = (0.05, 0.20)
PRODUCTION_MASK_RATE = 0.10

def rate_name(rate: float) -> str:
    return f"rho_{rate:.2f}".replace(".", "p")

def hidden_name(sizes: tuple[int, ...]) -> str:
    return "h" + "_".join(str(size) for size in sizes)

def boosting_name(leaves: int, minimum: int) -> str:
    return f"leaves{leaves}_minleaf{minimum}"

@lru_cache(maxsize=None)
def manifest_units() -> dict[str, tuple[Unit, Path]]:

    result = {}
    for manifest, selectors in MANIFESTS.items():
        for unit in load_units_manifest(manifest):
            result[unit.key] = (replace(unit, dataset=DATASET_NAMES.get(unit.key, unit.dataset)), selectors)
    return result

def units() -> dict[str, Unit]:
    return {key: manifest_units()[key][0] for key in UNIT_KEYS}

def production_teacher_root(unit: Unit) -> Path:
    return Path(unit.contracts["SVD"]).parent

def mask_rate_root(root: Path, rate: float, key: str) -> Path:
    return root / "mask_rate" / rate_name(rate) / key

def mask_rate_unit(unit: Unit, root: Path, rate: float) -> Unit:

    base = mask_rate_root(root, rate, unit.key) / "input"
    return replace(unit, corrupted=base / "corrupted.h5ad", coordinates=base / "coordinates.parquet")

@dataclass(frozen=True)
class Variant:
    name: str
    family: str
    scores: Path
    value: Path
    refits_value: bool
    refits_selector: bool

def ablation_variants(unit: Unit, root: Path = ABLATION_ROOT) -> list[Variant]:
    production_scores = manifest_units()[unit.key][1] / unit.key / "safe_fusion" / "test_scores.npy"
    production_value = production_teacher_root(unit) / "safe_fusion"
    result = [Variant("Production", "production", production_scores, production_value, False, False)]
    for rate in MASK_RATES:
        base = mask_rate_root(root, rate, unit.key)
        result.append(Variant(f"Mask rate {rate:.2f}", "mask_rate", base / "selector" / "test_scores.npy",
                              base / "methods" / "safe_fusion", True, True))
    base = root / "no_crossfit" / unit.key
    result.append(Variant("No cross-fitting", "cross_fitting", base / "selector" / "test_scores.npy",
                          base / "methods" / "safe_fusion", True, True))
    for sizes in HIDDEN_SIZES:
        result.append(Variant(f"Selector {'-'.join(map(str, sizes))}", "selector_size",
                              root / "selector_size" / hidden_name(sizes) / unit.key / "test_scores.npy",
                              production_value, False, True))
    for leaves, minimum in BOOSTING_GRID:
        result.append(Variant(f"Value {leaves} leaves, minimum leaf {minimum}", "value_boosting", production_scores,
                              root / "value_boosting" / boosting_name(leaves, minimum) / unit.key, True, False))
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unit", choices=UNIT_KEYS, required=True)
    parser.add_argument("--unit-paths", action="store_true", required=True)
    args = parser.parse_args()
    unit = units()[args.unit]
    print(" ".join(str(value) for value in (
        unit.corrupted, unit.coordinates, unit.splits, unit.truth, production_teacher_root(unit),
        unit.fit_split, unit.unit_column,
    )))

if __name__ == "__main__":
    main()
