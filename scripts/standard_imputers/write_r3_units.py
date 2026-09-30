#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "scripts"))

from masked_f1_units import EVIDENCE, ROOT, load_units_manifest, unit_manifest_entry
from write_colon_crossfit_units import colon_crossfit_units

NEW_METHODS = {"DCA": "dca", "DCA (ZINB)": "dca_zinb", "scImpute": "scimpute", "scRecover": "screcover", "EnImpute": "enimpute", "scVI (ZINB)": "scvi_zinb"}

FIT_NAMES = {"pancreas_0": "pancreas_0", "pancreas_1": "pancreas_1", "pancreas_2": "pancreas_2", "colon": "colon",
             "norman_crispra": "norman_crispra", "colon_0": "colon_cf_0", "colon_1": "colon_cf_1", "colon_2": "colon_cf_2"}

def with_new_contracts(entries: list[dict], root: Path) -> list[dict]:
    for entry in entries:
        for name, directory in NEW_METHODS.items():
            entry["contracts"][name] = str((root / "fits" / directory / FIT_NAMES[entry["key"]]).relative_to(ROOT))
    return entries

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=EVIDENCE / "review_round3" / "comparators")
    parser.add_argument("--table1-manifest", type=Path, default=EVIDENCE / "review_round2" / "leakage_free" / "units_manifest.json")
    parser.add_argument("--colon-crossfit-root", type=Path, default=EVIDENCE / "review_round3" / "colon_crossfit")
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    root = args.root if args.root.is_absolute() else ROOT / args.root
    table1 = [unit_manifest_entry(unit) for unit in load_units_manifest(args.table1_manifest)]
    crossfit_root = args.colon_crossfit_root if args.colon_crossfit_root.is_absolute() else ROOT / args.colon_crossfit_root
    crossfit = [unit_manifest_entry(unit) for unit in colon_crossfit_units(crossfit_root, 3, args.seed)]
    root.mkdir(parents=True, exist_ok=True)
    for name, entries in (("units_manifest.json", table1), ("colon_crossfit_units.json", crossfit)):
        (root / name).write_text(json.dumps(with_new_contracts(entries, root), indent=2) + "\n")
        print(f"wrote {len(entries)} units to {root / name}")

if __name__ == "__main__":
    main()
