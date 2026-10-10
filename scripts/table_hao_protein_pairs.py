import argparse
from pathlib import Path

import pandas as pd

METHODS = ("safe_fusion", "magic", "scvi")


def fmt(value):
    value = round(value, 2) + 0.0
    text = f"{abs(value):.2f}"
    return f"$-${text}" if value < 0 else text


def block(values):
    rounded = [round(v, 2) for v in values]
    best = max(rounded)
    return [rf"\textbf{{{fmt(v)}}}" if round(v, 2) == best else fmt(v) for v in values]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=Path, default=Path("artifacts/paper_evidence/review_round4/protein_depth/tables"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/paper_evidence/review_round4/protein_depth/tables/hao_pairs_table.tex"))
    args = parser.parse_args()
    assoc = pd.read_csv(args.tables / "association.csv")
    diag = pd.read_csv(args.tables / "diagnostics.csv")
    assoc = assoc[assoc["case"].str.startswith("hao_")]
    diag = diag[(diag["case"].str.startswith("hao_")) & (diag["metric"] == "protein_depth_spearman")]
    lines = []
    for case, rows in assoc.groupby("case", sort=True):
        first = rows.iloc[0]
        get = lambda method, metric: float(rows[(rows["method"] == method) & (rows["metric"] == metric)]["estimate"].iloc[0])
        pooled = block([get(m, "pooled_spearman") for m in METHODS])
        adjusted = block([get(m, "partial_depth_spearman") for m in METHODS])
        depth = float(diag[diag["case"] == case]["estimate"].iloc[0])
        name = f"{first['gene']} / {first['protein']}"
        lines.append(" & ".join([name, f"{int(first['n_train_detected']):,}", fmt(depth), *pooled, *adjusted]) + r" \\")
    args.output.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} rows written to {args.output}")


if __name__ == "__main__":
    main()
