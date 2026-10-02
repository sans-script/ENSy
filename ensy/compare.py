"""Compare two result files (e.g. local vs GitHub Actions) -- reproducibility check T4.

Run:  python -m ensy.compare results/main_X.json other/main_X.json [--tol 1e-3] [--markdown]

Configurations are matched by (model, algorithm); for every metric the maximum
absolute difference over K is reported. Exit status 1 when any difference
exceeds the tolerance and --strict is given.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = ("trustworthiness", "continuity", "Q_NX", "B_NX")


def _index(result: dict) -> dict:
    return {(c["model"], c["algorithm"]): c for c in result["configs"]}


def compare(a: dict, b: dict) -> list[dict]:
    if a["dataset_hash"] != b["dataset_hash"]:
        raise SystemExit(f"different datasets: {a['dataset_hash'][:12]} vs {b['dataset_hash'][:12]}")
    ia, ib = _index(a), _index(b)
    rows = []
    for key in sorted(set(ia) & set(ib)):
        ca, cb = ia[key], ib[key]
        row = {"model": key[0], "algorithm": key[1]}
        for m in METRICS:
            row[m] = max(abs(x[m] - y[m]) for x, y in zip(ca["curves"], cb["curves"]))
        row["silhouette"] = abs(ca["silhouette_projected"] - cb["silhouette_projected"])
        rows.append(row)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args(argv)
    a = json.loads(args.a.read_text(encoding="utf-8"))
    b = json.loads(args.b.read_text(encoding="utf-8"))
    rows = compare(a, b)
    cols = (*METRICS, "silhouette")
    worst = max((r[c] for r in rows for c in cols), default=0.0)
    env = lambda r: f"{r['environment']['platform']} | sklearn {r['environment']['scikit_learn']} | umap {r['environment']['umap_learn']}"
    if args.markdown:
        print(f"### Reproducibility: max |Δ| over K (tolerance {args.tol:g})\n")
        print(f"- A: `{args.a}` — {env(a)}\n- B: `{args.b}` — {env(b)}\n")
        print("| model | algorithm | " + " | ".join(cols) + " |")
        print("|---|---|" + "---|" * len(cols))
        for r in rows:
            cells = [f"{r[c]:.2e}{' ⚠' if r[c] > args.tol else ''}" for c in cols]
            print(f"| {r['model']} | {r['algorithm']} | " + " | ".join(cells) + " |")
        print(f"\n**Worst difference: {worst:.2e}** — {'within' if worst <= args.tol else 'ABOVE'} tolerance.")
    else:
        for r in rows:
            print(f"{r['model']:7s} {r['algorithm']:7s} " + " ".join(f"{c}={r[c]:.2e}" for c in cols))
        print(f"worst={worst:.2e} tol={args.tol:g}")
    if args.strict and worst > args.tol:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
