"""Experiment runner: 3 models x 4 algorithms (plan Section 10).

Run:  python -m ensy.experiment [--dataset data/dataset.json] [--models qwen,gemma,bert]
                                [--ablation] [--out results/]

Main results and ablations (PCA-NC) are written to separate files and never mixed.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import coranking, dataset as D, models as M, projection as P

K_MAIN = 10
K_CURVE = (5, 10, 15, 20)
CATEGORY_INDEX = {c: i for i, c in enumerate(D.CATEGORIES)}


def silhouette(Y: np.ndarray, labels: np.ndarray, metric: str = "euclidean") -> float:
    from sklearn.metrics import silhouette_score

    return float(silhouette_score(Y, labels, metric=metric))


def environment() -> dict:
    import scipy
    import sklearn
    import torch
    import transformers
    import umap

    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except Exception:
        sha, dirty = None, None
    return {
        "commit_sha": sha,
        "working_tree_dirty": dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "umap_learn": umap.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


# Sensitivity analysis (plan Sec. 7): reported separately, never mixed with the main matrix
SENSITIVITY_SEEDS = (0, 1, 2, 3, 42)
SENSITIVITY_GRID = (
    ("tsne", {"perplexity": 5.0}),
    ("tsne", {"perplexity": 30.0}),
    ("umap", {}),
)


def run_config(X: np.ndarray, labels: np.ndarray, algorithm: str, Ks=K_CURVE,
               overrides: dict | None = None) -> dict:
    Y, diag = P.project(X, algorithm, overrides)
    kept = np.asarray(diag.pop("kept", np.arange(len(X))))
    ev = coranking.evaluate(X[kept], Y, Ks, high_metric="cosine")
    return {
        "algorithm": algorithm,
        "params": {**P.PARAMS[algorithm], **(overrides or {})},
        "diagnostics": diag,
        "N_evaluated": int(len(kept)),
        "curves": ev["curves"],
        "silhouette_projected": silhouette(Y, labels[kept]),
        "Y": Y,
        "kept": kept,
        "local_b_nx": ev["local_b_nx"],
    }


def run(ds: dict, model_keys, algorithms, cache: M.EmbeddingCache | None = None) -> dict:
    cache = cache or M.EmbeddingCache()
    labels = np.array([CATEGORY_INDEX[it["category"]] for it in ds["items"]])
    out = {"configs": []}
    for mk in model_keys:
        X = cache.get(mk, ds)
        sil_high = silhouette(X, labels, metric="cosine")
        for alg in algorithms:
            res = run_config(X, labels, alg)
            for k in ("Y", "local_b_nx", "kept"):
                res.pop(k)
            if res["diagnostics"].get("removed"):
                res["diagnostics"]["removed_words"] = [
                    ds["items"][i]["word"] for i in res["diagnostics"]["removed"]]
            out["configs"].append({
                "model": mk,
                "hf_id": M.MODELS[mk].hf_id,
                "embedding_dim": int(X.shape[1]),
                "silhouette_high_dim_cosine": sil_high,
                **res,
            })
            print(f"  {mk:6s} {alg:7s} " + " ".join(
                f"K={c['K']}:T={c['trustworthiness']:.3f},C={c['continuity']:.3f},"
                f"B={c['B_NX']:+.3f}" for c in res["curves"] if c["K"] == K_MAIN))
    return out


def run_sensitivity(ds: dict, model_keys, cache: M.EmbeddingCache | None = None) -> dict:
    """Seed (and perplexity) sensitivity of the stochastic algorithms: mean and SD per K."""
    cache = cache or M.EmbeddingCache()
    labels = np.array([CATEGORY_INDEX[it["category"]] for it in ds["items"]])
    metrics = ("trustworthiness", "continuity", "Q_NX", "B_NX")
    out = {"seeds": list(SENSITIVITY_SEEDS), "configs": []}
    for mk in model_keys:
        X = cache.get(mk, ds)
        for alg, extra in SENSITIVITY_GRID:
            runs = [run_config(X, labels, alg, overrides={**extra, "random_state": seed})
                    for seed in SENSITIVITY_SEEDS]
            summary = []
            for ki, K in enumerate(K_CURVE):
                row = {"K": K}
                for m in metrics:
                    vals = np.array([r["curves"][ki][m] for r in runs])
                    row[m] = {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1)),
                              "min": float(vals.min()), "max": float(vals.max())}
                summary.append(row)
            sil = np.array([r["silhouette_projected"] for r in runs])
            out["configs"].append({
                "model": mk, "algorithm": alg, "overrides": extra, "curves": summary,
                "silhouette_projected": {"mean": float(sil.mean()), "sd": float(sil.std(ddof=1))},
            })
            k10 = summary[K_CURVE.index(K_MAIN)]
            print(f"  {mk:6s} {alg:5s} {extra or ''} K=10: T={k10['trustworthiness']['mean']:.3f}"
                  f"±{k10['trustworthiness']['sd']:.3f}  B_NX={k10['B_NX']['mean']:+.4f}±{k10['B_NX']['sd']:.4f}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, default=D.DEFAULT_OUT)
    ap.add_argument("--models", default=",".join(M.MODELS))
    ap.add_argument("--ablation", action="store_true", help="run PCA-NC instead of the main matrix")
    ap.add_argument("--sensitivity", action="store_true",
                    help="multi-seed / perplexity sensitivity of t-SNE and UMAP")
    ap.add_argument("--allow-provisional", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args(argv)

    ds = D.load(args.dataset)
    if not ds["frozen"] and not args.allow_provisional:
        raise SystemExit("dataset is not frozen (built without every model tokenizer); "
                         "pass --allow-provisional for exploratory runs only")
    keys = tuple(args.models.split(","))
    missing = set(keys) - set(ds["models"])
    if missing:
        raise SystemExit(f"dataset has no token ids for {sorted(missing)}")
    kind = "sensitivity" if args.sensitivity else "ablation" if args.ablation else "main"
    algorithms = ("pca_nc",) if args.ablation else P.MAIN_ALGORITHMS
    result = {
        "kind": kind,
        "provisional": not ds["frozen"],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_hash": ds["dataset_hash"],
        "N": ds["N"],
        "K_main": K_MAIN,
        "K_curve": list(K_CURVE),
        "preprocessing": "static input embedding, float32, L2-normalised",
        "extraction": "full download" if os.environ.get("ENSY_FULL_DOWNLOAD") else "HTTP Range rows",
        "high_dim_metric": "cosine (rank-equivalent to euclidean after L2)",
        "low_dim_metric": "euclidean",
        "seed": P.SEED,
        "isomap_policy": "giant component of the k-NN graph only (Tenenbaum et al., 2000, note 19)",
        "model_revisions": {k: ds["models"][k].get("revision") for k in keys},
        "environment": environment(),
        **(run_sensitivity(ds, keys) if args.sensitivity else run(ds, keys, algorithms)),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    name = f"{kind}{'_provisional' if result['provisional'] else ''}_{ds['dataset_hash'][:12]}.json"
    path = args.out / name
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
