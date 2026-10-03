"""ENSy local research server (plan v2.1.2).

Serves projections of the frozen PTB-derived lexical dataset with co-ranking
metrics. Nothing here fabricates data: if the dataset or a model is missing the
API says so (HTTP 503/409) instead of returning placeholder vectors.

Run:  .venv/Scripts/python.exe app.py   (port 3000)
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ensy import coranking, dataset as D, models as M, projection as P
from ensy.experiment import CATEGORY_INDEX, K_CURVE, K_MAIN, silhouette

ROOT = Path(__file__).resolve().parent
DATASET_PATHS = (ROOT / "data" / "dataset.json", ROOT / "data" / "dataset.provisional.json")
# The viewer's K slider: every quantity is precomputed for K = 1..K_SLIDER_MAX
# (about 0.25 s per projection), so the slider never waits on the server. The
# paper reports only K_CURVE; the slider is an exploration aid.
K_SLIDER_MAX = 50
K_SLIDER = tuple(range(1, K_SLIDER_MAX + 1))
K_PROBE_MAX = K_SLIDER_MAX
REDUCED_SAMPLE_N = 100  # engineering convention, not a statistical threshold (plan 9.6)

# Layer 1 (plan 9.7): reference partition. Green is reserved for probing.
CATEGORY_COLORS = {
    "content_noun": "#4338ca",
    "content_verb": "#a16207",
    "function_word": "#be185d",
}

app = FastAPI(title="ENSy — Embedding Navigation System")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"],
                   allow_methods=["*"], allow_headers=["*"])
# projections carry every K from 1 to 50 (~0.9 MB of JSON); gzip shrinks them several-fold
app.add_middleware(GZipMiddleware, minimum_size=1024)

_lock = threading.Lock()
# extracted rows are kept on disk: a model is fetched from the Hub once, not on every start
_cache = M.EmbeddingCache(disk_dir=ROOT / ".cache" / "embeddings")
_projections: dict[tuple, dict] = {}
_state: dict = {"dataset": None, "dataset_path": None, "error": None}


def _load_dataset() -> None:
    for path in DATASET_PATHS:
        if path.exists():
            try:
                _state["dataset"] = D.load(path)
                _state["dataset_path"] = str(path.relative_to(ROOT))
                _state["error"] = None
            except Exception as exc:  # hash mismatch, malformed file
                _state["error"] = f"{path.name}: {exc}"
            return
    _state["error"] = "no dataset found; build it with POST /api/dataset or python -m ensy.dataset"


_load_dataset()


def _require_dataset() -> dict:
    ds = _state["dataset"]
    if ds is None:
        raise HTTPException(503, detail=_state["error"])
    return ds


def _check(model: str, algorithm: str, ds: dict) -> None:
    if model not in M.MODELS:
        raise HTTPException(400, detail=f"unknown model '{model}'")
    if algorithm not in P.PARAMS:
        raise HTTPException(400, detail=f"unsupported projection engine '{algorithm}'")
    if model not in ds["models"]:
        raise HTTPException(409, detail=f"dataset has no token ids for '{model}' "
                                        f"(built with {sorted(ds['models'])})")


def _silhouette_2d(X: np.ndarray, algorithm: str, labels: np.ndarray) -> float:
    """Silhouette on a genuine 2-D projection (same parameters, n_components = 2)."""
    Y2, diag2 = P.project(X, algorithm, {"n_components": 2})
    kept2 = np.asarray(diag2.get("kept", np.arange(len(X))))
    return silhouette(Y2, labels[kept2])


def _labels(ds: dict) -> np.ndarray:
    return np.array([CATEGORY_INDEX[it["category"]] for it in ds["items"]])


def _with_silhouette_2d(res: dict, algorithm: str, ds: dict) -> float:
    """The 2D Silhouette needs a second projection, so it is computed (and cached)
    only when a request asks for it."""
    with _lock:
        if "silhouette_2d" not in res:
            res["silhouette_2d"] = _silhouette_2d(res["X"], algorithm, _labels(ds))
        return res["silhouette_2d"]


def _compute(model: str, algorithm: str, ds: dict) -> dict:
    key = (model, ds["dataset_hash"], algorithm)
    with _lock:
        if key in _projections:
            return _projections[key]
        try:
            X = _cache.get(model, ds)
        except Exception as exc:
            raise HTTPException(503, detail=f"cannot load embeddings for {model}: {exc}")
        labels = _labels(ds)
        Y, diag = P.project(X, algorithm)
        # Isomap may drop points outside the giant k-NN component; everything is
        # evaluated on the projected subset and indices are mapped back to items.
        kept = np.asarray(diag.pop("kept", np.arange(len(X))))
        if diag.get("removed"):
            diag["removed_words"] = [ds["items"][i]["word"] for i in diag["removed"]]
        ev = coranking.evaluate(X[kept], Y, K_SLIDER, high_metric="cosine")
        rho, r = ev["rho"], ev["r"]
        out = {
            "X": X, "Y": Y, "kept": kept, "diag": diag,
            "curves": [c for c in ev["curves"] if c["K"] in K_CURVE],
            "curves_all": ev["curves"],
            # full precision: the mean of B_NX,i(K) must equal B_NX(K) exactly
            "local": {K: v.tolist() for K, v in ev["local_b_nx"].items()},
            "hd_order": kept[np.argsort(rho, axis=1)[:, 1:K_PROBE_MAX + 1]],
            "ld_order": kept[np.argsort(r, axis=1)[:, 1:K_PROBE_MAX + 1]],
            "silhouette": silhouette(Y, labels[kept]),
        }
        _projections[key] = out
        return out


def _payload(model: str, algorithm: str, ds: dict, silhouette_2d: bool = False) -> dict:
    res = _compute(model, algorithm, ds)
    sil_2d = _with_silhouette_2d(res, algorithm, ds) if silhouette_2d else None
    X, Y, kept = res["X"], res["Y"], res["kept"]
    N = len(kept)
    nodes = []
    for row, i in enumerate(kept):
        i = int(i)
        item = ds["items"][i]
        hd = res["hd_order"][row]
        top5 = [{"label": ds["items"][j]["word"], "cosine_similarity": round(float(X[i] @ X[j]), 6)}
                for j in hd[:5]]
        nodes.append({
            "id": i,
            "label": item["word"],
            "token_id": item["token_ids"][model],
            "cluster": item["category"],
            "color": CATEGORY_COLORS[item["category"]],
            "coords": {"x2d": float(Y[row, 0]), "y2d": float(Y[row, 1]),
                       "x3d": float(Y[row, 0]), "y3d": float(Y[row, 1]), "z3d": float(Y[row, 2])},
            "top5_neighbors": top5,
            "hd_neighbors": hd.tolist(),
            "ld_neighbors": res["ld_order"][row].tolist(),
            "b_nx_local": {str(K): res["local"][K][row] for K in K_SLIDER},
        })
    return {
        "status": "STABLE",
        "meta": {
            "model": M.MODELS[model].hf_id,
            "model_key": model,
            "algorithm": algorithm,
            "params": P.PARAMS[algorithm],
            "diagnostics": res["diag"],
            "embedding_dim": int(X.shape[1]),
            "total_tokens": N,
            "dataset_hash": ds["dataset_hash"],
            "dataset_frozen": ds["frozen"],
            "dataset_note": ds.get("note"),
            "reduced_sample": N < REDUCED_SAMPLE_N,
            "K_main": K_MAIN,
            "K_curve": list(K_CURVE),
            "metrics": res["curves"],
            "K_slider_max": K_SLIDER_MAX,
            "metrics_all": res["curves_all"],
            "silhouette_projected": res["silhouette"],
            "silhouette_projected_2d": sil_2d,
            "category_colors": CATEGORY_COLORS,
        },
        "nodes": nodes,
    }


@app.get("/api/health")
def health():
    ds = _state["dataset"]
    return {"dataset_loaded": ds is not None, "dataset_path": _state["dataset_path"],
            "error": _state["error"], "N": ds["N"] if ds else None,
            "frozen": ds["frozen"] if ds else None}


@app.get("/api/models")
def list_models():
    ds = _state["dataset"]
    return [{"key": k, "hf_id": s.hf_id, "family": s.family, "gated": s.gated,
             "token_available": bool(M.hf_token()) if s.gated else True,
             "in_dataset": bool(ds and k in ds["models"])}
            for k, s in M.MODELS.items()]


@app.get("/api/dataset")
def get_dataset():
    ds = _require_dataset()
    return {k: v for k, v in ds.items() if k != "items"} | {
        "items": [{"word": it["word"], "category": it["category"]} for it in ds["items"]]}


class DatasetRequest(BaseModel):
    models: list[str] = list(M.MODELS)
    force: bool = False


@app.post("/api/dataset")
def build_dataset(req: DatasetRequest):
    unknown = set(req.models) - set(M.MODELS)
    if unknown:
        raise HTTPException(400, detail=f"unknown models {sorted(unknown)}")
    complete = set(req.models) == set(M.MODELS)
    path = DATASET_PATHS[0] if complete else DATASET_PATHS[1]
    if path.exists() and complete and not req.force:
        raise HTTPException(409, detail="a frozen dataset already exists; pass force=true to rebuild")
    try:
        ds = D.build(tuple(req.models))
    except Exception as exc:
        raise HTTPException(502, detail=f"dataset build failed: {exc}")
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ds, indent=1, ensure_ascii=False), encoding="utf-8")
    with _lock:
        _projections.clear()
        _cache.clear()
    _load_dataset()
    return {"path": str(path.relative_to(ROOT)), "N": ds["N"], "frozen": ds["frozen"],
            "dataset_hash": ds["dataset_hash"], "counts": ds["counts"]}


@app.get("/api/projection/{model}/{algorithm}")
def get_projection(model: str, algorithm: str, silhouette_2d: bool = Query(False)):
    ds = _require_dataset()
    model, algorithm = model.lower(), algorithm.lower()
    _check(model, algorithm, ds)
    return _payload(model, algorithm, ds, silhouette_2d)


@app.get("/api/projection/{algorithm}")
def get_projection_default(algorithm: str, model: str = Query("qwen"), silhouette_2d: bool = Query(False)):
    """Backwards-compatible route (defaults to Qwen)."""
    return get_projection(model, algorithm, silhouette_2d)


@app.get("/api/compare")
def compare(models: str = Query(None), silhouette_2d: bool = Query(False)):
    """Metric curves for every (model, algorithm) of the main matrix (F5)."""
    ds = _require_dataset()
    keys = models.split(",") if models else [k for k in M.MODELS if k in ds["models"]]
    rows = []
    for mk in keys:
        for alg in P.MAIN_ALGORITHMS:
            _check(mk, alg, ds)
            res = _compute(mk, alg, ds)
            rows.append({"model": mk, "algorithm": alg, "curves": res["curves"],
                         "silhouette_projected": res["silhouette"],
                         "silhouette_projected_2d": _with_silhouette_2d(res, alg, ds) if silhouette_2d else None})
    return {"dataset_hash": ds["dataset_hash"], "frozen": ds["frozen"], "N": ds["N"], "rows": rows}


def _warm_up() -> None:
    """
    Background start-up work so the first projection a user asks for is fast:
    load the embeddings of every model in the dataset (from disk after the first
    run), compile UMAP's numba kernels and t-SNE's first call on a toy problem,
    then compute the viewer's default projection.
    """
    ds = _state["dataset"]
    if ds is None:
        return
    try:
        rng = np.random.default_rng(0)
        toy = rng.normal(size=(60, 8))
        P.project(toy, "umap")
        P.project(toy, "tsne")
        for model in (k for k in M.MODELS if k in ds["models"]):
            with _lock:
                try:
                    _cache.get(model, ds)
                except Exception:  # gated model without a token, offline, ...
                    continue
        if "qwen" in ds["models"]:
            _compute("qwen", "umap", ds)
    except Exception as exc:  # warm-up is an optimisation; never take the server down
        print(f"[warm-up] skipped: {exc}")


if os.environ.get("ENSY_WARMUP", "1") != "0":
    threading.Thread(target=_warm_up, name="ensy-warm-up", daemon=True).start()


# ── static frontend (vite build output) ────────────────────────────────
dist_path = ROOT / "dist"
if dist_path.exists():
    if (dist_path / "assets").exists():
        app.mount("/assets", StaticFiles(directory=dist_path / "assets"), name="assets")

    @app.get("/")
    async def serve_index():
        return FileResponse(dist_path / "index.html")

    @app.get("/{rest_of_path:path}")
    async def spa_fallback(rest_of_path: str):
        if rest_of_path.startswith("api/"):
            raise HTTPException(404, detail="API route not found")
        target = (dist_path / rest_of_path).resolve()
        if dist_path.resolve() in target.parents and target.is_file():
            return FileResponse(target)
        return FileResponse(dist_path / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=int(os.environ.get("PORT", 3000)), reload=False)
