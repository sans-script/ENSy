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
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ensy import coranking, dataset as D, models as M, projection as P
from ensy.experiment import CATEGORY_INDEX, K_CURVE, K_MAIN, silhouette

ROOT = Path(__file__).resolve().parent
DATASET_PATHS = (ROOT / "data" / "dataset.json", ROOT / "data" / "dataset.provisional.json")
K_PROBE_MAX = max(K_CURVE)
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

_lock = threading.Lock()
_cache = M.EmbeddingCache()
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


def _compute(model: str, algorithm: str, ds: dict) -> dict:
    key = (model, ds["dataset_hash"], algorithm)
    with _lock:
        if key in _projections:
            return _projections[key]
        try:
            X = _cache.get(model, ds)
        except Exception as exc:
            raise HTTPException(503, detail=f"cannot load embeddings for {model}: {exc}")
        labels = np.array([CATEGORY_INDEX[it["category"]] for it in ds["items"]])
        Y, diag = P.project(X, algorithm)
        ev = coranking.evaluate(X, Y, K_CURVE, high_metric="cosine")
        rho, r = ev["rho"], ev["r"]
        out = {
            "X": X, "Y": Y, "diag": diag, "curves": ev["curves"],
            "local": {K: v.tolist() for K, v in ev["local_b_nx"].items()},
            "hd_order": np.argsort(rho, axis=1)[:, 1:K_PROBE_MAX + 1],
            "ld_order": np.argsort(r, axis=1)[:, 1:K_PROBE_MAX + 1],
            "silhouette": silhouette(Y, labels),
        }
        _projections[key] = out
        return out


def _payload(model: str, algorithm: str, ds: dict) -> dict:
    res = _compute(model, algorithm, ds)
    X, Y = res["X"], res["Y"]
    N = len(ds["items"])
    nodes = []
    for i, item in enumerate(ds["items"]):
        hd = res["hd_order"][i]
        top5 = [{"label": ds["items"][j]["word"], "cosine_similarity": round(float(X[i] @ X[j]), 6)}
                for j in hd[:5]]
        nodes.append({
            "id": i,
            "label": item["word"],
            "token_id": item["token_ids"][model],
            "cluster": item["category"],
            "color": CATEGORY_COLORS[item["category"]],
            "coords": {"x2d": float(Y[i, 0]), "y2d": float(Y[i, 1]),
                       "x3d": float(Y[i, 0]), "y3d": float(Y[i, 1]), "z3d": float(Y[i, 2])},
            "top5_neighbors": top5,
            "hd_neighbors": hd.tolist(),
            "ld_neighbors": res["ld_order"][i].tolist(),
            "b_nx_local": {str(K): res["local"][K][i] for K in K_CURVE},
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
            "silhouette_projected": res["silhouette"],
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
def get_projection(model: str, algorithm: str):
    ds = _require_dataset()
    model, algorithm = model.lower(), algorithm.lower()
    _check(model, algorithm, ds)
    return _payload(model, algorithm, ds)


@app.get("/api/projection/{algorithm}")
def get_projection_default(algorithm: str, model: str = Query("qwen")):
    """Backwards-compatible route (defaults to Qwen)."""
    return get_projection(model, algorithm)


@app.get("/api/compare")
def compare(models: str = Query(None)):
    """Metric curves for every (model, algorithm) of the main matrix (F5)."""
    ds = _require_dataset()
    keys = models.split(",") if models else [k for k in M.MODELS if k in ds["models"]]
    rows = []
    for mk in keys:
        for alg in P.MAIN_ALGORITHMS:
            _check(mk, alg, ds)
            res = _compute(mk, alg, ds)
            rows.append({"model": mk, "algorithm": alg, "curves": res["curves"],
                         "silhouette_projected": res["silhouette"]})
    return {"dataset_hash": ds["dataset_hash"], "frozen": ds["frozen"], "N": ds["N"], "rows": rows}


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
