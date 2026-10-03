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
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ensy import coranking, dataset as D, models as M, projection as P, projector as TP
from ensy.experiment import CATEGORY_INDEX, K_CURVE, K_MAIN, silhouette

ROOT = Path(__file__).resolve().parent
DATASET_PATHS = (ROOT / "data" / "dataset.json", ROOT / "data" / "dataset.provisional.json")
# The viewer's K slider: every quantity is precomputed for K = 1..K_SLIDER_MAX
# (about 0.25 s per projection), so the slider never waits on the server. The
# paper reports only K_CURVE; the slider is an exploration aid.
K_SLIDER_MAX = 50
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

# Datasets imported from the TF Embedding Projector (vectors + metadata TSV and
# optional bookmarks), keyed "ext-<hash>". They live in memory only and are never
# mixed with the frozen dataset of the paper. Bookmarks are served as algorithms
# "bm0", "bm1", ...: their coordinates are measured as given, never re-projected.
_external: dict[str, dict] = {}
EXT_MAX_POINTS = 3000  # the co-ranking matrix is N x N
EXT_PALETTE = ["#4338ca", "#a16207", "#be185d", "#0f766e", "#7c3aed",
               "#b45309", "#475569", "#9f1239", "#1d4ed8", "#4d7c0f"]
ALGO_NAMES = {"pca": "PCA", "pca_nc": "PCA (uncentred)", "tsne": "t-SNE (perplexity 5)",
              "tsne_p30": "t-SNE (perplexity 30)", "umap": "UMAP", "isomap": "Isomap"}
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


def _is_ext(model: str) -> bool:
    return model.startswith("ext-")


def _dataset_for(model: str) -> dict:
    """The frozen dataset, or an imported one for keys "ext-..."."""
    if _is_ext(model):
        if model not in _external:
            raise HTTPException(404, detail=f"no imported dataset '{model}' (imports live until the server restarts)")
        return _external[model]["ds"]
    return _require_dataset()


def _k_max(n: int) -> int:
    """Largest K for this many points: the slider's 50, or K < N/2 for small sets."""
    return max(1, min(K_SLIDER_MAX, (n - 1) // 2))


def _check(model: str, algorithm: str, ds: dict) -> None:
    if _is_ext(model):
        if algorithm.startswith("bm"):
            idx = algorithm[2:]
            if not idx.isdigit() or int(idx) >= len(_external[model]["bookmarks"]):
                raise HTTPException(400, detail=f"unknown bookmark '{algorithm}'")
        elif algorithm not in P.PARAMS:
            raise HTTPException(400, detail=f"unsupported projection engine '{algorithm}'")
        return
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
    index = ds.get("category_index", CATEGORY_INDEX)
    return np.array([index[it["category"]] for it in ds["items"]])


def _silhouette_or_none(Y: np.ndarray, labels: np.ndarray):
    """The Silhouette needs at least two categories, each with fewer points than N."""
    if len(set(labels.tolist())) < 2 or len(set(labels.tolist())) >= len(labels):
        return None
    return silhouette(Y, labels)


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
        if _is_ext(model):
            X = _external[model]["X"]
        else:
            try:
                X = _cache.get(model, ds)
            except Exception as exc:
                raise HTTPException(503, detail=f"cannot load embeddings for {model}: {exc}")
        labels = _labels(ds)
        if algorithm.startswith("bm"):
            bm = _external[model]["bookmarks"][int(algorithm[2:])]
            Y = bm["Y"]
            diag = {"source": "TF Embedding Projector bookmark", "bookmark": bm["label"],
                    "projector_projection": bm["kind"], "dimensions": bm["dims"], "kept": bm["kept"]}
        else:
            Y, diag = P.project(X, algorithm)
        # Isomap may drop points outside the giant k-NN component; everything is
        # evaluated on the projected subset and indices are mapped back to items.
        kept = np.asarray(diag.pop("kept", np.arange(len(X))))
        if diag.get("removed"):
            diag["removed_words"] = [ds["items"][i]["word"] for i in diag["removed"]]
        ks = tuple(range(1, _k_max(len(kept)) + 1))
        ev = coranking.evaluate(X[kept], Y, ks, high_metric="cosine")
        rho, r = ev["rho"], ev["r"]
        out = {
            "X": X, "Y": Y, "kept": kept, "diag": diag, "ks": ks,
            "curves": [c for c in ev["curves"] if c["K"] in K_CURVE],
            "curves_all": ev["curves"],
            # full precision: the mean of B_NX,i(K) must equal B_NX(K) exactly
            "local": {K: v.tolist() for K, v in ev["local_b_nx"].items()},
            "hd_order": kept[np.argsort(rho, axis=1)[:, 1:len(ks) + 1]],
            "ld_order": kept[np.argsort(r, axis=1)[:, 1:len(ks) + 1]],
            "silhouette": _silhouette_or_none(Y, labels[kept]),
        }
        _projections[key] = out
        return out


def _payload(model: str, algorithm: str, ds: dict, silhouette_2d: bool = False) -> dict:
    res = _compute(model, algorithm, ds)
    sil_2d = _with_silhouette_2d(res, algorithm, ds) if silhouette_2d else None
    X, Y, kept = res["X"], res["Y"], res["kept"]
    N = len(kept)
    colors = ds.get("category_colors", CATEGORY_COLORS)
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
            "color": colors[item["category"]],
            "coords": {"x2d": float(Y[row, 0]), "y2d": float(Y[row, 1]),
                       "x3d": float(Y[row, 0]), "y3d": float(Y[row, 1]), "z3d": float(Y[row, 2])},
            "top5_neighbors": top5,
            "hd_neighbors": hd.tolist(),
            "ld_neighbors": res["ld_order"][row].tolist(),
            "b_nx_local": {str(K): res["local"][K][row] for K in res["ks"]},
        })
    return {
        "status": "STABLE",
        "meta": {
            "model": ds.get("display_name") or M.MODELS[model].hf_id,
            "model_key": model,
            "algorithm": algorithm,
            "params": P.PARAMS.get(algorithm, {"bookmark": res["diag"].get("bookmark")}),
            "diagnostics": res["diag"],
            "embedding_dim": int(X.shape[1]),
            "total_tokens": N,
            "dataset_hash": ds["dataset_hash"],
            "dataset_frozen": ds["frozen"],
            "dataset_note": ds.get("note"),
            "reduced_sample": N < REDUCED_SAMPLE_N,
            "K_main": K_MAIN,
            "K_curve": [k for k in K_CURVE if k in res["ks"]],
            "metrics": res["curves"],
            "K_slider_max": len(res["ks"]),
            "metrics_all": res["curves_all"],
            "silhouette_projected": res["silhouette"],
            "silhouette_projected_2d": sil_2d,
            "category_colors": colors,
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
    model, algorithm = model.lower(), algorithm.lower()
    ds = _dataset_for(model)
    _check(model, algorithm, ds)
    return _payload(model, algorithm, ds, silhouette_2d)


@app.get("/api/projection/{algorithm}")
def get_projection_default(algorithm: str, model: str = Query("qwen"), silhouette_2d: bool = Query(False)):
    """Backwards-compatible route (defaults to Qwen)."""
    return get_projection(model, algorithm, silhouette_2d)


@app.get("/api/export/projector")
def export_projector(model: str = Query("qwen")):
    """A zip for the TF Embedding Projector: vectors.tsv, metadata.tsv, a bookmark per
    ENSy projection (state.txt, ENSy's exact coordinates) and projector_config.json."""
    model = model.lower()
    ds = _dataset_for(model)
    _check(model, "pca", ds)
    name = ds.get("display_name") or M.MODELS[model].hf_id
    algorithms = list(P.MAIN_ALGORITHMS)
    states, local_cols = [], {}
    X = None
    for alg in algorithms:
        res = _compute(model, alg, ds)
        X = res["X"]
        k = 10 if 10 in res["ks"] else res["ks"][-1]
        m = next(c for c in res["curves_all"] if c["K"] == k)
        label = (f"ENSy — {name} — {ALGO_NAMES[alg]} "
                 f"(M_T({k}) = {m['trustworthiness']:.3f}, B_NX({k}) = {m['B_NX']:+.3f})")
        states.append(TP.make_state(label, alg, res["Y"], res["kept"], len(ds["items"]), X.shape[1]))
        col = {int(i): res["local"][k][row] for row, i in enumerate(res["kept"])}
        local_cols[f"b_nx_{k}_{alg}"] = col
    columns = ["word", "category", "token_id"] + list(local_cols)
    rows = []
    for i, it in enumerate(ds["items"]):
        rows.append([it["word"], it["category"], it["token_ids"].get(model, i)]
                    + [(f"{c[i]:.4f}" if i in c else "") for c in local_cols.values()])
    body = TP.export_zip(f"ENSy — {name}", X, columns, rows, states)
    return Response(body, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="ensy_{model}_projector.zip"'})


class ImportRequest(BaseModel):
    name: str = "Imported"
    vectors: str
    metadata: str | None = None
    state: str | None = None
    label_column: str | None = None
    partition_column: str | None = None


@app.post("/api/import/projector")
def import_projector(req: ImportRequest):
    """Vectors and metadata TSV (and optionally a bookmark file) from the TF Embedding
    Projector, as an external dataset: ENSy projects it with its own algorithms and
    measures the bookmarked layouts as given."""
    import hashlib

    try:
        X, vec_labels = TP.parse_vectors_tsv(req.vectors)
        n = len(X)
        if not 10 <= n <= EXT_MAX_POINTS:
            raise ValueError(f"{n} points; ENSy measures between 10 and {EXT_MAX_POINTS}")
        cols, rows = TP.parse_metadata_tsv(req.metadata) if req.metadata else ([], [])
        if rows and len(rows) != n:
            raise ValueError(f"metadata has {len(rows)} rows, the vectors have {n}")
        bookmarks = TP.parse_states(req.state, n) if req.state else []
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc))

    column = {c: [r[j] for r in rows] for j, c in enumerate(cols)}
    for c in (req.label_column, req.partition_column):
        if c and c not in column:
            raise HTTPException(400, detail=f"metadata has no column '{c}' (columns: {cols})")
    label_col = req.label_column or next((c for c in ("word", "label", "name") if c in column), None)
    if label_col is None:
        label_col = next((c for c in cols if not all(_is_number(v) for v in column[c])), None)
    words = (column[label_col] if label_col else vec_labels) or [f"#{i}" for i in range(n)]
    part_col = req.partition_column or ("category" if "category" in column else None)
    if part_col is None:
        part_col = next((c for c in cols if c != label_col and 2 <= len(set(column[c])) <= len(EXT_PALETTE)), None)
    categories = column[part_col] if part_col else ["all"] * n
    order = sorted(set(categories), key=categories.index)
    if len(order) > len(EXT_PALETTE):
        raise HTTPException(400, detail=f"partition '{part_col}' has {len(order)} values; at most {len(EXT_PALETTE)}")

    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)  # as ENSy's embeddings
    digest = hashlib.sha256((req.vectors + "\0" + (req.metadata or "")).encode()).hexdigest()
    key = f"ext-{digest[:10]}"
    ds = {
        "display_name": req.name, "dataset_hash": digest, "frozen": False, "N": n,
        "note": f"from the TF Embedding Projector ({req.name}); not part of the paper's dataset",
        "models": {key: {}}, "category_index": {c: i for i, c in enumerate(order)},
        "category_colors": {c: EXT_PALETTE[i] for i, c in enumerate(order)},
        "items": [{"word": str(words[i]), "category": categories[i], "token_ids": {key: i}} for i in range(n)],
    }
    with _lock:
        for k in [k for k in _projections if k[0] == key]:
            del _projections[k]
        _external[key] = {"ds": ds, "X": X, "bookmarks": bookmarks, "name": req.name,
                          "label_column": label_col, "partition_column": part_col}
    return _external_summary(key)


def _is_number(v: str) -> bool:
    try:
        float(v)
        return True
    except ValueError:
        return False


def _external_summary(key: str) -> dict:
    e = _external[key]
    return {"key": key, "name": e["name"], "N": e["ds"]["N"], "dim": int(e["X"].shape[1]),
            "label_column": e["label_column"], "partition_column": e["partition_column"],
            "categories": list(e["ds"]["category_index"]),
            "bookmarks": [{"id": f"bm{i}", "label": b["label"], "kind": b["kind"], "dims": b["dims"]}
                          for i, b in enumerate(e["bookmarks"])]}


@app.get("/api/external")
def list_external():
    return [_external_summary(k) for k in _external]


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
