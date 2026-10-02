"""Projection algorithms with the frozen parameter table (plan Section 7).

Every value that influences the output is spelled out, including the library
defaults the plan leaves implicit, so the record does not depend on the
installed library version's defaults.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.sparse.csgraph import connected_components

SEED = 42
N_COMPONENTS = 3

PARAMS: dict[str, dict] = {
    "pca": {"n_components": N_COMPONENTS, "centering": True, "solver": "full SVD of X_c"},
    "pca_nc": {"n_components": N_COMPONENTS, "centering": False, "solver": "full SVD of X",
               "role": "ablation"},
    "tsne": {"n_components": N_COMPONENTS, "perplexity": 5.0, "metric": "cosine",
             "random_state": SEED, "init": "pca", "learning_rate": "auto",
             "max_iter": 1000, "early_exaggeration": 12.0, "method": "barnes_hut",
             "angle": 0.5},
    # Reported alongside the frozen perplexity 5: the sign of B_NX depends on it
    # (sensitivity analysis); 30 is the library default.
    "tsne_p30": {"n_components": N_COMPONENTS, "perplexity": 30.0, "metric": "cosine",
                 "random_state": SEED, "init": "pca", "learning_rate": "auto",
                 "max_iter": 1000, "early_exaggeration": 12.0, "method": "barnes_hut",
                 "angle": 0.5},
    "umap": {"n_components": N_COMPONENTS, "n_neighbors": 15, "min_dist": 0.1,
             "metric": "cosine", "random_state": SEED},
    "isomap": {"n_components": N_COMPONENTS, "n_neighbors": 15, "metric": "cosine"},
}

MAIN_ALGORITHMS = ("pca", "tsne", "tsne_p30", "umap", "isomap")


def _svd_projection(X: np.ndarray, center: bool, n: int = N_COMPONENTS) -> tuple[np.ndarray, dict]:
    Xc = X - X.mean(axis=0) if center else X
    U, S, Vt = np.linalg.svd(Xc.astype(np.float64), full_matrices=False)
    # SVD sign is arbitrary: fix it so each component's largest loading is positive.
    signs = np.sign(Vt[np.arange(n), np.abs(Vt[:n]).argmax(axis=1)])
    V = Vt[:n].T * signs
    Y = Xc @ V
    var = S**2
    info = {"explained_variance_ratio": (var[:n] / var.sum()).tolist()}
    return Y, info


def isomap_graph_components(X: np.ndarray, n_neighbors: int, metric: str) -> np.ndarray:
    """Connected-component label of every point in the (symmetrised) k-NN graph."""
    from sklearn.neighbors import kneighbors_graph

    G = kneighbors_graph(X, n_neighbors, metric=metric, mode="connectivity")
    _, labels = connected_components(G, directed=False)
    return labels


def project(X: np.ndarray, algorithm: str, overrides: dict | None = None) -> tuple[np.ndarray, dict]:
    """Return (Y in R^3, diagnostics). X is expected L2-normalised.

    ``diagnostics["kept"]`` (Isomap only) lists the row indices that were
    projected; every other algorithm projects all rows. ``overrides`` replaces
    parameters for sensitivity analyses only -- never for the main results.
    """
    p = {**PARAMS[algorithm], **(overrides or {})}
    diag: dict = {}
    if algorithm in ("pca", "pca_nc"):
        Y, diag = _svd_projection(X, center=p["centering"], n=p["n_components"])
    elif algorithm in ("tsne", "tsne_p30"):
        from sklearn.manifold import TSNE

        if p["perplexity"] >= len(X):
            raise ValueError("t-SNE perplexity must be smaller than N")
        Y = TSNE(**p).fit_transform(X)
    elif algorithm == "umap":
        import umap

        with warnings.catch_warnings():
            # "n_jobs overridden by setting random_state" is the expected price of determinism
            warnings.filterwarnings("ignore", message=".*n_jobs.*")
            Y = umap.UMAP(**p).fit_transform(X)
    elif algorithm == "isomap":
        from sklearn.manifold import Isomap

        # Policy (TENENBAUM et al., 2000, note 19): points disconnected from the
        # giant component of the neighbourhood graph have no finite geodesic
        # distance and are deleted from further analysis, not bridged.
        labels = isomap_graph_components(X, p["n_neighbors"], p["metric"])
        giant = np.bincount(labels).argmax()
        kept = np.flatnonzero(labels == giant)
        diag["knn_graph_components"] = int(labels.max() + 1)
        diag["kept"] = kept.tolist()
        diag["removed"] = np.flatnonzero(labels != giant).tolist()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Y = Isomap(**p).fit_transform(X[kept])
        diag["warnings"] = sorted({str(w.message)[:160] for w in caught})
    else:
        raise KeyError(algorithm)
    Y = np.asarray(Y, dtype=np.float64)
    if not np.all(np.isfinite(Y)):
        raise FloatingPointError(f"{algorithm} produced non-finite coordinates")
    return Y, diag
