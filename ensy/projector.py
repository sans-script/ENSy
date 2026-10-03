"""TensorFlow Embedding Projector interchange (projector.tensorflow.org).

The Projector (tensorboard/plugins/projector/vz_projector) reads:

* vectors   — TSV, one row per point, tab-separated floats (``parseTensors``);
* metadata  — TSV; with more than one column the first row is the header
  (``parseMetadata``);
* bookmarks — the "state" file: a JSON array of ``State`` objects whose
  ``projections[i]`` hold point i's coordinates under keys such as
  ``"umap-0"``; loading a bookmark restores these coordinates without
  recomputing them (``vz-projector.ts``, ``loadState``);
* ``projector_config.json`` — the files above, by path or URL.

ENSy exports its exact layouts as bookmarks, so the Projector shows ENSy's
projection, and imports vectors, metadata and bookmarks so that ENSy can
measure layouts made in the Projector.
"""
from __future__ import annotations

import io
import json
import math
import zipfile

import numpy as np

# ENSy algorithm -> (Projector projection type, extra State fields). The Projector
# has no Isomap or uncentred PCA slot: those are stored under umap-* and named in
# the bookmark label.
STATE_SLOTS = {
    "pca": ("pca", {"pcaComponentDimensions": [0, 1, 2]}),
    "pca_nc": ("umap", {"umapIs3d": True}),
    "tsne": ("tsne", {"tSNEis3d": True, "tSNEIteration": 1000, "tSNEPerplexity": 5}),
    "tsne_p30": ("tsne", {"tSNEis3d": True, "tSNEIteration": 1000, "tSNEPerplexity": 30}),
    "umap": ("umap", {"umapIs3d": True}),
    "isomap": ("umap", {"umapIs3d": True}),
}


def _fmt(v: float) -> str:
    return repr(float(v)) if math.isfinite(v) else "0"


def vectors_tsv(X: np.ndarray) -> str:
    """One row per point, tab-separated, no header (Projector ``parseTensors``)."""
    return "\n".join("\t".join(_fmt(v) for v in row) for row in X) + "\n"


def metadata_tsv(columns: list[str], rows: list[list]) -> str:
    """Header row plus one row per point; tabs and newlines in values become spaces."""
    clean = lambda v: str(v).replace("\t", " ").replace("\n", " ")  # noqa: E731
    return "\n".join(["\t".join(columns)] + ["\t".join(clean(v) for v in r) for r in rows]) + "\n"


def make_state(label: str, algorithm: str, Y: np.ndarray, kept: np.ndarray, n_points: int, dim: int) -> dict:
    """A Projector bookmark holding ENSy's 3-D coordinates for every point.

    Points outside ``kept`` (Isomap's giant component) get zeros and are left out
    of ``filteredPoints``, so the Projector shows only the measured subset.
    """
    kind, extra = STATE_SLOTS[algorithm]
    full = np.zeros((n_points, 3))
    full[np.asarray(kept)] = Y[:, :3]
    state = {
        "label": label,
        "isSelected": False,
        "selectedProjection": kind,
        "dataSetDimensions": [n_points, dim],
        "tSNEIteration": 0, "tSNEPerplexity": 0, "tSNELearningRate": 10, "tSNEis3d": True,
        "umapIs3d": True, "umapNeighbors": 15, "umapMinDist": 0.1,
        "pcaComponentDimensions": [0, 1, 2],
        "customSelectedSearchByMetadataOption": "", "customXLeftText": "", "customXLeftRegex": False,
        "customXRightText": "", "customXRightRegex": False, "customYUpText": "", "customYUpRegex": False,
        "customYDownText": "", "customYDownRegex": False,
        "projections": [{f"{kind}-{c}": float(full[i, c]) for c in range(3)} for i in range(n_points)],
        "filteredPoints": None if len(kept) == n_points else [int(i) for i in kept],
        "selectedPoints": [],
        # no "shuffledDataIndices": an empty list is truthy in JavaScript and would
        # replace the Projector's own index list (loadState), breaking the view
        "cameraDef": None,
        "selectedColorOptionName": "category",
        "forceCategoricalColoring": False,
        "selectedLabelOption": "word",
    }
    state.update(extra)
    return state


def projector_config(name: str, shape: tuple[int, int], prefix: str = "") -> dict:
    return {"embeddings": [{
        "tensorName": name,
        "tensorShape": [int(shape[0]), int(shape[1])],
        "tensorPath": f"{prefix}vectors.tsv",
        "metadataPath": f"{prefix}metadata.tsv",
        "bookmarksPath": f"{prefix}state.txt",
    }]}


def export_zip(name: str, X: np.ndarray, columns: list[str], rows: list[list], states: list[dict]) -> bytes:
    """vectors.tsv, metadata.tsv, state.txt and projector_config.json in one zip."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("vectors.tsv", vectors_tsv(X))
        z.writestr("metadata.tsv", metadata_tsv(columns, rows))
        z.writestr("state.txt", json.dumps(states))
        z.writestr("projector_config.json", json.dumps(projector_config(name, X.shape), indent=2))
        z.writestr("README.txt", README)
    return buf.getvalue()


README = """Exported from ENSy for the TensorFlow Embedding Projector.

Local use: open https://projector.tensorflow.org, click "Load", choose
vectors.tsv (step 1) and metadata.tsv (step 2); then open the bookmarks panel
(bottom right) and upload state.txt. Each bookmark shows one of ENSy's layouts
exactly, labelled with its model, algorithm and co-ranking metrics at K = 10.

Shareable link: host these files at public URLs (e.g. a GitHub gist), set the
paths in projector_config.json to those URLs, and open
https://projector.tensorflow.org/?config=<URL of projector_config.json>.

Known issue in the live Projector (not in these files): clicking a bookmark
fails in current Chrome, Edge and Firefox, because the page reads the removed,
non-standard Event.path. Before clicking a bookmark, paste this line in the
browser console (F12):

  Object.defineProperty(Event.prototype, 'path', {get() { return this.composedPath(); }});

The Projector has no Isomap or uncentred-PCA slot; those layouts are stored
under its UMAP slot and named in the bookmark label. The Projector rescales
coordinates into its own cube, so absolute scales differ; layouts are equal.
"""


# ── import ──────────────────────────────────────────────────────────────────
def parse_vectors_tsv(text: str) -> tuple[np.ndarray, list[str] | None]:
    """Rows of floats; a non-numeric first field is taken as the label, as in the Projector."""
    rows, labels = [], []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.rstrip("\r").split("\t")
        try:
            float(fields[0])
            labels.append(None)
            rows.append([float(f) for f in fields])
        except ValueError:
            labels.append(fields[0])
            rows.append([float(f) for f in fields[1:]])
    if not rows:
        raise ValueError("vectors file is empty")
    dims = {len(r) for r in rows}
    if len(dims) != 1:
        raise ValueError(f"vector dimensions do not match: {sorted(dims)}")
    if dims.pop() < 2:
        raise ValueError("vectors need at least two dimensions")
    X = np.asarray(rows, dtype=np.float64)
    if not np.all(np.isfinite(X)):
        raise ValueError("vectors contain non-finite values")
    return X, (labels if any(l is not None for l in labels) else None)


def parse_metadata_tsv(text: str) -> tuple[list[str], list[list[str]]]:
    """A single column has no header; several columns have one (Projector rule)."""
    lines = [l.rstrip("\r") for l in text.splitlines() if l.strip() != ""]
    if not lines:
        raise ValueError("metadata file is empty")
    if "\t" not in lines[0]:
        return ["label"], [[l] for l in lines]
    header = lines[0].split("\t")
    rows = [l.split("\t") for l in lines[1:]]
    rows = [r + [""] * (len(header) - len(r)) for r in rows]
    return header, rows


def parse_states(text: str, n_points: int) -> list[dict]:
    """Bookmarks -> [{"label", "kind", "Y": (n, 3), "kept": indices}] in the file's order."""
    data = json.loads(text)
    if isinstance(data, dict):
        data = [data]
    out = []
    for k, st in enumerate(data):
        projs = st.get("projections") or []
        if len(projs) != n_points:
            raise ValueError(f"bookmark {k}: {len(projs)} points, the vectors have {n_points}")
        kind = st.get("selectedProjection") or "pca"
        prefix = "linear" if kind == "custom" else kind
        comps = sorted({key.split("-", 1)[1] for p in projs for key in p if key.startswith(prefix + "-")},
                       key=lambda c: (not c.isdigit(), int(c) if c.isdigit() else 0, c))
        if kind == "pca" and st.get("pcaComponentDimensions"):
            comps = [str(c) for c in st["pcaComponentDimensions"]]
        if len(comps) < 2:
            raise ValueError(f"bookmark {k}: no {kind} coordinates")
        comps = comps[:3]
        Y = np.zeros((n_points, 3))
        for i, p in enumerate(projs):
            for c, comp in enumerate(comps):
                Y[i, c] = float(p.get(f"{prefix}-{comp}", 0.0))
        kept = st.get("filteredPoints")
        kept = np.arange(n_points) if not kept else np.asarray(sorted(set(int(i) for i in kept)))
        out.append({"label": st.get("label") or f"{kind} bookmark {k + 1}", "kind": kind,
                    "Y": Y[kept], "kept": kept, "dims": len(comps)})
    return out
