"""ENSy <-> TF Embedding Projector through the API (in-process, no server needed).

The round trip exports ENSy's layouts as Projector bookmarks, imports the same
files back, and checks that ENSy measures the bookmarked layouts exactly as it
measured its own projections.
"""
import io
import json
import math
import zipfile

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import app as ensy_app  # noqa: E402

client = TestClient(ensy_app.app)


def _small_dataset(n=40, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    meta = "word\tcategory\n" + "\n".join(f"w{i}\t{'ab'[i % 2]}" for i in range(n)) + "\n"
    vec = "\n".join("\t".join(repr(float(v)) for v in row) for row in X) + "\n"
    return vec, meta


def test_import_then_measure_and_probe():
    vec, meta = _small_dataset()
    r = client.post("/api/import/projector", json={"name": "toy", "vectors": vec, "metadata": meta})
    assert r.status_code == 200, r.text
    info = r.json()
    assert info["N"] == 40 and info["partition_column"] == "category" and info["categories"] == ["a", "b"]
    data = client.get(f"/api/projection/{info['key']}/pca").json()
    meta_ = data["meta"]
    assert meta_["dataset_frozen"] is False and meta_["model"] == "toy"
    k_max = meta_["K_slider_max"]
    assert k_max == 19  # K < N/2 for 40 points
    assert data["nodes"][0]["label"] == "w0"
    for c in meta_["metrics"]:
        local = [n["b_nx_local"][str(c["K"])] for n in data["nodes"]]
        assert math.isclose(sum(local) / len(local), c["B_NX"], abs_tol=1e-9)


def test_round_trip_export_import_reproduces_metrics():
    vec, meta = _small_dataset(seed=1)
    key = client.post("/api/import/projector", json={"name": "rt", "vectors": vec, "metadata": meta}).json()["key"]
    z = zipfile.ZipFile(io.BytesIO(client.get(f"/api/export/projector?model={key}").content))
    states = z.read("state.txt").decode()
    assert len(json.loads(states)) == 5
    # import the exported files: each bookmark is measured as given
    info = client.post("/api/import/projector", json={
        "name": "rt2", "vectors": z.read("vectors.tsv").decode(),
        "metadata": z.read("metadata.tsv").decode(), "state": states}).json()
    assert [b["kind"] for b in info["bookmarks"]] == ["pca", "tsne", "tsne", "umap", "umap"]
    for bm, alg in zip(info["bookmarks"], ["pca", "tsne", "tsne_p30", "umap", "isomap"]):
        orig = client.get(f"/api/projection/{key}/{alg}").json()["meta"]["metrics"]
        back = client.get(f"/api/projection/{info['key']}/{bm['id']}").json()["meta"]["metrics"]
        for a, b in zip(orig, back):
            assert a["K"] == b["K"]
            assert math.isclose(a["trustworthiness"], b["trustworthiness"], abs_tol=1e-9)
            assert math.isclose(a["B_NX"], b["B_NX"], abs_tol=1e-9)


def test_import_rejects_mismatched_metadata():
    vec, _ = _small_dataset()
    r = client.post("/api/import/projector", json={"vectors": vec, "metadata": "word\tcategory\nx\ty\n"})
    assert r.status_code == 400 and "rows" in r.json()["detail"]


def test_unknown_bookmark_is_400():
    vec, meta = _small_dataset(seed=2)
    key = client.post("/api/import/projector", json={"vectors": vec, "metadata": meta}).json()["key"]
    assert client.get(f"/api/projection/{key}/bm3").status_code == 400
