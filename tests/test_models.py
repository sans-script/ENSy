"""Extraction protocol tests (plan Section 4). Need the HF cache or network."""
import numpy as np
import pytest

from ensy import models as M
from ensy import projection as P

@pytest.mark.network
@pytest.mark.parametrize("key,expected_dim", [("qwen", 896), ("bert", 768)])
def test_safetensors_matrix_equals_get_input_embeddings(key, expected_dim):
    from transformers import AutoModel

    W = M.embedding_matrix(key)
    assert W.dtype == np.float32 and W.shape[1] == expected_dim
    model = AutoModel.from_pretrained(M.MODELS[key].hf_id, torch_dtype="float32")
    ref = model.get_input_embeddings().weight.detach().numpy()
    assert W.shape == ref.shape
    np.testing.assert_array_equal(W, ref)


@pytest.mark.network
@pytest.mark.parametrize("key", ["qwen", "bert"])
def test_ranged_rows_equal_full_matrix(key):
    """HTTP Range extraction is bit-identical to the full-download reference path."""
    W = M.embedding_matrix(key)
    ids = [0, 1, 2, 1000, 1001, 2603, W.shape[0] - 1, 1000]  # contiguous runs, repeat, last row
    np.testing.assert_array_equal(M.embedding_rows(key, ids), W[ids])


def test_bf16_decoding_is_exact():
    import torch

    x = torch.tensor([1.0, -2.5, 3.14159, 1e-8, 65504.0, -0.0], dtype=torch.bfloat16)
    raw = x.view(torch.int16).numpy().tobytes()
    np.testing.assert_array_equal(M._decode(raw, "BF16"), x.float().numpy())


def test_l2_normalisation():
    X = M.l2_normalise(np.array([[3.0, 4.0], [1.0, 0.0]]))
    np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1.0, rtol=1e-6)
    with pytest.raises(ValueError):
        M.l2_normalise(np.zeros((1, 3)))


def test_projection_is_deterministic():
    rng = np.random.default_rng(0)
    X = M.l2_normalise(rng.normal(size=(60, 32)))
    for alg in ("pca", "pca_nc", "tsne", "umap", "isomap"):
        Y1, _ = P.project(X, alg)
        Y2, _ = P.project(X, alg)
        assert Y1.shape == (60, 3)
        np.testing.assert_allclose(Y1, Y2, err_msg=alg)


def test_isomap_keeps_only_giant_component():
    """Tenenbaum et al. (2000, note 19): disconnected points are dropped, not bridged."""
    from ensy import experiment as E

    rng = np.random.default_rng(3)
    big = rng.normal(size=(60, 8)) * 0.05 + np.eye(8)[0]
    small = rng.normal(size=(30, 8)) * 0.05 + np.eye(8)[1]  # 30 > k = 15: no bridging edge
    X = M.l2_normalise(np.vstack([big, small]))
    res = E.run_config(X, np.arange(90) % 3, "isomap")
    assert res["diagnostics"]["knn_graph_components"] == 2
    assert res["diagnostics"]["removed"] == list(range(60, 90))
    assert res["N_evaluated"] == 60 and res["Y"].shape == (60, 3)
    assert all(c["N"] == 60 for c in res["curves"])


def test_pca_matches_centered_svd():
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 10))
    Y, info = P.project(X, "pca")
    ref = PCA(n_components=3, svd_solver="full").fit_transform(X)
    np.testing.assert_allclose(np.abs(Y), np.abs(ref), atol=1e-8)  # up to column sign
    assert len(info["explained_variance_ratio"]) == 3
