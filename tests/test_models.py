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


def test_pca_matches_centered_svd():
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 10))
    Y, info = P.project(X, "pca")
    ref = PCA(n_components=3, svd_solver="full").fit_transform(X)
    np.testing.assert_allclose(np.abs(Y), np.abs(ref), atol=1e-8)  # up to column sign
    assert len(info["explained_variance_ratio"]) == 3
