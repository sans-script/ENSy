"""Validation suite for the co-ranking metrics (plan v2.1.2, Sections 9.0-9.4).

Synthetic cases are built by controlled rank permutation, not by moving points,
so each case produces exactly the co-ranking entries it claims to.
"""
import numpy as np
import pytest
from sklearn.manifold import trustworthiness as sk_trustworthiness

from ensy.coranking import (
    coranking_matrix,
    local_b_nx,
    metrics_from_q,
    neighbourhood_probe,
    rank_matrix,
    ranks_from_points,
)
from scipy.spatial.distance import cdist

KS = (5, 10, 15, 20)


@pytest.fixture(scope="module")
def cloud():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 16))
    Y = X[:, :3] + 0.3 * rng.normal(size=(120, 3))
    return X, Y


def _move_rank(r_row: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Move the item at rank ``src`` to rank ``dst``; items in between shift by one."""
    out = r_row.copy()
    j = int(np.flatnonzero(r_row == src)[0])
    if dst < src:
        out[(r_row >= dst) & (r_row < src)] += 1
    else:
        out[(r_row > src) & (r_row <= dst)] -= 1
    out[j] = dst
    return out


def _base_ranks(N=60, seed=0):
    X = np.random.default_rng(seed).normal(size=(N, 8))
    return ranks_from_points(X, "euclidean")


# ── structural properties ────────────────────────────────────────────────

def test_rank_matrix_rows_are_permutations(cloud):
    R = ranks_from_points(cloud[0])
    N = len(R)
    assert np.all(np.diag(R) == 0)
    assert np.all(np.sort(R, axis=1) == np.arange(N))


def test_rank_ties_broken_by_index():
    D = np.zeros((4, 4))  # all ties
    R = rank_matrix(D)
    assert R[0].tolist() == [0, 1, 2, 3]
    assert R[2].tolist() == [1, 2, 0, 3]


def test_q_margins_and_hard_balance(cloud):
    X, Y = cloud
    Q = coranking_matrix(ranks_from_points(X), ranks_from_points(Y))
    N = len(X)
    assert np.all(Q.sum(axis=0) == N) and np.all(Q.sum(axis=1) == N)
    for K in KS:
        m = metrics_from_q(Q, K)
        assert m["hard_intrusions"] == m["hard_extrusions"]  # Lee & Verleysen Eq. (8)


def test_cosine_and_euclidean_ranks_agree_after_l2(cloud):
    X = cloud[0] / np.linalg.norm(cloud[0], axis=1, keepdims=True)
    assert np.array_equal(ranks_from_points(X, "cosine"), ranks_from_points(X, "euclidean"))


# ── trustworthiness / continuity against an independent implementation ──

@pytest.mark.parametrize("K", KS)
def test_trustworthiness_continuity_match_sklearn(cloud, K):
    X, Y = cloud
    m = metrics_from_q(coranking_matrix(ranks_from_points(X), ranks_from_points(Y)), K)
    assert m["trustworthiness"] == pytest.approx(sk_trustworthiness(X, Y, n_neighbors=K), abs=1e-12)
    assert m["continuity"] == pytest.approx(sk_trustworthiness(Y, X, n_neighbors=K), abs=1e-12)


def test_q_nx_lcmc_relation(cloud):
    X, Y = cloud
    Q = coranking_matrix(ranks_from_points(X), ranks_from_points(Y))
    N = len(X)
    for K in KS:
        m = metrics_from_q(Q, K)
        assert m["Q_NX"] == pytest.approx(m["U_P"] + m["U_N"] + m["U_X"])
        assert m["LCMC"] == pytest.approx(m["Q_NX"] - K / (N - 1))
        # direct definition: mean fraction of shared K-neighbours
        rho, r = ranks_from_points(X), ranks_from_points(Y)
        shared = (((rho >= 1) & (rho <= K)) & ((r >= 1) & (r <= K))).sum()
        assert m["Q_NX"] == pytest.approx(shared / (K * N))


# ── Case A: identity ─────────────────────────────────────────────────────

def test_case_a_identity():
    rho = _base_ranks()
    Q = coranking_matrix(rho, rho.copy())
    for K in KS:
        m = metrics_from_q(Q, K)
        assert m["B_NX"] == 0 and m["Q_NX"] == 1
        assert m["trustworthiness"] == 1 and m["continuity"] == 1
        assert np.all(local_b_nx(rho, rho, K)["B_NX_i"] == 0)


# ── element-level arithmetic: which triangle feeds which term ────────────

def test_single_swap_is_one_mild_intrusion_and_one_mild_extrusion():
    rho = _base_ranks()
    r = rho.copy()
    i, K = 3, 10
    a, b = int(np.flatnonzero(rho[i] == 2)[0]), int(np.flatnonzero(rho[i] == 6)[0])
    r[i, a], r[i, b] = 6, 2
    m = metrics_from_q(coranking_matrix(rho, r), K)
    N = len(rho)
    assert m["U_N"] == pytest.approx(1 / (K * N))  # b promoted 6 -> 2 : LT_K
    assert m["U_X"] == pytest.approx(1 / (K * N))  # a demoted 2 -> 6 : UT_K
    assert m["B_NX"] == 0


# ── Case B / C (corrected): intrusive vs extrusive rank events ───────────
# Promoting one item from rank m to rank 1 inside the K-neighbourhood yields
# 1 mild intrusion (LT) and m-1 shifted items as mild extrusions (UT), so
# B_NX,i = (m-2)/K. With B = U_X - U_N the event reads as intrusive (B > 0).
# The plan's original Cases B/C asserted the opposite signs; see the review.

@pytest.mark.parametrize("m", [3, 5, 9])
def test_case_b_promotion_is_intrusive(m):
    rho = _base_ranks()
    r = rho.copy()
    i, K = 7, 10
    r[i] = _move_rank(rho[i], src=m, dst=1)
    m_ = metrics_from_q(coranking_matrix(rho, r), K)
    loc = local_b_nx(rho, r, K)["B_NX_i"]
    assert m_["B_NX"] > 0
    assert loc[i] == pytest.approx((m - 1 - 1) / K)
    assert np.all(np.delete(loc, i) == 0)


@pytest.mark.parametrize("m", [3, 5, 9])
def test_case_c_demotion_is_extrusive(m):
    rho = _base_ranks()
    r = rho.copy()
    i, K = 7, 10
    r[i] = _move_rank(rho[i], src=1, dst=m)
    m_ = metrics_from_q(coranking_matrix(rho, r), K)
    loc = local_b_nx(rho, r, K)["B_NX_i"]
    assert m_["B_NX"] < 0
    assert loc[i] == pytest.approx(-(m - 1 - 1) / K)
    assert np.all(np.delete(loc, i) == 0)


def test_hard_intrusion_shows_up_as_mild_extrusions():
    """A far point jumping to rank 1 shifts the true neighbours -> B_NX,i > 0."""
    rho = _base_ranks()
    r = rho.copy()
    i, K = 0, 10
    r[i] = _move_rank(rho[i], src=40, dst=1)
    loc = local_b_nx(rho, r, K)
    assert loc["U_N_i"][i] == 0
    assert loc["U_X_i"][i] == pytest.approx((K - 1) / K)


# ── sign anchor against B_TC = M_C - M_T (Lee & Verleysen 2008, Sec. 4) ──

def _sphere(n=600, seed=0):
    P = np.random.default_rng(seed).normal(size=(n, 3))
    return P / np.linalg.norm(P, axis=1, keepdims=True)


@pytest.mark.parametrize("K", [10, 30])
def test_sign_anchor_crushed_sphere_is_intrusive(K):
    P = _sphere()
    Y = P[:, :2]  # orthographic projection: front/back hemispheres collapse
    m = metrics_from_q(coranking_matrix(ranks_from_points(P), ranks_from_points(Y)), K)
    assert m["B_TC"] > 0 and m["B_NX"] > 0


@pytest.mark.parametrize("K", [10, 30])
def test_sign_anchor_torn_sphere_is_extrusive(K):
    P = _sphere()
    Y = np.c_[np.arctan2(P[:, 1], P[:, 0]), np.arccos(P[:, 2])]  # unrolled: seam tear
    m = metrics_from_q(coranking_matrix(ranks_from_points(P), ranks_from_points(Y)), K)
    assert m["B_TC"] < 0 and m["B_NX"] < 0


# ── local decomposition consistency (plan Sec. 9.4) ──────────────────────

@pytest.mark.parametrize("K", KS)
def test_local_mean_equals_global(cloud, K):
    X, Y = cloud
    rho, r = ranks_from_points(X), ranks_from_points(Y)
    g = metrics_from_q(coranking_matrix(rho, r), K)["B_NX"]
    loc = local_b_nx(rho, r, K)["B_NX_i"]
    assert abs(g - loc.mean()) < 1e-12
    assert np.all(np.abs(loc) <= (K - 1) / K)


def test_probe_partitions_neighbourhoods(cloud):
    X, Y = cloud
    rho, r = ranks_from_points(X), ranks_from_points(Y)
    K = 10
    p = neighbourhood_probe(rho, r, 5, K)
    assert len(p["true"]) + len(p["false"]) == K
    assert len(p["false"]) == len(p["missing"])  # the identity of plan Sec. 9.4


def test_k_bounds():
    rho = _base_ranks(N=12)
    Q = coranking_matrix(rho, rho)
    with pytest.raises(ValueError):
        metrics_from_q(Q, 11)
