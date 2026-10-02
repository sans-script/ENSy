"""Co-ranking framework (LEE; VERLEYSEN, 2008, 2009).

Conventions (plan v2.1.2, Section 9):

* ``rho[i, j]`` is the rank of j w.r.t. i in the high-dimensional space and
  ``r[i, j]`` the rank in the projection. Reflexive ranks are 0 and ties are
  broken by index, as in Lee & Verleysen (2008, Sec. 2), so non-reflexive ranks
  are a permutation of 1..N-1 in every row.
* ``Q[k-1, l-1] = |{(i, j): rho_ij = k and r_ij = l}|`` -- rows are
  high-dimensional ranks, columns are low-dimensional ranks.
* LT_K (k > l, both <= K) holds mild K-intrusions, UT_K (k < l) mild
  K-extrusions, LL_K hard K-intrusions and UR_K hard K-extrusions.
* ``B_NX(K) = U_X(K) - U_N(K)`` with U_N over LT_K and U_X over UT_K, so
  that B_NX > 0 marks an intrusive embedding. This sign was validated
  empirically against B_TC = M_C - M_T (tests/test_coranking.py).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def rank_matrix(D: np.ndarray) -> np.ndarray:
    """Rank matrix of a square distance matrix (self = 0, ties broken by index)."""
    D = np.asarray(D, dtype=np.float64)
    N = D.shape[0]
    D = D.copy()
    np.fill_diagonal(D, -np.inf)
    order = np.argsort(D, axis=1, kind="stable")
    R = np.empty((N, N), dtype=np.int64)
    R[np.arange(N)[:, None], order] = np.arange(N)[None, :]
    return R


def ranks_from_points(X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    return rank_matrix(cdist(X, X, metric=metric))


def coranking_matrix(rho: np.ndarray, r: np.ndarray) -> np.ndarray:
    """(N-1)x(N-1) co-ranking matrix Q. O(N^2) after the O(N^2 log N) ranking."""
    N = rho.shape[0]
    mask = ~np.eye(N, dtype=bool)
    k = rho[mask] - 1
    l = r[mask] - 1
    Q = np.zeros((N - 1, N - 1), dtype=np.int64)
    np.add.at(Q, (k, l), 1)
    return Q


def _blocks(Q: np.ndarray, K: int):
    n = Q.shape[0]
    idx = np.arange(1, n + 1)
    k = idx[:, None]
    l = idx[None, :]
    inK_k, inK_l = k <= K, l <= K
    return {
        "D": inK_k & inK_l & (k == l),
        "LT": inK_k & inK_l & (k > l),
        "UT": inK_k & inK_l & (k < l),
        "LL": ~inK_k & inK_l,
        "UR": inK_k & ~inK_l,
    }, k, l


def _g_k(N: int, K: int) -> float:
    if K < N / 2:
        return N * K * (2 * N - 3 * K - 1)
    return N * (N - K) * (N - K - 1)


def check_k(N: int, K: int) -> None:
    if not 1 <= K <= N - 2:
        raise ValueError(f"K must satisfy 1 <= K <= N-2 (N={N}, K={K})")


def metrics_from_q(Q: np.ndarray, K: int) -> dict:
    """All global co-ranking metrics for one neighbourhood size K."""
    N = Q.shape[0] + 1
    check_k(N, K)
    b, k, l = _blocks(Q, K)
    s = {name: int(Q[m].sum()) for name, m in b.items()}
    G = _g_k(N, K)
    m_t = 1.0 - 2.0 / G * float(((k - K) * Q)[b["LL"]].sum())
    m_c = 1.0 - 2.0 / G * float(((l - K) * Q)[b["UR"]].sum())
    u_p = s["D"] / (K * N)
    u_n = s["LT"] / (K * N)  # mild K-intrusions
    u_x = s["UT"] / (K * N)  # mild K-extrusions
    q_nx = u_p + u_n + u_x
    lcmc = q_nx - K / (N - 1)
    r_nx = ((N - 1) * q_nx - K) / (N - 1 - K)
    return {
        "K": K,
        "N": N,
        "trustworthiness": m_t,
        "continuity": m_c,
        "U_P": u_p,
        "U_N": u_n,
        "U_X": u_x,
        "Q_NX": q_nx,
        "B_NX": u_x - u_n,
        "LCMC": lcmc,
        "R_NX": r_nx,
        "B_TC": m_c - m_t,
        "hard_intrusions": s["LL"],
        "hard_extrusions": s["UR"],
    }


def local_b_nx(rho: np.ndarray, r: np.ndarray, K: int) -> dict:
    """Per-point decomposition B_NX,i(K) = U_X,i(K) - U_N,i(K) (plan Sec. 9.4).

    Uses the per-row co-ranking counts q_kl^(i) directly, so
    mean_i B_NX,i(K) == B_NX(K) holds exactly. |B_NX,i| <= (K-1)/K.
    """
    N = rho.shape[0]
    check_k(N, K)
    off = ~np.eye(N, dtype=bool)
    inside = (rho <= K) & (r <= K) & off
    u_n = (inside & (r < rho)).sum(axis=1) / K  # LT_K: promoted (mild intrusion)
    u_x = (inside & (r > rho)).sum(axis=1) / K  # UT_K: demoted (mild extrusion)
    return {"U_N_i": u_n, "U_X_i": u_x, "B_NX_i": u_x - u_n}


def neighbourhood_probe(rho: np.ndarray, r: np.ndarray, i: int, K: int) -> dict:
    """True/false/missing K-neighbours of point i (plan Sec. 9.7, layer 3)."""
    hi = set(np.flatnonzero((rho[i] >= 1) & (rho[i] <= K)).tolist())
    lo = set(np.flatnonzero((r[i] >= 1) & (r[i] <= K)).tolist())
    return {
        "true": sorted(hi & lo),
        "false": sorted(lo - hi),
        "missing": sorted(hi - lo),
    }


def evaluate(X_high: np.ndarray, Y_low: np.ndarray, Ks, high_metric: str = "cosine") -> dict:
    """Global curves over K plus per-point B_NX,i for every K."""
    rho = ranks_from_points(X_high, metric=high_metric)
    r = ranks_from_points(Y_low, metric="euclidean")
    Q = coranking_matrix(rho, r)
    curves = [metrics_from_q(Q, K) for K in Ks]
    local = {K: local_b_nx(rho, r, K)["B_NX_i"] for K in Ks}
    return {"curves": curves, "local_b_nx": local, "rho": rho, "r": r}
