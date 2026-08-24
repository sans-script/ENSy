"""
tests/test_api.py — REST API contract tests for the SMVV FastAPI backend.

Validates response schema, field types, value ranges, and error handling
for all four projection engines. No browser required — pure HTTP.

Usage:
    .venv-tests/Scripts/pytest tests/test_api.py -v
    (requires the backend to be running: npm run dev:backend)
"""

import json
import math
import urllib.error
import urllib.request

import pytest


BACKEND_URL = "http://127.0.0.1:3000"
ENGINES = ["umap", "pca", "tsne", "isomap"]


# ──────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────

def fetch_json(url: str) -> dict:
    """
    Perform a GET request and return the parsed JSON body.
    Skips the test if the server is unreachable.
    """
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.URLError as exc:
        pytest.skip(f"Backend unreachable: {exc}")


# ──────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────

class TestProjectionAPI:
    """Contract tests for GET /api/projection/{engine}."""

    @pytest.mark.parametrize("engine", ENGINES)
    def test_endpoint_returns_data(self, engine: str) -> None:
        """Each engine must return a non-null JSON body."""
        data = fetch_json(f"{BACKEND_URL}/api/projection/{engine}")
        assert data is not None

    @pytest.mark.parametrize("engine", ENGINES)
    def test_top_level_schema(self, engine: str) -> None:
        """Response must contain the three top-level keys: status, meta, nodes."""
        data = fetch_json(f"{BACKEND_URL}/api/projection/{engine}")
        assert "status" in data, "Missing field: 'status'"
        assert "meta"   in data, "Missing field: 'meta'"
        assert "nodes"  in data, "Missing field: 'nodes'"

    @pytest.mark.parametrize("engine", ENGINES)
    def test_status_is_valid(self, engine: str) -> None:
        """status must be either 'STABLE' or 'LOADING'."""
        data = fetch_json(f"{BACKEND_URL}/api/projection/{engine}")
        assert data["status"] in ("STABLE", "LOADING"), (
            f"Unexpected status value: {data['status']!r}"
        )

    @pytest.mark.parametrize("engine", ENGINES)
    def test_meta_required_fields(self, engine: str) -> None:
        """meta must include model, embedding_dim, and total_tokens."""
        meta = fetch_json(f"{BACKEND_URL}/api/projection/{engine}")["meta"]
        assert "model"         in meta, "Missing meta field: 'model'"
        assert "embedding_dim" in meta, "Missing meta field: 'embedding_dim'"
        assert "total_tokens"  in meta, "Missing meta field: 'total_tokens'"

    @pytest.mark.parametrize("engine", ENGINES)
    def test_embedding_dim_is_896(self, engine: str) -> None:
        """Qwen 2.5 0.5B embedding dimension must be 896."""
        meta = fetch_json(f"{BACKEND_URL}/api/projection/{engine}")["meta"]
        assert meta["embedding_dim"] == 896, (
            f"Expected embedding_dim=896, got {meta['embedding_dim']}"
        )

    @pytest.mark.parametrize("engine", ENGINES)
    def test_nodes_list_not_empty(self, engine: str) -> None:
        """At least one node must be returned."""
        nodes = fetch_json(f"{BACKEND_URL}/api/projection/{engine}")["nodes"]
        assert len(nodes) > 0, "Node list is empty"

    @pytest.mark.parametrize("engine", ENGINES)
    def test_node_required_fields(self, engine: str) -> None:
        """Every node must have the required keys and a complete coords object."""
        nodes = fetch_json(f"{BACKEND_URL}/api/projection/{engine}")["nodes"]
        required_node   = {"label", "token_id", "cluster", "color", "coords", "top5_neighbors"}
        required_coords = {"x2d", "y2d", "x3d", "y3d", "z3d"}

        for node in nodes:
            missing_node = required_node - node.keys()
            assert not missing_node, (
                f"Node '{node.get('label')}' is missing fields: {missing_node}"
            )
            missing_coords = required_coords - node["coords"].keys()
            assert not missing_coords, (
                f"Node '{node['label']}' coords missing: {missing_coords}"
            )

    @pytest.mark.parametrize("engine", ENGINES)
    def test_3d_coordinates_are_finite(self, engine: str) -> None:
        """3D coordinates must not be NaN or Inf."""
        nodes = fetch_json(f"{BACKEND_URL}/api/projection/{engine}")["nodes"]
        for node in nodes:
            c = node["coords"]
            for axis in ("x3d", "y3d", "z3d"):
                assert math.isfinite(c[axis]), (
                    f"Node '{node['label']}': {axis} is not finite ({c[axis]})"
                )

    @pytest.mark.parametrize("engine", ENGINES)
    def test_neighbors_count_at_most_five(self, engine: str) -> None:
        """Each node must have at most 5 nearest neighbors."""
        nodes = fetch_json(f"{BACKEND_URL}/api/projection/{engine}")["nodes"]
        for node in nodes:
            count = len(node["top5_neighbors"])
            assert count <= 5, (
                f"Node '{node['label']}' has {count} neighbors (max allowed: 5)"
            )

    @pytest.mark.parametrize("engine", ENGINES)
    def test_cosine_similarity_in_valid_range(self, engine: str) -> None:
        """Cosine similarity values must lie within [-1.0, 1.0]."""
        nodes = fetch_json(f"{BACKEND_URL}/api/projection/{engine}")["nodes"]
        for node in nodes:
            for nb in node["top5_neighbors"]:
                sim = nb["cosine_similarity"]
                assert -1.0 <= sim <= 1.0, (
                    f"Similarity out of range for '{node['label']}' → "
                    f"'{nb['label']}': {sim}"
                )

    def test_invalid_engine_returns_http_error(self) -> None:
        """An unrecognised engine name must not return HTTP 200."""
        try:
            with urllib.request.urlopen(
                f"{BACKEND_URL}/api/projection/invalid_engine", timeout=10
            ):
                pytest.fail("Invalid engine returned HTTP 200 unexpectedly")
        except urllib.error.HTTPError as exc:
            assert exc.code == 400, (
                f"Expected HTTP 400 for invalid engine, got {exc.code}"
            )
        except urllib.error.URLError:
            pytest.skip("Backend unreachable")
