"""Round trip with the TensorFlow Embedding Projector formats (ensy/projector.py)."""
import io
import json
import zipfile

import numpy as np
import pytest

from ensy import projector as TP


def test_vectors_round_trip_is_exact():
    X = np.random.default_rng(0).normal(size=(30, 7))
    Y, labels = TP.parse_vectors_tsv(TP.vectors_tsv(X))
    assert labels is None
    assert np.array_equal(X, Y)  # repr() keeps every float bit


def test_vectors_with_label_column():
    X, labels = TP.parse_vectors_tsv("cat\t1\t2\ndog\t3\t4\n")
    assert labels == ["cat", "dog"] and X.tolist() == [[1, 2], [3, 4]]


def test_vectors_reject_ragged_rows():
    with pytest.raises(ValueError):
        TP.parse_vectors_tsv("1\t2\n3\t4\t5\n")


def test_metadata_header_rule():
    assert TP.parse_metadata_tsv("a\nb\n") == (["label"], [["a"], ["b"]])
    cols, rows = TP.parse_metadata_tsv("word\tcategory\nrun\tverb\n")
    assert cols == ["word", "category"] and rows == [["run", "verb"]]


@pytest.mark.parametrize("algorithm", list(TP.STATE_SLOTS))
def test_state_round_trip_reproduces_coordinates(algorithm):
    n = 12
    Y = np.random.default_rng(1).normal(size=(n, 3))
    kept = np.arange(n)
    if algorithm == "isomap":  # a subset, as when Isomap drops points
        kept = np.array([0, 2, 3, 5, 7, 8, 11])
        Y = Y[kept]
    st = TP.make_state("x", algorithm, Y, kept, n, 5)
    back = TP.parse_states(json.dumps([st]), n)[0]
    assert np.array_equal(back["kept"], kept)
    assert np.allclose(back["Y"], Y)


def test_export_zip_contents():
    X = np.eye(4)
    st = TP.make_state("pca", "pca", X[:, :3], np.arange(4), 4, 4)
    z = zipfile.ZipFile(io.BytesIO(TP.export_zip("t", X, ["word"], [["a"], ["b"], ["c"], ["d"]], [st])))
    assert set(z.namelist()) >= {"vectors.tsv", "metadata.tsv", "state.txt", "projector_config.json"}
    cfg = json.loads(z.read("projector_config.json"))
    assert cfg["embeddings"][0]["tensorShape"] == [4, 4]
    assert len(json.loads(z.read("state.txt"))[0]["projections"]) == 4


def test_state_point_count_must_match():
    st = TP.make_state("pca", "pca", np.zeros((3, 3)), np.arange(3), 3, 3)
    with pytest.raises(ValueError):
        TP.parse_states(json.dumps([st]), 4)
