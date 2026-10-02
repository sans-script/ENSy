"""Model abstraction and static input-embedding extraction (plan Section 4).

Only the input embedding matrix is read -- no forward pass. The tensor is read
straight from the checkpoint's safetensors file -- by default only the needed
rows, via HTTP Range requests -- and is the same tensor that
``model.get_input_embeddings().weight`` returns (verified for Qwen/BERT in
tests/test_models.py) without materialising the whole model in RAM.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    family: str  # "decoder" | "encoder"
    # Surface form looked up in the tokenizer. Byte-level BPE (Qwen) and
    # SentencePiece (Gemma) distinguish "run" (word-internal / text-initial)
    # from " run" (word-initial in running text); the latter is the lexical
    # word token. WordPiece (BERT) has no such distinction for whole words.
    space_prefix: bool
    embedding_key_suffixes: tuple[str, ...]
    gated: bool = False


MODELS: dict[str, ModelSpec] = {
    "qwen": ModelSpec(
        "qwen", "Qwen/Qwen2.5-0.5B-Instruct", "decoder", True,
        ("embed_tokens.weight",),
    ),
    # GPT-2 (OpenAI, 2019): an earlier generation with a different corpus
    # (WebText) and training recipe; byte-level BPE like Qwen.
    "gpt2": ModelSpec(
        "gpt2", "openai-community/gpt2", "decoder", True,
        ("wte.weight",),
    ),
    "gemma2": ModelSpec(
        "gemma2", "google/gemma-2-2b", "decoder", True,
        ("embed_tokens.weight",), gated=True,
    ),
    "bert": ModelSpec(
        "bert", "bert-base-uncased", "encoder", False,
        ("embeddings.word_embeddings.weight",),
    ),
}


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def load_tokenizer(key: str, revision: str | None = None):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODELS[key].hf_id, revision=revision, token=hf_token())


def resolved_revision(key: str) -> str | None:
    """Commit SHA of the checkpoint actually used (reproducibility record)."""
    try:
        from huggingface_hub import model_info

        return model_info(MODELS[key].hf_id, token=hf_token()).sha
    except Exception:
        return None


def surface_form(key: str, word: str) -> str:
    return (" " + word) if MODELS[key].space_prefix else word


def single_token_id(tokenizer, key: str, word: str, n_rows: int | None = None) -> int | None:
    """Return the token id if ``word`` is exactly one valid token, else None.

    Inclusion criterion (plan Sec. 4): exactly one id without special tokens,
    the id is a real embedding row, it is not the unknown token, and it decodes
    back to the word (guards against normalisation collapsing distinct forms).
    """
    ids = tokenizer.encode(surface_form(key, word), add_special_tokens=False)
    if len(ids) != 1:
        return None
    tid = int(ids[0])
    if tokenizer.unk_token_id is not None and tid == tokenizer.unk_token_id:
        return None
    if tid in set(tokenizer.all_special_ids):
        return None
    if n_rows is not None and not 0 <= tid < n_rows:
        return None
    if tokenizer.decode([tid]).strip().lower() != word:
        return None
    return tid


def _embedding_location(key: str, revision: str | None = None) -> tuple[str, str]:
    """(safetensors file name, tensor name) of the input embedding matrix."""
    import requests
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    spec = MODELS[key]
    try:
        index_path = hf_hub_download(spec.hf_id, "model.safetensors.index.json",
                                     revision=revision, token=hf_token())
        candidates = json.load(open(index_path, encoding="utf-8"))["weight_map"]
    except EntryNotFoundError:
        header = _safetensors_header(key, "model.safetensors", revision, requests.Session())
        candidates = {name: "model.safetensors" for name in header if name != "__metadata__"}
    for name, fname in candidates.items():
        if any(name.endswith(s) for s in spec.embedding_key_suffixes) and "position" not in name:
            return fname, name
    raise KeyError(f"no input-embedding tensor found for {spec.hf_id}")


def _ranged_get(session, url: str, start: int, end: int) -> bytes:
    """Inclusive byte range; the Hub redirects to a CDN that honours Range."""
    headers = {"Range": f"bytes={start}-{end}"}
    token = hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(4):
        try:
            r = session.get(url, headers=headers, timeout=60)
            if r.status_code == 206 and len(r.content) == end - start + 1:
                return r.content
            r.raise_for_status()
            raise IOError(f"server ignored Range (HTTP {r.status_code})")
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def _safetensors_header(key: str, fname: str, revision: str | None, session) -> dict:
    from huggingface_hub import hf_hub_url

    url = hf_hub_url(MODELS[key].hf_id, fname, revision=revision)
    n = struct.unpack("<Q", _ranged_get(session, url, 0, 7))[0]
    header = json.loads(_ranged_get(session, url, 8, 8 + n - 1))
    header["__data_start__"] = 8 + n
    header["__url__"] = url
    return header


_DTYPES = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16}


def _decode(raw: bytes, dtype: str) -> np.ndarray:
    a = np.frombuffer(raw, dtype=_DTYPES[dtype])
    if dtype == "BF16":  # bfloat16 is the upper half of a float32: exact widening
        return (a.astype(np.uint32) << 16).view(np.float32)
    return a.astype(np.float32)


def embedding_rows(key: str, ids, revision: str | None = None, workers: int = 8) -> np.ndarray:
    """Selected rows of the input-embedding matrix, fetched with HTTP Range requests.

    Only the safetensors header and the requested rows are transferred (a few MB
    instead of a multi-GB checkpoint); the values are bit-identical to
    ``embedding_matrix(key)[ids]`` (tests/test_models.py).
    """
    import requests
    from concurrent.futures import ThreadPoolExecutor

    ids = [int(i) for i in ids]
    fname, name = _embedding_location(key, revision)
    session = requests.Session()
    header = _safetensors_header(key, fname, revision, session)
    meta = header[name]
    rows, dim = meta["shape"]
    if meta["dtype"] not in _DTYPES:
        raise TypeError(f"unsupported dtype {meta['dtype']}")
    itemsize = np.dtype(_DTYPES[meta["dtype"]]).itemsize
    row_bytes = dim * itemsize
    base = header["__data_start__"] + meta["data_offsets"][0]
    if any(not 0 <= i < rows for i in ids):
        raise IndexError("token id outside the embedding matrix")

    # Merge sorted ids into contiguous runs -> one request per run
    runs, uniq = [], sorted(set(ids))
    for i in uniq:
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])

    def fetch(run):
        a, b = run
        raw = _ranged_get(session, header["__url__"], base + a * row_bytes, base + (b + 1) * row_bytes - 1)
        return a, _decode(raw, meta["dtype"]).reshape(b - a + 1, dim)

    found = {}
    with ThreadPoolExecutor(workers) as pool:
        for a, block in pool.map(fetch, runs):
            for off, row in enumerate(block):
                found[a + off] = row
    return np.stack([found[i] for i in ids]).astype(np.float32, copy=False)


def embedding_matrix(key: str, revision: str | None = None) -> np.ndarray:
    """Full input-embedding matrix as float32 (reference path: downloads the shard)."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    fname, name = _embedding_location(key, revision)
    path = hf_hub_download(MODELS[key].hf_id, fname, revision=revision, token=hf_token())
    with safe_open(path, "pt") as f:  # "pt" handles bf16 checkpoints
        W = f.get_tensor(name)
    return W.float().numpy().astype(np.float32, copy=False)


def l2_normalise(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero-norm embedding row")
    return (X / norms).astype(np.float32)


class EmbeddingCache:
    """In-memory cache keyed as in plan Sec. 10:
    (model_id, dataset_hash, extraction_config, preprocessing_config)."""

    def __init__(self, disk_dir: str | os.PathLike | None = None):
        """disk_dir: optional folder that keeps the extracted raw rows between runs
        (one .npy per model revision and token-id list); None keeps memory only."""
        self._store: dict[tuple, np.ndarray] = {}
        self._disk = Path(disk_dir) if disk_dir else None

    def _disk_path(self, key: str, ids: list, revision: str | None) -> Path | None:
        if self._disk is None:
            return None
        digest = hashlib.sha256(
            json.dumps([MODELS[key].hf_id, revision, ids]).encode()).hexdigest()[:16]
        return self._disk / f"{key}_{digest}.npy"

    def get(self, key: str, dataset: dict, normalise: bool = True) -> np.ndarray:
        ck = (MODELS[key].hf_id, dataset["dataset_hash"], "static_input_embedding",
              "l2" if normalise else "raw")
        if ck not in self._store:
            ids = [item["token_ids"][key] for item in dataset["items"]]
            revision = dataset["models"][key].get("revision")
            path = self._disk_path(key, ids, revision)
            if path is not None and path.exists():
                X = np.load(path)
            else:
                if os.environ.get("ENSY_FULL_DOWNLOAD"):
                    X = embedding_matrix(key, revision)[ids]
                else:
                    X = embedding_rows(key, ids, revision)
                if path is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, X)
            self._store[ck] = l2_normalise(X) if normalise else X
        return self._store[ck]

    def clear(self):
        self._store.clear()
