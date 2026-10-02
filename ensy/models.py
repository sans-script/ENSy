"""Model abstraction and static input-embedding extraction (plan Section 4).

Only the input embedding matrix is read -- no forward pass. The tensor is read
straight from the checkpoint's safetensors shard, which is the same tensor that
``model.get_input_embeddings().weight`` returns (verified for Qwen/BERT in
tests/test_models.py) without materialising the whole model in RAM.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

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
    "gemma": ModelSpec(
        "gemma", os.environ.get("ENSY_GEMMA_ID", "google/gemma-2b"), "decoder", True,
        ("embed_tokens.weight",), gated=True,
    ),
    "bert": ModelSpec(
        "bert", "bert-base-uncased", "encoder", False,
        ("embeddings.word_embeddings.weight",),
    ),
}


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def load_tokenizer(key: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODELS[key].hf_id, token=hf_token())


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


def _find_embedding_tensor(key: str):
    """(local shard path, tensor name) of the input embedding matrix."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    spec = MODELS[key]
    tok = hf_token()
    try:
        index_path = hf_hub_download(spec.hf_id, "model.safetensors.index.json", token=tok)
        weight_map = json.load(open(index_path, encoding="utf-8"))["weight_map"]
        candidates = weight_map
    except EntryNotFoundError:
        shard = hf_hub_download(spec.hf_id, "model.safetensors", token=tok)
        from safetensors import safe_open

        with safe_open(shard, "np") as f:
            candidates = {name: "model.safetensors" for name in f.keys()}
    for name, fname in candidates.items():
        if any(name.endswith(s) for s in spec.embedding_key_suffixes) and "position" not in name:
            return hf_hub_download(spec.hf_id, fname, token=tok), name
    raise KeyError(f"no input-embedding tensor found for {spec.hf_id}")


def embedding_matrix(key: str) -> np.ndarray:
    """Full input-embedding matrix as float32 (rows = token ids)."""
    from safetensors import safe_open

    path, name = _find_embedding_tensor(key)
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

    def __init__(self):
        self._store: dict[tuple, np.ndarray] = {}

    def get(self, key: str, dataset: dict, normalise: bool = True) -> np.ndarray:
        ck = (MODELS[key].hf_id, dataset["dataset_hash"], "static_input_embedding",
              "l2" if normalise else "raw")
        if ck not in self._store:
            W = embedding_matrix(key)
            ids = [item["token_ids"][key] for item in dataset["items"]]
            X = W[ids]
            self._store[ck] = l2_normalise(X) if normalise else X
        return self._store[ck]

    def clear(self):
        self._store.clear()
