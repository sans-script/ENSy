"""Lexical dataset derived from the Penn Treebank (plan Section 5, steps 1-10).

Source: the free PTB sample distributed with NLTK (WSJ section: 199 files,
3,914 sentences, 100,676 tokens). The full PTB is LDC-licensed; if it becomes
available, point ``ptb_tagged_words`` at it and rebuild -- everything else is
unchanged and the dataset hash will reflect the new source.

Run:  python -m ensy.dataset [--models qwen,gemma2,gpt2,bert] [--out data/dataset.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from . import models as M

CATEGORIES = ("content_noun", "content_verb", "function_word")

# PTB tag -> reference partition. Proper nouns (NNP/NNPS) are excluded because
# the lower-casing needed for BERT-uncased would merge them with common words.
TAG_TO_CATEGORY = {
    "NN": "content_noun", "NNS": "content_noun",
    "VB": "content_verb", "VBD": "content_verb", "VBG": "content_verb",
    "VBN": "content_verb", "VBP": "content_verb", "VBZ": "content_verb",
    "DT": "function_word", "PDT": "function_word", "IN": "function_word",
    "CC": "function_word", "TO": "function_word", "MD": "function_word",
    "PRP": "function_word", "PRP$": "function_word", "WDT": "function_word",
    "WP": "function_word", "WP$": "function_word", "WRB": "function_word",
    "EX": "function_word", "RP": "function_word",
}

# PTB tags auxiliaries with VB*; linguistically they are function words, so
# their verb occurrences are counted as function_word (fixed closed list).
AUXILIARIES = frozenset(
    "be am is are was were been being have has had do does did".split()
)

WORD_RE = re.compile(r"^[a-z]+$")
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "dataset.json"


def ptb_tagged_words():
    import nltk

    nltk.download("treebank", quiet=True)
    from nltk.corpus import treebank

    return treebank.tagged_words()


def occurrence_category(word: str, tag: str) -> str:
    cat = TAG_TO_CATEGORY.get(tag, "other")
    if cat == "content_verb" and word in AUXILIARIES:
        return "function_word"
    return cat


def lexical_types(tagged) -> tuple[dict[str, Counter], dict[str, str]]:
    """Steps 4-5: occurrences -> lexical types with per-category counts.

    Two fixed artefact rules remove forms that are not free lexical words:
    clitic stems split off by PTB tokenisation (``ca``/``wo`` in ``ca n't``,
    every occurrence precedes ``n't``) and name-internal function forms
    (``de``, ``la``, ``del`` from foreign proper names). The second rule only
    applies to forms whose majority category is function_word, and flags them
    when they occur at least as often as NNP as otherwise, or when most of their
    occurrences sit between two NNP tokens.
    """
    types: dict[str, Counter] = defaultdict(Counter)
    n_total, n_clitic, n_inside_name, n_as_nnp = Counter(), Counter(), Counter(), Counter()
    tagged = list(tagged)
    for idx, (token, tag) in enumerate(tagged):
        if tag in ("NNP", "NNPS"):
            n_as_nnp[token.lower()] += 1
        if tag == "-NONE-" or tag in ("NNP", "NNPS"):
            continue
        w = token.lower()
        if not WORD_RE.match(w):
            continue
        types[w][occurrence_category(w, tag)] += 1
        n_total[w] += 1
        nxt = tagged[idx + 1] if idx + 1 < len(tagged) else ("", "")
        prv = tagged[idx - 1] if idx > 0 else ("", "")
        if nxt[0] == "n't":
            n_clitic[w] += 1
        if prv[1].startswith("NNP") and nxt[1].startswith("NNP"):
            n_inside_name[w] += 1
    artefacts = {}
    for w, n in n_total.items():
        if n_clitic[w] == n:
            artefacts[w] = "clitic_stem"
        elif types[w].most_common(1)[0][0] == "function_word" and (
            n_as_nnp[w] >= n or n_inside_name[w] / n > 0.5
        ):
            artefacts[w] = "name_internal"
    return {w: c for w, c in types.items() if w not in artefacts}, artefacts


def disambiguate(types: dict[str, Counter]) -> tuple[dict[str, dict], dict[str, str]]:
    """Step 6: majority category over all occurrences; ties or a majority of
    out-of-partition tags (adjectives, adverbs, ...) exclude the form."""
    kept, excluded = {}, {}
    for w, counts in types.items():
        (top, n1), *rest = counts.most_common()
        if rest and rest[0][1] == n1:
            excluded[w] = "tie"
        elif top == "other":
            excluded[w] = "majority_outside_partition"
        else:
            kept[w] = {
                "category": top,
                "frequency": sum(counts.values()),
                "category_share": n1 / sum(counts.values()),
            }
    return kept, excluded


def single_token_filter(words: dict[str, dict], model_keys, revisions: dict | None = None) -> tuple[dict, dict]:
    """Steps 1-3 and 7: keep forms that are exactly one valid token in every model."""
    revisions = revisions or {}
    tokenizers = {k: M.load_tokenizer(k, revisions.get(k)) for k in model_keys}
    kept, dropped = {}, {}
    for w, meta in words.items():
        ids = {}
        for k, tok in tokenizers.items():
            tid = M.single_token_id(tok, k, w)
            if tid is None:
                dropped[w] = k
                break
            ids[k] = tid
        else:
            kept[w] = {**meta, "token_ids": ids}
    return kept, dropped


def balance(words: dict[str, dict]) -> tuple[list[dict], dict]:
    """Step 9: equal count per category = size of the smallest category; within a
    category the most frequent forms are kept (ties broken alphabetically)."""
    by_cat = {c: [] for c in CATEGORIES}
    for w, meta in words.items():
        by_cat[meta["category"]].append((w, meta))
    available = {c: len(v) for c, v in by_cat.items()}
    n = min(available.values())
    items = []
    for c in CATEGORIES:
        ranked = sorted(by_cat[c], key=lambda wm: (-wm[1]["frequency"], wm[0]))[:n]
        items.extend({"word": w, **meta} for w, meta in ranked)
    items.sort(key=lambda it: (CATEGORIES.index(it["category"]), it["word"]))
    return items, available


def dataset_hash(items: list[dict], model_keys) -> str:
    canonical = json.dumps(
        {"models": sorted(model_keys), "items": items}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build(model_keys=tuple(M.MODELS)) -> dict:
    tagged = ptb_tagged_words()
    types, artefacts = lexical_types(tagged)
    words, excluded = disambiguate(types)
    revisions = {k: M.resolved_revision(k) for k in model_keys}  # tokenizer and record share one commit
    single, dropped = single_token_filter(words, model_keys, revisions)
    items, available = balance(single)
    complete = set(model_keys) == set(M.MODELS)
    return {
        "name": "ENSy lexical dataset derived from the Penn Treebank",
        "version": "2.1.2",
        "frozen": complete,
        "note": None if complete else
            f"PROVISIONAL: built without all {len(M.MODELS)} tokenizers; not valid for experiments",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "nltk.corpus.treebank (PTB WSJ sample)",
        "models": {k: {"hf_id": M.MODELS[k].hf_id, "revision": revisions[k],
                       "surface_form": "' '+word" if M.MODELS[k].space_prefix else "word"}
                   for k in model_keys},
        "rules": {
            "normalisation": "lower-case, alphabetic only, proper nouns (NNP/NNPS) excluded",
            "artefacts": "clitic stems (always before n't) excluded; function-word forms "
                         "that are name-internal (NNP count >= other count, or >50% of "
                         "occurrences between two NNP tokens) excluded",
            "tag_map": TAG_TO_CATEGORY,
            "auxiliaries_as_function_words": sorted(AUXILIARIES),
            "disambiguation": "majority category over all PTB occurrences; ties and "
                              "out-of-partition majorities excluded",
            "single_token": "exactly one id, add_special_tokens=False, not unk/special, "
                            "decodes back to the word, in every model",
            "balancing": "n per category = min category size; most frequent forms kept",
        },
        "counts": {
            "ptb_tokens": len(tagged),
            "lexical_types": len(types) + len(artefacts),
            "excluded_artefacts": artefacts,
            "after_disambiguation": len(words),
            "excluded_tie": sum(v == "tie" for v in excluded.values()),
            "excluded_outside_partition": sum(v != "tie" for v in excluded.values()),
            "dropped_multi_token": {k: sum(v == k for v in dropped.values()) for k in model_keys},
            "single_token_per_category": available,
            "N": len(items),
        },
        "N": len(items),
        "dataset_hash": dataset_hash(items, model_keys),
        "items": items,
    }


def load(path: Path = DEFAULT_OUT) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = dataset_hash(data["items"], data["models"].keys())
    if expected != data["dataset_hash"]:
        raise ValueError(f"dataset hash mismatch in {path}")
    return data


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", default=",".join(M.MODELS))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    keys = tuple(args.models.split(","))
    ds = build(keys)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(ds, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(ds["counts"], indent=1))
    print(f"N={ds['N']} hash={ds['dataset_hash'][:12]} frozen={ds['frozen']} -> {args.out}")


if __name__ == "__main__":
    main()
