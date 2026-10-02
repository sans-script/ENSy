"""Dataset protocol tests (plan Section 5). Tokenizer tests need the HF cache/network."""
from collections import Counter

import pytest

from ensy import dataset as D
from ensy import models as M


def test_auxiliaries_are_function_words():
    assert D.occurrence_category("is", "VBZ") == "function_word"
    assert D.occurrence_category("runs", "VBZ") == "content_verb"
    assert D.occurrence_category("fast", "RB") == "other"


def test_disambiguation_majority_tie_and_outside():
    words, excluded = D.disambiguate({
        "run": Counter(content_verb=5, content_noun=2),
        "set": Counter(content_verb=3, content_noun=3),
        "fast": Counter(other=4, content_verb=1),
    })
    assert words["run"]["category"] == "content_verb"
    assert excluded == {"set": "tie", "fast": "majority_outside_partition"}


def test_artefact_rules():
    tagged = [("They", "PRP"), ("ca", "MD"), ("n't", "RB"), ("go", "VB"), (".", "."),
              ("Domaine", "NNP"), ("de", "IN"), ("Romanee", "NNP"),
              ("Mr.", "NNP"), ("Smith", "NNP"), ("authorized", "VBD"), ("Acme", "NNP")]
    types, artefacts = D.lexical_types(tagged)
    assert artefacts == {"ca": "clitic_stem", "de": "name_internal"}
    assert "authorized" in types  # content verbs are never treated as name-internal
    assert "smith" not in types   # proper nouns are excluded


def test_balance_is_equal_and_frequency_ranked():
    words = {f"n{i}": {"category": "content_noun", "frequency": i} for i in range(5)}
    words |= {f"v{i}": {"category": "content_verb", "frequency": i} for i in range(3)}
    words |= {f"f{i}": {"category": "function_word", "frequency": i} for i in range(4)}
    items, available = D.balance(words)
    counts = Counter(it["category"] for it in items)
    assert set(counts.values()) == {3}
    assert {it["word"] for it in items if it["category"] == "content_noun"} == {"n4", "n3", "n2"}
    assert available == {"content_noun": 5, "content_verb": 3, "function_word": 4}


def test_hash_is_order_sensitive_and_stable():
    items = [{"word": "a"}, {"word": "b"}]
    h = D.dataset_hash(items, ["qwen", "bert"])
    assert h == D.dataset_hash(items, ["bert", "qwen"])
    assert h != D.dataset_hash(items[::-1], ["qwen", "bert"])


@pytest.mark.network
@pytest.mark.parametrize("key", ["qwen", "bert"])
def test_single_token_criterion(key):
    tok = M.load_tokenizer(key)
    assert M.single_token_id(tok, key, "the") is not None
    assert M.single_token_id(tok, key, "xylophone") is None      # several subwords
    assert M.single_token_id(tok, key, "zzqx") is None


@pytest.mark.network
def test_qwen_uses_word_initial_form():
    tok = M.load_tokenizer("qwen")
    tid = M.single_token_id(tok, "qwen", "run")
    assert tok.convert_ids_to_tokens(tid) == "Ġrun"  # 'Ġrun', not the word-internal 'run'
