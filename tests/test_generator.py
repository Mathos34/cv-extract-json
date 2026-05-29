"""Unit tests for the synthetic CV generator."""
from src.generator import generate_dataset, make_cv


def test_make_cv_is_deterministic_given_seed():
    t1, s1 = make_cv(seed=42)
    t2, s2 = make_cv(seed=42)
    assert t1 == t2
    assert len(s1) == len(s2)
    for a, b in zip(s1, s2, strict=True):
        assert (a.start, a.end, a.label) == (b.start, b.end, b.label)


def test_make_cv_different_seeds_differ():
    t1, _ = make_cv(seed=1)
    t2, _ = make_cv(seed=2)
    assert t1 != t2


def test_all_spans_round_trip_to_source():
    text, spans = make_cv(seed=123)
    for sp in spans:
        sub = text[sp.start: sp.end]
        assert sub != ""
        assert sub == sp.text(text)


def test_spans_have_valid_labels():
    allowed = {"PERSON", "SKILL", "ORG", "DATE", "TITLE", "DEGREE", "SCHOOL"}
    _, spans = make_cv(seed=7)
    for sp in spans:
        assert sp.label in allowed


def test_spans_do_not_overlap_within_a_single_cv():
    text, spans = make_cv(seed=11)
    sorted_spans = sorted(spans, key=lambda s: s.start)
    for prev, nxt in zip(sorted_spans, sorted_spans[1:], strict=False):
        assert prev.end <= nxt.start
    for sp in spans:
        assert 0 <= sp.start < sp.end <= len(text)


def test_generate_dataset_size_matches_request():
    out = generate_dataset(n=5, seed=0)
    assert len(out) == 5
    for text, spans in out:
        assert isinstance(text, str) and len(text) > 0
        assert len(spans) > 0
