"""Unit tests for the verbatim anti-hallucination guardrail."""
from src.schema import Education, Experience, StructuredCV
from src.verbatim import normalize, verify


def test_normalize_lowercases_and_strips_diacritics():
    assert normalize("EPiTA") == "epita"
    assert normalize("Université Paris-Saclay") == "universite paris-saclay"
    assert normalize("  Élise  ") == "elise"


def test_normalize_collapses_simple_whitespace_edges():
    assert normalize("Hello world") == "hello world"
    assert normalize("") == ""


def test_verify_full_match_returns_confidence_one():
    source = "Jean Dupont\n\nDeveloper at Acme (2020-2024)\nPython\nSQL\n"
    cv = StructuredCV(
        person="Jean Dupont",
        skills=["Python", "SQL"],
        experiences=[Experience(title="Developer", org="Acme", period="2020-2024")],
        educations=[],
    )
    report = verify(cv, source)
    assert report.confidence == 1.0
    assert report.total_fields == 6  # person + 2 skills + 3 experience fields
    assert report.verified_fields == 6
    assert report.missing == []


def test_verify_flags_hallucinated_field():
    source = "Jean Dupont\nPython\n"
    cv = StructuredCV(person="Jean Dupont", skills=["Python", "Rust"])
    report = verify(cv, source)
    assert report.verified_fields == 2
    assert report.total_fields == 3
    assert ("skills[1]", "Rust") in report.missing
    assert 0 < report.confidence < 1


def test_verify_handles_diacritics_via_normalization():
    source = "diplome Universite Paris-Saclay 2024"
    cv = StructuredCV(
        person=None,
        educations=[Education(degree="Diplôme", school="Université Paris-Saclay", year="2024")],
    )
    report = verify(cv, source)
    assert report.confidence == 1.0


def test_verify_ignores_none_and_empty_fields():
    source = "anything"
    cv = StructuredCV(
        person=None,
        skills=[],
        experiences=[Experience(title=None, org=None, period=None)],
        educations=[],
    )
    report = verify(cv, source)
    assert report.total_fields == 0
    assert report.confidence == 1.0
