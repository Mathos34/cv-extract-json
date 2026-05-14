"""Synthetic French CV generator. Produces (text, character-span entity list).

Each generated CV is a short free-form CV with deterministic templates so we
can record exact (start, end, label) spans for free, no manual annotation.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from faker import Faker

SKILL_POOL = [
    "Python", "PyTorch", "TensorFlow", "JavaScript", "TypeScript", "React",
    "Next.js", "Node.js", "FastAPI", "Django", "Flask", "PostgreSQL", "MySQL",
    "MongoDB", "Redis", "Docker", "Kubernetes", "AWS", "GCP", "Azure",
    "Git", "Linux", "Bash", "SQL", "NumPy", "Pandas", "scikit-learn",
    "Hugging Face", "LangChain", "Spark", "Airflow", "Tableau", "Power BI",
    "C++", "Rust", "Go", "Java", "Scala", "Kotlin", "Swift",
]

TITLE_POOL = [
    "Data Scientist", "Machine Learning Engineer", "Software Engineer",
    "Backend Developer", "Frontend Developer", "Full Stack Developer",
    "DevOps Engineer", "AI Research Engineer", "Data Engineer",
    "Product Manager", "Technical Lead", "Solutions Architect",
]

DEGREE_POOL = [
    "Master en Informatique", "Master Data Science", "Diplome d'Ingenieur",
    "Licence Informatique", "Bachelor Computer Science", "MBA",
    "Master Intelligence Artificielle", "Doctorat en Informatique",
]

SCHOOL_POOL = [
    "ECE Paris", "EPITA", "Sorbonne Universite", "Universite Paris-Saclay",
    "INSA Lyon", "Centrale Lyon", "ENSIMAG", "EPF",
    "Universite de Bordeaux", "Telecom Paris", "Polytechnique",
]

ORG_POOL = [
    "BNP Paribas", "Societe Generale", "Capgemini", "Atos", "Orange",
    "Thales", "Dassault Systemes", "Criteo", "Doctolib", "BlaBlaCar",
    "Mistral AI", "OpenClassrooms", "Sopra Steria", "Renault Group",
    "Deezer", "Total Energies", "L Oreal Tech",
]


@dataclass
class Span:
    start: int
    end: int
    label: str

    def text(self, source: str) -> str:
        return source[self.start: self.end]


def _add(parts: list[str], spans: list[Span], piece: str, label: str | None) -> None:
    """Append a piece to the running text, possibly with a span label."""
    if label is None:
        parts.append(piece)
        return
    cursor = sum(len(p) for p in parts)
    spans.append(Span(cursor, cursor + len(piece), label))
    parts.append(piece)


def make_cv(seed: int) -> tuple[str, list[Span]]:
    rng = random.Random(seed)
    faker = Faker("fr_FR")
    faker.seed_instance(seed)

    parts: list[str] = []
    spans: list[Span] = []

    name = faker.name()
    _add(parts, spans, name, "PERSON")
    _add(parts, spans, "\n", None)
    _add(parts, spans, faker.email(), None)
    _add(parts, spans, " | " + faker.phone_number(), None)
    _add(parts, spans, "\n\n", None)
    _add(parts, spans, "EXPERIENCE\n", None)
    n_exp = rng.randint(2, 3)
    for _ in range(n_exp):
        _add(parts, spans, "- ", None)
        _add(parts, spans, rng.choice(TITLE_POOL), "TITLE")
        _add(parts, spans, " chez ", None)
        _add(parts, spans, rng.choice(ORG_POOL), "ORG")
        _add(parts, spans, " (", None)
        y1 = rng.randint(2015, 2024)
        y2 = rng.randint(y1, 2026)
        period = f"{y1}-{y2}"
        _add(parts, spans, period, "DATE")
        _add(parts, spans, ")\n", None)

    _add(parts, spans, "\nFORMATION\n", None)
    n_edu = rng.randint(1, 2)
    for _ in range(n_edu):
        _add(parts, spans, "- ", None)
        _add(parts, spans, rng.choice(DEGREE_POOL), "DEGREE")
        _add(parts, spans, ", ", None)
        _add(parts, spans, rng.choice(SCHOOL_POOL), "SCHOOL")
        _add(parts, spans, ", ", None)
        _add(parts, spans, str(rng.randint(2018, 2026)), "DATE")
        _add(parts, spans, "\n", None)

    _add(parts, spans, "\nCOMPETENCES\n", None)
    skills = rng.sample(SKILL_POOL, k=rng.randint(4, 7))
    for i, s in enumerate(skills):
        _add(parts, spans, "- ", None)
        _add(parts, spans, s, "SKILL")
        _add(parts, spans, "\n", None)

    text = "".join(parts)
    return text, spans


def generate_dataset(n: int, seed: int = 42) -> list[tuple[str, list[Span]]]:
    return [make_cv(seed + i) for i in range(n)]
