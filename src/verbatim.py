"""Verbatim verification: every extracted field must appear textually in the source.

Normalization: lowercase, strip diacritics. We accept a match if the normalized
candidate string is a substring of the normalized source.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from .schema import StructuredCV


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().strip()


@dataclass
class VerbatimReport:
    total_fields: int
    verified_fields: int
    missing: list[tuple[str, str]]  # (path, value)

    @property
    def confidence(self) -> float:
        if self.total_fields == 0:
            return 1.0
        return self.verified_fields / self.total_fields


def _check(value: str | None, source_norm: str, path: str, report_missing: list[tuple[str, str]]):
    if value is None or value == "":
        return 0, 0
    if normalize(value) in source_norm:
        return 1, 1
    report_missing.append((path, value))
    return 1, 0


def verify(cv: StructuredCV, source: str) -> VerbatimReport:
    source_norm = normalize(source)
    total = 0
    verified = 0
    missing: list[tuple[str, str]] = []
    t, v = _check(cv.person, source_norm, "person", missing)
    total += t; verified += v
    for i, sk in enumerate(cv.skills):
        t, v = _check(sk, source_norm, f"skills[{i}]", missing)
        total += t; verified += v
    for i, exp in enumerate(cv.experiences):
        for k in ("title", "org", "period"):
            t, v = _check(getattr(exp, k), source_norm, f"experiences[{i}].{k}", missing)
            total += t; verified += v
    for i, edu in enumerate(cv.educations):
        for k in ("degree", "school", "year"):
            t, v = _check(getattr(edu, k), source_norm, f"educations[{i}].{k}", missing)
            total += t; verified += v
    return VerbatimReport(total, verified, missing)
