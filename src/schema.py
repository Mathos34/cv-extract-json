"""Strict pydantic schema for the structured CV output."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Experience(BaseModel):
    title: str | None = None
    org: str | None = None
    period: str | None = None


class Education(BaseModel):
    degree: str | None = None
    school: str | None = None
    year: str | None = None


class StructuredCV(BaseModel):
    person: str | None = None
    skills: list[str] = Field(default_factory=list)
    experiences: list[Experience] = Field(default_factory=list)
    educations: list[Education] = Field(default_factory=list)


# BIO label set: entity types we recognize at token level.
ENTITY_TYPES = ["PERSON", "SKILL", "ORG", "DATE", "TITLE", "DEGREE", "SCHOOL"]
LABELS = ["O"] + [f"{p}-{e}" for e in ENTITY_TYPES for p in ("B", "I")]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
