"""Inference: run the trained model on raw text -> StructuredCV."""
from __future__ import annotations

import torch

from .schema import ENTITY_TYPES, ID2LABEL, Education, Experience, StructuredCV


def predict_spans(text: str, model, tokenizer, max_length: int = 256) -> list[tuple[int, int, str]]:
    enc = tokenizer(text, truncation=True, max_length=max_length, return_offsets_mapping=True,
                    return_tensors="pt")
    offsets = enc.pop("offset_mapping")[0].tolist()
    with torch.no_grad():
        logits = model(**enc).logits[0]
    pred_ids = logits.argmax(dim=-1).tolist()
    spans: list[tuple[int, int, str]] = []
    cur_label: str | None = None
    cur_start = -1
    cur_end = -1
    for tid, (s, e) in zip(pred_ids, offsets):
        if s == e:
            continue
        label = ID2LABEL[tid]
        if label == "O":
            if cur_label is not None:
                spans.append((cur_start, cur_end, cur_label))
                cur_label = None
            continue
        prefix, ent = label.split("-", 1)
        if prefix == "B" or cur_label != ent:
            if cur_label is not None:
                spans.append((cur_start, cur_end, cur_label))
            cur_label = ent
            cur_start = s
            cur_end = e
        else:
            cur_end = e
    if cur_label is not None:
        spans.append((cur_start, cur_end, cur_label))
    return spans


def assemble(text: str, spans: list[tuple[int, int, str]]) -> StructuredCV:
    by_type: dict[str, list[str]] = {t: [] for t in ENTITY_TYPES}
    for s, e, t in spans:
        by_type[t].append(text[s:e].strip())
    person = by_type["PERSON"][0] if by_type["PERSON"] else None
    skills = list(dict.fromkeys(by_type["SKILL"]))
    experiences: list[Experience] = []
    titles, orgs, dates = by_type["TITLE"], by_type["ORG"], by_type["DATE"]
    for i in range(max(len(titles), len(orgs))):
        experiences.append(Experience(
            title=titles[i] if i < len(titles) else None,
            org=orgs[i] if i < len(orgs) else None,
            period=dates[i] if i < len(dates) else None,
        ))
    educations: list[Education] = []
    degrees, schools = by_type["DEGREE"], by_type["SCHOOL"]
    edu_dates = dates[len(titles):]
    for i in range(max(len(degrees), len(schools))):
        educations.append(Education(
            degree=degrees[i] if i < len(degrees) else None,
            school=schools[i] if i < len(schools) else None,
            year=edu_dates[i] if i < len(edu_dates) else None,
        ))
    return StructuredCV(person=person, skills=skills, experiences=experiences, educations=educations)
