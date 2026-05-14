"""Convert (text, char spans) examples into BIO-tagged token sequences for training."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from .generator import Span
from .schema import LABEL2ID


@dataclass
class Example:
    text: str
    spans: list[Span]


def encode_example(text: str, spans: list[Span], tokenizer, max_length: int = 256):
    enc = tokenizer(text, truncation=True, max_length=max_length, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    labels = []
    for tok_idx, (s, e) in enumerate(offsets):
        if s == e:
            labels.append(-100)
            continue
        # Find span that contains the token's start char.
        chosen = None
        for sp in spans:
            if sp.start <= s < sp.end:
                chosen = sp
                break
        if chosen is None:
            labels.append(LABEL2ID["O"])
        else:
            prefix = "B" if s == chosen.start else "I"
            labels.append(LABEL2ID[f"{prefix}-{chosen.label}"])
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels,
        "offset_mapping": offsets,
    }


class CVDataset(Dataset):
    def __init__(self, examples: list[Example], tokenizer, max_length: int = 256):
        self.encoded = [encode_example(ex.text, ex.spans, tokenizer, max_length) for ex in examples]
        self.examples = examples

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int):
        e = self.encoded[idx]
        return {
            "input_ids": torch.tensor(e["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(e["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(e["labels"], dtype=torch.long),
        }


def collate(batch):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        input_ids[i, :n] = b["input_ids"]
        attention[i, :n] = b["attention_mask"]
        labels[i, :n] = b["labels"]
    return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}
