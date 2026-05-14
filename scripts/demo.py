"""Run inference on a synthetic CV and print the structured output + verification."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from transformers import AutoModelForTokenClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.extract import assemble, predict_spans  # noqa: E402
from src.generator import make_cv  # noqa: E402
from src.verbatim import verify  # noqa: E402

MODEL_DIR = ROOT / "runs" / "model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
model.eval()

text, _ = make_cv(seed=9999)
print("=== INPUT TEXT ===")
print(text)
print("=== EXTRACTED JSON ===")
spans = predict_spans(text, model, tokenizer)
cv = assemble(text, spans)
print(json.dumps(cv.model_dump(), ensure_ascii=False, indent=2))
print("=== VERBATIM CHECK ===")
rep = verify(cv, text)
print(f"Confidence: {rep.confidence*100:.1f}% ({rep.verified_fields}/{rep.total_fields} fields verified)")
if rep.missing:
    print("Missing:")
    for path, val in rep.missing:
        print(f"  - {path}: {val!r}")
