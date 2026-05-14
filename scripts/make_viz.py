"""Generate result.png (extraction example side-by-side + per-entity F1 + confusion matrix)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForTokenClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.extract import assemble, predict_spans  # noqa: E402
from src.generator import make_cv  # noqa: E402

MODEL_DIR = ROOT / "runs" / "model"

ENTITY_COLORS = {
    "PERSON": "#fbb4b4",
    "SKILL": "#bce4b5",
    "ORG":   "#b4cefb",
    "DATE":  "#ffe4a8",
    "TITLE": "#e4b4fb",
    "DEGREE": "#fbcfb4",
    "SCHOOL": "#b4fbf2",
}


def main():
    runs = ROOT / "runs"
    out = ROOT / "assets"
    out.mkdir(exist_ok=True)
    with open(runs / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    confusion = np.array(metrics["confusion"])
    labels = metrics["confusion_labels"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    model.eval()

    text, _ = make_cv(seed=12345)
    spans = predict_spans(text, model, tokenizer)
    cv = assemble(text, spans)

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.4, 1])

    ax_text = fig.add_subplot(gs[0, 0])
    ax_text.axis("off")
    ax_text.set_title("Source text (entities highlighted)", fontsize=11)
    ax_text.set_xlim(0, 1); ax_text.set_ylim(0, 1)
    cur_y = 0.96
    line_h = 0.038
    cursor = 0
    spans_sorted = sorted(spans, key=lambda x: x[0])
    for line in text.split("\n"):
        x = 0.02
        line_end = cursor + len(line)
        line_spans = [(s, e, t) for s, e, t in spans_sorted
                      if not (e <= cursor or s >= line_end)]
        idx = cursor
        for s, e, t in line_spans:
            if s > idx:
                seg = text[idx:s]
                ax_text.text(x, cur_y, seg, fontsize=8, family="monospace", va="top")
                x += 0.0055 * len(seg)
            seg = text[max(s, cursor):min(e, line_end)]
            ax_text.add_patch(patches.Rectangle(
                (x - 0.002, cur_y - 0.022), 0.0055 * len(seg) + 0.004, 0.026,
                facecolor=ENTITY_COLORS.get(t, "#ddd"), edgecolor="none", alpha=0.85))
            ax_text.text(x, cur_y, seg, fontsize=8, family="monospace", va="top", weight="bold")
            x += 0.0055 * len(seg)
            idx = e
        if idx < line_end:
            ax_text.text(x, cur_y, text[idx:line_end], fontsize=8, family="monospace", va="top")
        cursor = line_end + 1
        cur_y -= line_h

    ax_json = fig.add_subplot(gs[0, 1])
    ax_json.axis("off")
    ax_json.set_title("Structured output (pydantic-validated)", fontsize=11)
    js = json.dumps(cv.model_dump(), ensure_ascii=False, indent=2)
    ax_json.text(0.02, 0.96, js, fontsize=8, family="monospace", va="top",
                 bbox=dict(facecolor="#f7f7f7", edgecolor="#ccc"))

    ax_f1 = fig.add_subplot(gs[0, 2])
    per = metrics["per_entity_f1"]
    names = list(per.keys())
    vals = [per[n] * 100 for n in names]
    colors = [ENTITY_COLORS.get(n, "#888") for n in names]
    ax_f1.barh(names[::-1], vals[::-1], color=colors[::-1])
    ax_f1.set_xlim(0, 102)
    ax_f1.set_xlabel("F1 (%)")
    ax_f1.set_title(f"Per-entity F1 (macro = {metrics['final_macro_f1']*100:.1f}%)")
    for i, v in enumerate(vals[::-1]):
        ax_f1.text(v + 1, i, f"{v:.1f}", va="center", fontsize=9)
    ax_f1.grid(axis="x", alpha=0.3)

    ax_conf = fig.add_subplot(gs[1, :])
    norm = confusion / confusion.sum(axis=1, keepdims=True).clip(min=1)
    im = ax_conf.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax_conf.set_xticks(range(len(labels))); ax_conf.set_xticklabels(labels, rotation=20)
    ax_conf.set_yticks(range(len(labels))); ax_conf.set_yticklabels(labels)
    ax_conf.set_xlabel("predicted"); ax_conf.set_ylabel("true")
    ax_conf.set_title("Token-level confusion matrix on test set")
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = int(confusion[i, j])
            if v == 0:
                continue
            ax_conf.text(j, i, str(v), ha="center", va="center",
                         color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax_conf, fraction=0.025)

    fig.suptitle("cv-extract-json: NER + pydantic + verbatim verification", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "result.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out / 'result.png'}")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis("off")

    def box(x, y, w, h, label, color):
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.07",
                                            edgecolor=color, facecolor="white", linewidth=2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10, color=color)

    box(0.02, 0.45, 0.14, 0.30, "raw CV text\n(French)", "#444")
    box(0.18, 0.45, 0.16, 0.30, "DistilCamemBERT\nNER head", "#1f77b4")
    box(0.36, 0.45, 0.14, 0.30, "BIO tags\n(15 classes)", "#888")
    box(0.52, 0.45, 0.14, 0.30, "Span aggregator", "#ff7f0e")
    box(0.68, 0.45, 0.14, 0.30, "Pydantic\nschema", "#2ca02c")
    box(0.84, 0.45, 0.14, 0.30, "Verbatim\nguardrail", "#d62728")
    for x in [0.16, 0.34, 0.50, 0.66, 0.82]:
        ax.annotate("", xy=(x + 0.02, 0.60), xytext=(x, 0.60), arrowprops=dict(arrowstyle="->", color="#555"))

    ax.text(0.5, 0.92, "Architecture: token classification + JSON schema + anti-hallucination check",
            ha="center", fontsize=12, weight="bold")
    ax.text(0.5, 0.18, "Verbatim guardrail rejects any field that does not appear textually in the source CV.",
            ha="center", fontsize=9, color="#555")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out / "architecture.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out / 'architecture.png'}")


if __name__ == "__main__":
    main()
