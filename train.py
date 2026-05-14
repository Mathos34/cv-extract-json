"""Fine-tune distilcamembert-base for CV NER on a synthetic French dataset."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.data import CVDataset, Example, collate
from src.extract import assemble, predict_spans
from src.generator import generate_dataset
from src.schema import ENTITY_TYPES, ID2LABEL, LABEL2ID
from src.verbatim import verify

SEED = 42
MODEL_NAME = "cmarkea/distilcamembert-base"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def token_metrics(model, loader, device):
    """Macro-F1 and per-entity F1 over BIO tags (excluding O and -100)."""
    model.eval()
    tp = {e: 0 for e in ENTITY_TYPES}
    fp = {e: 0 for e in ENTITY_TYPES}
    fn = {e: 0 for e in ENTITY_TYPES}
    confusion = np.zeros((len(ENTITY_TYPES) + 1, len(ENTITY_TYPES) + 1), dtype=np.int64)
    type_to_idx = {e: i + 1 for i, e in enumerate(ENTITY_TYPES)}
    type_to_idx["O"] = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
            preds = logits.argmax(dim=-1)
            mask = batch["labels"] != -100
            for p, t in zip(preds[mask].tolist(), batch["labels"][mask].tolist()):
                pl = ID2LABEL[p]
                tl = ID2LABEL[t]
                pe = "O" if pl == "O" else pl.split("-", 1)[1]
                te = "O" if tl == "O" else tl.split("-", 1)[1]
                confusion[type_to_idx[te], type_to_idx[pe]] += 1
                if te == "O" and pe == "O":
                    continue
                if pe == te:
                    tp[te] += 1
                else:
                    if te != "O":
                        fn[te] += 1
                    if pe != "O":
                        fp[pe] += 1
    f1s = {}
    for e in ENTITY_TYPES:
        denom = (2 * tp[e] + fp[e] + fn[e])
        f1s[e] = 0.0 if denom == 0 else (2 * tp[e]) / denom
    macro = sum(f1s.values()) / len(f1s)
    return macro, f1s, confusion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=400)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--out", type=str, default="runs")
    args = parser.parse_args()

    set_seed(SEED)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n_train + args.n_test} synthetic CVs...")
    raw = generate_dataset(args.n_train + args.n_test, seed=SEED)
    examples = [Example(t, s) for t, s in raw]
    train_ex = examples[: args.n_train]
    test_ex = examples[args.n_train:]

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL2ID),
        id2label=ID2LABEL, label2id=LABEL2ID,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.1f}M params")

    train_ds = CVDataset(train_ex, tokenizer)
    test_ds = CVDataset(test_ex, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate)

    device = torch.device("cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history = {"epoch": [], "train_loss": [], "macro_f1": []}
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        n = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            opt.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += out.loss.item() * batch["input_ids"].shape[0]
            n += batch["input_ids"].shape[0]
        train_loss = running / n
        macro, per_ent, confusion = token_metrics(model, test_loader, device)
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["macro_f1"].append(macro)
        print(f"  epoch {epoch+1}: loss={train_loss:.4f} macro_f1={macro*100:.2f}%")
    print(f"Training took {time.time()-t0:.1f}s")

    model.save_pretrained(out_dir / "model")
    tokenizer.save_pretrained(out_dir / "model")

    print("Running verbatim verification on test set...")
    verbatim_scores = []
    for ex in test_ex:
        spans = predict_spans(ex.text, model, tokenizer)
        cv = assemble(ex.text, spans)
        rep = verify(cv, ex.text)
        verbatim_scores.append(rep.confidence)
    avg_verbatim = float(np.mean(verbatim_scores))
    print(f"Average verbatim confidence: {avg_verbatim*100:.1f}%")

    metrics = {
        "n_train": len(train_ex),
        "n_test": len(test_ex),
        "epochs": args.epochs,
        "params_millions": round(n_params / 1e6, 1),
        "final_macro_f1": history["macro_f1"][-1],
        "per_entity_f1": per_ent,
        "verbatim_confidence_mean": avg_verbatim,
        "history": history,
        "confusion": confusion.tolist(),
        "confusion_labels": ["O"] + ENTITY_TYPES,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
