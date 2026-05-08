"""
Modal token-class stratification for Pythia training-dynamics curvature.

This is the follow-up to the Pythia-1B checkpoint sweep. The prior sweep saved
only per-layer aggregate correlations. This script re-extracts selected
checkpoint/layer/token records so we can test whether early negative
curvature->entropy correlations are driven by lexical/tokenization classes.

Example:
    modal run viz/modal_pythia_token_stratification.py \
        --revision step512 \
        --layers 1,5 \
        --limit 32 \
        --max-length 160

Output:
    results/modal_pythia_token_stratification/<model>_<revision>_layers-*.json
    results/modal_pythia_token_stratification/<model>_<revision>_layers-*.jsonl
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import modal

app = modal.App("memory-gravity-pythia-token-stratification")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.48.3",
        "datasets==3.2.0",
        "numpy==2.2.2",
        "accelerate==1.2.1",
    )
)


def _safe_name(value: str | None) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value or "none")


def _parse_layers(layers: str) -> list[int]:
    out = []
    for part in layers.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise ValueError("--layers must contain at least one integer layer")
    return out


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60,
    memory=65536,
)
def run_remote(model_id: str = "EleutherAI/pythia-1b",
               revision: str = "step512",
               layers: str = "5",
               dataset_name: str = "lambada",
               split: str = "validation",
               limit: int = 32,
               max_length: int = 160,
               seed: int = 0) -> dict:
    import math
    import string
    import time

    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    selected_layers = _parse_layers(layers)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    last_exc = None
    for attempt in range(3):
        try:
            ds = load_dataset(dataset_name, split=split)
            break
        except Exception as exc:
            last_exc = exc
            if attempt == 2:
                raise last_exc
            time.sleep(10 * (attempt + 1))

    texts = []
    for row in ds:
        text = row.get("text") or row.get("sentence") or row.get("passage")
        if text and len(text.split()) >= 20:
            texts.append(text)
        if len(texts) >= limit:
            break

    punct = set(string.punctuation)

    def token_class(raw_token: str, decoded: str) -> str:
        stripped = decoded.strip()
        if not stripped:
            return "whitespace_newline"
        if raw_token.startswith("Ġ"):
            return "word_start"
        if stripped.isdigit():
            return "digit"
        if all(ch in punct for ch in stripped):
            return "punctuation"
        if stripped.replace("_", "").isalnum():
            return "word_piece_continuation"
        return "other"

    def pearson(x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.size < 3 or y.size != x.size:
            return float("nan")
        x = x - x.mean()
        y = y - y.mean()
        denom = math.sqrt(float(np.sum(x * x) * np.sum(y * y)))
        return float(np.sum(x * y) / denom) if denom else float("nan")

    rows = []

    @torch.no_grad()
    def process(passage_id: int, text: str):
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(device)
        if enc.input_ids.shape[1] < 12:
            return
        token_ids = enc.input_ids[0].detach().cpu().tolist()
        raw_tokens = tokenizer.convert_ids_to_tokens(token_ids)
        decoded_tokens = [tokenizer.decode([tok]) for tok in token_ids]

        out = model(**enc, output_hidden_states=True, return_dict=True)
        logits = out.logits[0].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = (-(probs * log_probs).sum(dim=-1)).detach().cpu().numpy().astype(np.float32)

        for layer in selected_layers:
            h = out.hidden_states[layer + 1][0].float().detach().cpu().numpy()
            v = h[1:] - h[:-1]
            if v.shape[0] < 6:
                continue
            speeds = np.linalg.norm(v, axis=-1).astype(np.float32)
            norms = np.linalg.norm(v, axis=-1, keepdims=True)
            unit = v / np.maximum(norms, 1e-12)
            cos = np.sum(unit[:-1] * unit[1:], axis=-1)
            raw_curv = np.arccos(np.clip(cos, -1.0, 1.0)).astype(np.float32)

            for k in range(6, h.shape[0]):
                speed_window = speeds[max(0, k - 3):k]
                curv_window = raw_curv[max(0, k - 4):max(0, k - 1)]
                if speed_window.size == 0 or curv_window.size == 0:
                    continue
                raw_token = raw_tokens[k]
                decoded = decoded_tokens[k]
                rows.append({
                    "revision": revision,
                    "passage_id": passage_id,
                    "layer": layer,
                    "token_index": k,
                    "token_id": int(token_ids[k]),
                    "token_raw": raw_token,
                    "token_text": decoded,
                    "token_class": token_class(raw_token, decoded),
                    "entropy": float(entropy[k]),
                    "contextual_speed": float(np.mean(speed_window)),
                    "contextual_curvature": float(np.mean(curv_window)),
                    "contextual_curvature_degrees": float(np.degrees(np.mean(curv_window))),
                })

    for passage_id, text in enumerate(texts):
        process(passage_id, text)

    groups: dict[tuple[int, str], list[dict]] = {}
    for row in rows:
        key = (int(row["layer"]), str(row["token_class"]))
        groups.setdefault(key, []).append(row)

    summary = []
    for (layer, cls), subset in sorted(groups.items()):
        curv = [r["contextual_curvature"] for r in subset]
        ent = [r["entropy"] for r in subset]
        speed = [r["contextual_speed"] for r in subset]
        summary.append({
            "revision": revision,
            "layer": layer,
            "token_class": cls,
            "n": len(subset),
            "mean_curvature": float(np.mean(curv)) if curv else float("nan"),
            "mean_curvature_degrees": float(np.degrees(np.mean(curv))) if curv else float("nan"),
            "mean_entropy": float(np.mean(ent)) if ent else float("nan"),
            "mean_speed": float(np.mean(speed)) if speed else float("nan"),
            "curvature_entropy_pearson": pearson(curv, ent),
        })

    return {
        "schema_version": "pythia_token_stratification_v1",
        "model_id": model_id,
        "revision": revision,
        "dataset_name": dataset_name,
        "split": split,
        "limit": limit,
        "max_length": max_length,
        "n_texts": len(texts),
        "layers": selected_layers,
        "summary": summary,
        "rows": rows,
    }


@app.local_entrypoint()
def main(model_id: str = "EleutherAI/pythia-1b",
         revision: str = "step512",
         layers: str = "5",
         dataset_name: str = "lambada",
         split: str = "validation",
         limit: int = 32,
         max_length: int = 160,
         output_dir: str = "results/modal_pythia_token_stratification"):
    result = run_remote.remote(
        model_id=model_id,
        revision=revision,
        layers=layers,
        dataset_name=dataset_name,
        split=split,
        limit=limit,
        max_length=max_length,
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_suffix = "layers-" + _safe_name(layers.replace(",", "-"))
    stem = f"{_safe_name(model_id)}_{_safe_name(revision)}_{layer_suffix}"
    rows = result.pop("rows")
    summary_path = out_dir / f"{stem}_summary.json"
    rows_path = out_dir / f"{stem}_rows.jsonl"
    summary_path.write_text(json.dumps(result, indent=2))
    with rows_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "model_id": result["model_id"],
        "revision": result["revision"],
        "layers": result["layers"],
        "n_rows": len(rows),
        "summary_path": str(summary_path),
        "rows_path": str(rows_path),
    }, indent=2))
