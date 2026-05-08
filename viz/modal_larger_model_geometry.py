"""
Modal larger-model speed/curvature test.

Runs a compact remote experiment on larger Hugging Face causal LMs and reports
per-layer correlations between next-token entropy and:

  - contextual speed: mean recent ||h[t+1] - h[t]||
  - paper-style contextual curvature:
      C_k = mean(c_{k-4}, c_{k-3}, c_{k-2})
      c_i = arccos(v_i dot v_{i+1} / (|v_i||v_{i+1}|))

This is intentionally separate from the v1 TinyStories diagnostic. It is the
first step toward a paper-faithful curvature replication track.

Usage from repo root:
    modal run viz/modal_larger_model_geometry.py --model-id gpt2-xl --limit 48
    modal run viz/modal_larger_model_geometry.py --model-id EleutherAI/pythia-1b \
        --limit 32 --max-length 160 --output-dir results/modal_pythia_sweep
    modal run viz/modal_larger_model_geometry.py --model-id EleutherAI/pythia-1b \
        --revision step32000 --limit 32 --max-length 160 \
        --output-dir results/modal_pythia_training_dynamics

The local entrypoint writes:
    <output_dir>/<safe_model_id>[_<safe_revision>]_summary.json
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import modal

app = modal.App("memory-gravity-larger-geometry")

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


def _safe_name(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id)


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60,
    memory=65536,
)
def run_remote(model_id: str = "gpt2-xl",
               revision: str | None = None,
               dataset_name: str = "lambada",
               split: str = "validation",
               limit: int = 48,
               max_length: int = 192,
               seed: int = 0) -> dict:
    import numpy as np
    import torch
    import time
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
                raise
            time.sleep(10 * (attempt + 1))
    texts = []
    for row in ds:
        text = row.get("text") or row.get("sentence") or row.get("passage")
        if text and len(text.split()) >= 20:
            texts.append(text)
        if len(texts) >= limit:
            break

    def count_layers() -> int:
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return len(model.transformer.h)
        if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
            return len(model.gpt_neox.layers)
        if hasattr(model, "model"):
            inner = model.model
            if hasattr(inner, "decoder") and hasattr(inner.decoder, "layers"):
                return len(inner.decoder.layers)
            if hasattr(inner, "layers"):
                return len(inner.layers)
        raise ValueError(f"Unsupported architecture for layer counting: {type(model).__name__}")

    n_layers = count_layers()
    pooled = {
        layer: {
            "speed": [],
            "curvature": [],
            "entropy_for_speed": [],
            "entropy_for_curvature": [],
        }
        for layer in range(n_layers)
    }

    @torch.no_grad()
    def process(text: str):
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(device)
        if enc.input_ids.shape[1] < 12:
            return
        out = model(**enc, output_hidden_states=True, return_dict=True)
        logits = out.logits[0].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = (-(probs * log_probs).sum(dim=-1)).detach().cpu().numpy().astype(np.float32)

        for layer in range(n_layers):
            # HF hidden_states[0] is embedding output; [layer+1] is post-block residual.
            h = out.hidden_states[layer + 1][0].float().detach().cpu().numpy()
            v = h[1:] - h[:-1]  # (T-1, d)
            if v.shape[0] < 6:
                continue
            speeds = np.linalg.norm(v, axis=-1).astype(np.float32)
            norms = np.linalg.norm(v, axis=-1, keepdims=True)
            unit = v / np.maximum(norms, 1e-12)
            cos = np.sum(unit[:-1] * unit[1:], axis=-1)
            raw_curv = np.arccos(np.clip(cos, -1.0, 1.0)).astype(np.float32)  # (T-2,)

            # Contextual speed and curvature aligned to token k. For k >= 6:
            # speed_ctx[k] = mean speed over steps k-3:k
            # curv_ctx[k] = mean raw_curv over k-4:k-1 (paper window).
            for k in range(6, h.shape[0]):
                speed_window = speeds[max(0, k - 3):k]
                curv_window = raw_curv[max(0, k - 4):max(0, k - 1)]
                if speed_window.size == 0 or curv_window.size == 0:
                    continue
                pooled[layer]["speed"].append(float(np.mean(speed_window)))
                pooled[layer]["entropy_for_speed"].append(float(entropy[k]))
                pooled[layer]["curvature"].append(float(np.mean(curv_window)))
                pooled[layer]["entropy_for_curvature"].append(float(entropy[k]))

    for text in texts:
        process(text)

    def pearson(x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.size < 3 or y.size != x.size:
            return float("nan")
        x = x - x.mean()
        y = y - y.mean()
        denom = math.sqrt(float(np.sum(x * x) * np.sum(y * y)))
        return float(np.sum(x * y) / denom) if denom else float("nan")

    def rankdata(a):
        a = np.asarray(a, dtype=np.float64)
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, a.size + 1, dtype=np.float64)
        sorted_vals = a[order]
        i = 0
        while i < a.size:
            j = i + 1
            while j < a.size and sorted_vals[j] == sorted_vals[i]:
                j += 1
            if j - i > 1:
                ranks[order[i:j]] = ranks[order[i:j]].mean()
            i = j
        return ranks

    def spearman(x, y):
        if len(x) < 3:
            return float("nan")
        return pearson(rankdata(x), rankdata(y))

    layers = []
    for layer in range(n_layers):
        d = pooled[layer]
        layers.append({
            "layer": layer,
            "n": len(d["speed"]),
            "speed_entropy_pearson": pearson(d["speed"], d["entropy_for_speed"]),
            "speed_entropy_spearman": spearman(d["speed"], d["entropy_for_speed"]),
            "curvature_entropy_pearson": pearson(d["curvature"], d["entropy_for_curvature"]),
            "curvature_entropy_spearman": spearman(d["curvature"], d["entropy_for_curvature"]),
            "mean_speed": float(np.mean(d["speed"])) if d["speed"] else float("nan"),
            "mean_curvature_degrees": float(np.degrees(np.mean(d["curvature"]))) if d["curvature"] else float("nan"),
        })

    best_speed = max(layers, key=lambda r: abs(r["speed_entropy_pearson"]) if math.isfinite(r["speed_entropy_pearson"]) else -1)
    best_curv = max(layers, key=lambda r: abs(r["curvature_entropy_pearson"]) if math.isfinite(r["curvature_entropy_pearson"]) else -1)
    return {
        "schema_version": "modal_larger_geometry_v1",
        "model_id": model_id,
        "revision": revision,
        "dataset_name": dataset_name,
        "split": split,
        "limit": limit,
        "max_length": max_length,
        "n_texts": len(texts),
        "n_layers": n_layers,
        "best_speed_layer": best_speed,
        "best_curvature_layer": best_curv,
        "layers": layers,
    }


@app.local_entrypoint()
def main(model_id: str = "gpt2-xl",
         revision: str | None = None,
         dataset_name: str = "lambada",
         split: str = "validation",
         limit: int = 48,
         max_length: int = 192,
         output_dir: str = "results/modal_larger_geometry"):
    result = run_remote.remote(
        model_id=model_id,
        revision=revision,
        dataset_name=dataset_name,
        split=split,
        limit=limit,
        max_length=max_length,
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    revision_suffix = f"_{_safe_name(revision)}" if revision else ""
    out_path = out_dir / f"{_safe_name(model_id)}{revision_suffix}_summary.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "model_id": result["model_id"],
        "revision": result["revision"],
        "n_texts": result["n_texts"],
        "best_speed_layer": result["best_speed_layer"],
        "best_curvature_layer": result["best_curvature_layer"],
        "out_path": str(out_path),
    }, indent=2))
