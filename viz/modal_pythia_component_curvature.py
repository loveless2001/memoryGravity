"""
Modal Pythia-1B component curvature extraction.

This is the next discriminant for the Pythia curvature sign reversal. It
compares curvature/entropy coupling for three trajectories at selected layers:

  - post_block_residual: the normal hidden state after the transformer block
  - attention_output: the selected block's attention module output
  - mlp_output: the selected block's MLP module output

For GPT-NeoX/Pythia these component outputs are residual deltas before they are
added back into the stream. The goal is not to reproduce the full residual
trajectory; it is to test whether the training-time sign flip is concentrated
in attention, MLP, or both.

Example:
    modal run viz/modal_pythia_component_curvature.py \
        --revision step512 \
        --layers 1,5 \
        --limit 32 \
        --max-length 160

Output:
    results/modal_pythia_component_curvature/<model>_<revision>_layers-*_summary.json
    results/modal_pythia_component_curvature/<model>_<revision>_layers-*_rows.jsonl
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import modal


app = modal.App("memory-gravity-pythia-component-curvature")

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

    if not hasattr(model, "gpt_neox") or not hasattr(model.gpt_neox, "layers"):
        raise ValueError(f"Expected GPTNeoX/Pythia architecture, got {type(model).__name__}")
    blocks = model.gpt_neox.layers

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

    def component_tensor(output):
        # GPTNeoX attention returns a tuple whose first element is the attention
        # output. MLP returns a tensor directly.
        if isinstance(output, tuple):
            output = output[0]
        return output.detach().float().cpu().numpy()[0]

    def trajectory_metrics(h: np.ndarray):
        v = h[1:] - h[:-1]
        if v.shape[0] < 6:
            return None
        speeds = np.linalg.norm(v, axis=-1).astype(np.float32)
        norms = np.linalg.norm(v, axis=-1, keepdims=True)
        unit = v / np.maximum(norms, 1e-12)
        cos = np.sum(unit[:-1] * unit[1:], axis=-1)
        raw_curv = np.arccos(np.clip(cos, -1.0, 1.0)).astype(np.float32)
        return speeds, raw_curv

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

        captured: dict[tuple[int, str], np.ndarray] = {}
        handles = []
        for layer in selected_layers:
            block = blocks[layer]

            def attn_hook(_module, _inputs, output, layer=layer):
                captured[(layer, "attention_output")] = component_tensor(output)

            def mlp_hook(_module, _inputs, output, layer=layer):
                captured[(layer, "mlp_output")] = component_tensor(output)

            handles.append(block.attention.register_forward_hook(attn_hook))
            handles.append(block.mlp.register_forward_hook(mlp_hook))

        try:
            out = model(**enc, output_hidden_states=True, return_dict=True, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()

        logits = out.logits[0].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = (-(probs * log_probs).sum(dim=-1)).detach().cpu().numpy().astype(np.float32)

        for layer in selected_layers:
            trajectories = {
                "post_block_residual": out.hidden_states[layer + 1][0].float().detach().cpu().numpy(),
            }
            for source in ["attention_output", "mlp_output"]:
                key = (layer, source)
                if key in captured:
                    trajectories[source] = captured[key]

            for source, h in trajectories.items():
                metrics = trajectory_metrics(h)
                if metrics is None:
                    continue
                speeds, raw_curv = metrics
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
                        "source": source,
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
        key = (int(row["layer"]), str(row["source"]))
        groups.setdefault(key, []).append(row)

    summary = []
    for (layer, source), subset in sorted(groups.items()):
        curv = [r["contextual_curvature"] for r in subset]
        ent = [r["entropy"] for r in subset]
        speed = [r["contextual_speed"] for r in subset]
        summary.append({
            "revision": revision,
            "layer": layer,
            "source": source,
            "n": len(subset),
            "mean_curvature": float(np.mean(curv)) if curv else float("nan"),
            "mean_curvature_degrees": float(np.degrees(np.mean(curv))) if curv else float("nan"),
            "mean_entropy": float(np.mean(ent)) if ent else float("nan"),
            "mean_speed": float(np.mean(speed)) if speed else float("nan"),
            "curvature_entropy_pearson": pearson(curv, ent),
        })

    return {
        "schema_version": "pythia_component_curvature_v1",
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
         output_dir: str = "results/modal_pythia_component_curvature"):
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
