"""
Phase 1 Trace Extractor for the Dynamic Semantic Trajectory Visualizer.

Loads a TinyStories-33M (GPT-Neo) checkpoint, runs a single prompt forward,
captures residual stream hidden states for selected layers, and writes a
trace to disk that conforms to the locked artifact contract in
`memoryGravity/plans/dynamic_semantic_trajectory_visualizer.md`.

The output is consumed by:
- `geometry.py` (speed / curvature / null calibration)
- `pre_mvp_geometry_entropy_spike.py` (Phase 0 falsification spike)
- future `intervene.py` (Phase 4 Behavioral Validation, owned by codex)

Schema fields written here are the non-optional parts of the contract;
optional Memory Gravity overlays (anchor_strength, clpg, adm) are added by
downstream tools.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Default base model used for tokenizer + architecture skeleton.
# All checkpoints in `memoryGravity/checkpoints/` are TinyStories-33M state dicts.
BASE_MODEL_ID = "roneneldan/TinyStories-33M"
LAYER_NORM_CONVENTION = "post_block_residual"  # We capture residual stream after each transformer block.
SCHEMA_VERSION = "trace_v1"  # Stable across Phase 0/0.5/3/4 outputs.


@dataclass
class TraceMetadata:
    """Mirrors the `trace.json` schema in the plan."""
    model_id: str
    layer_indices: list[int]
    tokenizer_id: str
    prompt: str
    token_ids: list[int]
    token_strings: list[str]
    layer_norm_convention: str
    null_baseline_method: str
    seed: int
    prompt_family: str
    metric_overlays: list[str]
    schema_version: str = SCHEMA_VERSION


def load_model(checkpoint_path: str | None,
               base_model_id: str = BASE_MODEL_ID,
               device: str | None = None):
    """Load tokenizer + model, optionally overlaying a local fine-tuned checkpoint.

    Mirrors the loading pattern used in `scan/check_trigger_clpg.py` and
    related scripts: load the HF base, then load_state_dict from a `.pt`,
    handling the `{"model": ...}` wrapper variant.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model_id)
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            ckpt = ckpt["model"]
        model.load_state_dict(ckpt)
    model.to(device).eval()
    return model, tokenizer, device


def _select_residual_stream(hidden_states: tuple[torch.Tensor, ...],
                            layer_indices: Iterable[int]) -> torch.Tensor:
    """Extract residual stream after each requested block.

    `hidden_states` from HF GPT-Neo is a tuple of length n_layers+1:
    index 0 is the embedding output, index i+1 is the residual stream
    after block i. We take indices i+1 for each requested layer i, so the
    saved tensor's first index aligns with `layer_indices`.
    """
    selected = [hidden_states[i + 1] for i in layer_indices]  # each: (1, T, d)
    stacked = torch.stack(selected, dim=1)  # (1, n_layers_selected, T, d)
    stacked = stacked.squeeze(0).transpose(0, 1).contiguous()  # (T, n_layers_selected, d)
    return stacked


@torch.no_grad()
def run_prompt(model, tokenizer, device: str, prompt: str,
               layer_indices: Iterable[int],
               topk: int = 32) -> dict:
    """Forward `prompt` once and return per-token hidden states + decode signals.

    Returns a dict with arrays already converted to numpy with the dtypes
    specified by the artifact contract:
      hidden_states: float16 (T, n_layers_selected, d)
      entropy:       float32 (T,)
      logit_margin:  float32 (T,)
      logits_topk:   float16 (T, k)   (top-k log-probs for compactness)
      topk_indices:  int32   (T, k)
      token_ids:     int     (T,)
      token_strings: list[str]

    `entropy[t]` and `logit_margin[t]` describe the next-token distribution
    *predicted from* position t, i.e. derived from `logits[t]`. For a prompt
    of T input tokens the model produces T output distributions, so all
    arrays have length T.
    """
    layer_indices = list(layer_indices)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc.input_ids  # (1, T)

    out = model(input_ids, output_hidden_states=True, return_dict=True)
    logits = out.logits[0]  # (T, V)
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()

    entropy = -(probs * log_probs).sum(dim=-1)  # (T,)
    top2 = torch.topk(logits, k=2, dim=-1).values  # (T, 2)
    logit_margin = top2[:, 0] - top2[:, 1]  # (T,)

    topk_log_probs, topk_idx = torch.topk(log_probs, k=topk, dim=-1)  # both (T, k)

    hidden_stream = _select_residual_stream(out.hidden_states, layer_indices)  # (T, L, d)

    token_ids = input_ids[0].tolist()
    token_strings = [tokenizer.decode([t]) for t in token_ids]

    return {
        "hidden_states": hidden_stream.to(torch.float16).cpu().numpy(),
        "entropy": entropy.to(torch.float32).cpu().numpy(),
        "logit_margin": logit_margin.to(torch.float32).cpu().numpy(),
        "logits_topk": topk_log_probs.to(torch.float16).cpu().numpy(),
        "topk_indices": topk_idx.to(torch.int32).cpu().numpy(),
        "token_ids": token_ids,
        "token_strings": token_strings,
    }


def save_trace(out_dir: str | Path, name: str, run_data: dict,
               geom: dict | None,
               metadata: TraceMetadata) -> tuple[Path, Path]:
    """Write `<name>.npz` and `<name>.json` to `out_dir`.

    `run_data` carries the forward-pass arrays from `run_prompt`. `geom` may
    carry geometry-derived arrays (step_speeds, curvatures_q, stall_mask)
    when the geometry pipeline has been run; otherwise those fields are
    omitted from the npz and downstream consumers can recompute them.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{name}.npz"
    json_path = out_dir / f"{name}.json"

    npz_payload = {
        "hidden_states": run_data["hidden_states"],
        "entropy": run_data["entropy"],
        "logit_margin": run_data["logit_margin"],
        "logits_topk": run_data["logits_topk"],
        "topk_indices": run_data["topk_indices"],
    }
    if geom is not None:
        for key in ("step_speeds", "curvatures_q", "stall_mask"):
            if key in geom:
                npz_payload[key] = geom[key]

    np.savez_compressed(npz_path, **npz_payload)
    with json_path.open("w") as f:
        json.dump(asdict(metadata), f, indent=2)
    return npz_path, json_path
