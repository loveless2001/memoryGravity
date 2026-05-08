#!/usr/bin/env python
"""Step 1 benchmark: eager additive vs eager query_gated vs baseline.

torch.compile is tested separately since the sequential loop causes very long
compilation times — that finding itself informs the Step 1 decision.

Measures forward+backward throughput on the Selective-Reach v1 config.
"""
from __future__ import annotations

import argparse
import time
import torch
from train.mg_core import MGConfig, TinyMemoryGravityLM

SEQ_LEN = 128
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
BATCH_SIZE = 64
WARMUP = 5
ITERS = 20


def make_batch(cfg: MGConfig, device: str):
    idx = torch.randint(0, cfg.vocab_size, (BATCH_SIZE, SEQ_LEN), device=device)
    glyph_mask = (idx == 7).float()
    return idx, glyph_mask


def bench_forward_backward(model, idx, glyph_mask, label: str, device: str):
    # Warmup
    for _ in range(WARMUP):
        out = model(idx, glyph_mask=glyph_mask, targets=idx, return_attn=False, return_mass=False)
        out["loss"].backward()
        model.zero_grad(set_to_none=True)

    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(ITERS):
        out = model(idx, glyph_mask=glyph_mask, targets=idx, return_attn=False, return_mass=False)
        out["loss"].backward()
        model.zero_grad(set_to_none=True)

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0
    ms_per_iter = (elapsed / ITERS) * 1000
    tok_per_sec = (ITERS * BATCH_SIZE * SEQ_LEN) / elapsed
    print(f"  {label:40s} | {ms_per_iter:8.2f} ms/iter | {tok_per_sec:12,.0f} tok/s")
    return ms_per_iter, tok_per_sec


def make_model(mass_mode: str, device: str, local_window: int = 0) -> TinyMemoryGravityLM:
    cfg = MGConfig(
        vocab_size=128, d_model=D_MODEL, hidden_dim=D_MODEL * 2,
        max_seq_len=SEQ_LEN, n_heads=N_HEADS, n_layers=N_LAYERS,
        alpha=0.95, lambda_mass=0.5, glyph_deposit=2.0,
        use_mass_weighting=True, use_glyphs=True,
        mass_mode=mass_mode, local_window=local_window,
        use_triton=False, use_fast_path=True,
    )
    if mass_mode == "none":
        cfg.use_mass_weighting = False
        cfg.use_glyphs = False
    return TinyMemoryGravityLM(cfg).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="Also test torch.compile (slow compilation)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    print(f"Benchmark: device={device}, seq_len={SEQ_LEN}, batch={BATCH_SIZE}, "
          f"d_model={D_MODEL}, n_heads={N_HEADS}, n_layers={N_LAYERS}")
    print(f"  warmup={WARMUP}, iters={ITERS}")
    print("-" * 80)

    results = {}

    # 1. Baseline (no mass) for reference
    model_base = make_model("none", device)
    idx, glyph = make_batch(model_base.cfg, device)
    ms, tps = bench_forward_backward(model_base, idx, glyph, "baseline (no mass)", device)
    results["baseline"] = (ms, tps)
    del model_base

    # 2. Additive (old path via jit-scripted mg_head_loop)
    model_add = make_model("additive", device)
    idx, glyph = make_batch(model_add.cfg, device)
    ms, tps = bench_forward_backward(model_add, idx, glyph, "additive (jit, eager)", device)
    results["additive_eager"] = (ms, tps)
    del model_add

    # 3. Query-gated (new vectorized loop, eager)
    model_qg = make_model("query_gated", device, local_window=64)
    idx, glyph = make_batch(model_qg.cfg, device)
    ms, tps = bench_forward_backward(model_qg, idx, glyph, "query_gated_local (eager)", device)
    results["qg_eager"] = (ms, tps)
    del model_qg

    # 4. Query-gated without local window
    model_qg_full = make_model("query_gated", device, local_window=0)
    idx, glyph = make_batch(model_qg_full.cfg, device)
    ms, tps = bench_forward_backward(model_qg_full, idx, glyph, "query_gated_full (eager)", device)
    results["qg_full_eager"] = (ms, tps)
    del model_qg_full

    # 5. Optional: torch.compile
    if args.compile:
        print("\n  [torch.compile] Compiling query_gated_local — this may take minutes...")
        model_qg_c = make_model("query_gated", device, local_window=64)
        model_qg_c = torch.compile(model_qg_c, fullgraph=False)
        idx, glyph = make_batch(
            model_qg_c._orig_mod.cfg if hasattr(model_qg_c, '_orig_mod') else model_qg_c.cfg,
            device,
        )
        ms, tps = bench_forward_backward(model_qg_c, idx, glyph, "query_gated_local (torch.compile)", device)
        results["qg_compile"] = (ms, tps)
        del model_qg_c

    print("-" * 80)
    print("Summary:")
    add_ms = results["additive_eager"][0]
    for label, (ms, tps) in results.items():
        speedup = add_ms / ms if ms > 0 else 0
        print(f"  {label:40s} | {ms:8.2f} ms | {speedup:.2f}x vs additive")


if __name__ == "__main__":
    main()
