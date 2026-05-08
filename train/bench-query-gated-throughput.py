#!/usr/bin/env python
"""Benchmark query-gated MG throughput: measures tok/s for training steps."""
import sys
import time

import torch

sys.path.insert(0, "/home/lenovo/projects/memoryGravity")
from train.mg_core import MGConfig, TinyMemoryGravityLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 256
BATCH = 16
WARMUP = 10
STEPS = 50
VOCAB = 128


def make_random_batch(batch_size, seq_len, device):
    """Random tokens + glyph mask for benchmarking (no arena data gen overhead)."""
    idx = torch.randint(0, VOCAB, (batch_size, seq_len), device=device)
    glyph_mask = (torch.rand(batch_size, seq_len, device=device) > 0.8).float()
    return idx, glyph_mask


def bench(label: str, compile_model: bool = False):
    cfg = MGConfig(
        vocab_size=VOCAB,
        d_model=64,
        hidden_dim=128,
        max_seq_len=SEQ_LEN,
        n_heads=4,
        n_layers=2,
        mass_mode="query_gated",
        local_window=64,
        use_triton=False,
    )
    model = TinyMemoryGravityLM(cfg).to(DEVICE)
    if compile_model and DEVICE == "cuda":
        model = torch.compile(model)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Warmup
    for _ in range(WARMUP):
        idx, gm = make_random_batch(BATCH, SEQ_LEN, DEVICE)
        out = model(idx, glyph_mask=gm, targets=idx, return_attn=False, return_mass=False)
        out["loss"].backward()
        opt.step()
        opt.zero_grad()

    # Timed steps
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    total_tokens = 0

    for _ in range(STEPS):
        idx, gm = make_random_batch(BATCH, SEQ_LEN, DEVICE)
        out = model(idx, glyph_mask=gm, targets=idx, return_attn=False, return_mass=False)
        out["loss"].backward()
        opt.step()
        opt.zero_grad()
        total_tokens += BATCH * SEQ_LEN

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    tok_s = total_tokens / elapsed
    print(f"{label}: {tok_s:.0f} tok/s ({elapsed:.1f}s for {STEPS} steps, device={DEVICE})")
    return tok_s


if __name__ == "__main__":
    print(f"Device: {DEVICE}, seq_len={SEQ_LEN}, batch={BATCH}")
    base = bench("optimized (precomputed gates)")
    if DEVICE == "cuda":
        compiled = bench("torch.compile", compile_model=True)
        print(f"\nSpeedup from compile: {compiled/base:.2f}x")
