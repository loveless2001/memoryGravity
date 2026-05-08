import torch
import time
import sys
import os

# Set up path to import mg_core and mg_triton
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mg_core import MGConfig, MemoryGravityHead, mg_head_loop
from mg_triton import mg_head_triton

def benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Triton requires CUDA. Skipping.")
        return

    print(f"Device: {device}")

    B, H, T, D = 8, 4, 1024, 64
    d_h = D // H
    alpha, lambda_mass = 0.95, 1.0

    base_scores = torch.randn(B, H, T, T, device=device)
    deposit_rate = torch.ones(B, 1, T, device=device)

    # Warmup JIT
    for _ in range(5):
        _ = mg_head_loop(base_scores, deposit_rate, alpha, lambda_mass, True, False, True, True)
        torch.cuda.synchronize()

    # JIT timing
    start = time.time()
    for _ in range(20):
        _ = mg_head_loop(base_scores, deposit_rate, alpha, lambda_mass, True, False, True, True)
        torch.cuda.synchronize()
    end = time.time()
    jit_ms = (end - start) / 20 * 1000
    print(f"JIT Loop: {jit_ms:.2f} ms")

    # Warmup Triton
    for _ in range(5):
        _ = mg_head_triton(base_scores, deposit_rate, alpha, lambda_mass, True, False, True, True)
        torch.cuda.synchronize()

    # Triton timing
    start = time.time()
    for _ in range(20):
        _ = mg_head_triton(base_scores, deposit_rate, alpha, lambda_mass, True, False, True, True)
        torch.cuda.synchronize()
    end = time.time()
    triton_ms = (end - start) / 20 * 1000
    print(f"Triton Kernel: {triton_ms:.2f} ms")
    print(f"Speedup: {jit_ms / triton_ms:.2f}x")

    # Parity check
    eff_jit, mh_jit, so_jit = mg_head_loop(base_scores, deposit_rate, alpha, lambda_mass, True, False, True, True)
    eff_tr, mh_tr, so_tr = mg_head_triton(base_scores, deposit_rate, alpha, lambda_mass, True, False, True, True)

    print(f"Effective Diff: {(eff_jit - eff_tr).abs().max().item():.2e}")
    print(f"Mass Hist Diff: {(mh_jit - mh_tr).abs().max().item():.2e}")
    # Handle -inf in scores
    mask = torch.isfinite(so_jit)
    if mask.any():
        print(f"Scores Out Diff: {(so_jit[mask] - so_tr[mask]).abs().max().item():.2e}")
    else:
        print("Scores Out: No finite values to compare")

if __name__ == "__main__":
    benchmark()
