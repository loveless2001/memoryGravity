from __future__ import annotations

import argparse
import time

import torch

from mg_core import HAS_TRITON, MGConfig, MemoryGravityHead


def maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def clone_weights(src: MemoryGravityHead, dst: MemoryGravityHead) -> None:
    dst.load_state_dict(src.state_dict())


def make_cfg(base: MGConfig, backend: str) -> MGConfig:
    overrides = dict(base.__dict__)
    if backend == "slow":
        overrides["use_fast_path"] = False
        overrides["use_triton"] = False
    elif backend == "jit":
        overrides["use_fast_path"] = True
        overrides["use_triton"] = False
    elif backend == "triton":
        overrides["use_fast_path"] = True
        overrides["use_triton"] = True
    else:
        raise ValueError(f"unknown backend: {backend}")
    return MGConfig(**overrides)


def max_diff(lhs: torch.Tensor | None, rhs: torch.Tensor | None) -> float:
    if lhs is None and rhs is None:
        return 0.0
    if lhs is None or rhs is None:
        return float("inf")
    finite_mask = torch.isfinite(lhs) & torch.isfinite(rhs)
    if not torch.equal(torch.isfinite(lhs), torch.isfinite(rhs)):
        return float("inf")
    if finite_mask.any():
        return (lhs[finite_mask] - rhs[finite_mask]).abs().max().item()
    return 0.0


@torch.no_grad()
def run_once(
    head: MemoryGravityHead,
    x: torch.Tensor,
    glyph_mask: torch.Tensor | None,
    *,
    return_mass: bool,
    return_scores: bool,
):
    y, attn, mass, scores = head(
        x,
        glyph_mask=glyph_mask,
        return_attn=True,
        return_mass=return_mass,
        return_scores=return_scores,
    )
    return {
        "y": y,
        "attn": attn,
        "mass": mass,
        "scores": scores,
    }


@torch.no_grad()
def benchmark_forward(
    head: MemoryGravityHead,
    x: torch.Tensor,
    glyph_mask: torch.Tensor | None,
    *,
    iters: int,
    warmup: int,
    return_mass: bool,
    return_scores: bool,
) -> float:
    for _ in range(warmup):
        _ = head(
            x,
            glyph_mask=glyph_mask,
            return_attn=False,
            return_mass=return_mass,
            return_scores=return_scores,
        )
    maybe_sync(x.device)

    start = time.perf_counter()
    for _ in range(iters):
        _ = head(
            x,
            glyph_mask=glyph_mask,
            return_attn=False,
            return_mass=return_mass,
            return_scores=return_scores,
        )
    maybe_sync(x.device)
    elapsed = time.perf_counter() - start
    return 1000.0 * elapsed / max(iters, 1)


def backend_available(backend: str, device: torch.device) -> bool:
    if backend != "triton":
        return True
    return device.type == "cuda" and HAS_TRITON


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare mg_core backends")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=1)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no-glyphs", action="store_true")
    parser.add_argument("--skip-mass", action="store_true")
    parser.add_argument("--skip-scores", action="store_true")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["slow", "jit", "triton"],
        choices=["slow", "jit", "triton"],
        help="Backends to run. Triton is skipped automatically if unavailable.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    cfg = MGConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        max_seq_len=args.seq_len,
        use_mass_weighting=True,
        use_glyphs=not args.no_glyphs,
    )
    requested_backends = [backend for backend in args.backends if backend_available(backend, device)]
    if not requested_backends:
        raise RuntimeError("no requested backends are available on this device")
    if "slow" not in requested_backends:
        requested_backends = ["slow", *requested_backends]

    heads: dict[str, MemoryGravityHead] = {}
    reference_head = MemoryGravityHead(make_cfg(cfg, "slow")).to(device)
    reference_head.eval()
    heads["slow"] = reference_head

    for backend in requested_backends:
        if backend == "slow":
            continue
        head = MemoryGravityHead(make_cfg(cfg, backend)).to(device)
        clone_weights(reference_head, head)
        head.eval()
        heads[backend] = head

    x = torch.randn(args.batch_size, args.seq_len, args.d_model, device=device)
    glyph_mask = None
    if not args.no_glyphs:
        glyph_mask = torch.randint(0, 2, (args.batch_size, args.seq_len), device=device).to(torch.float32)

    return_mass = not args.skip_mass
    return_scores = not args.skip_scores

    print(f"Device: {device.type}")
    print(f"Shape: B={args.batch_size} T={args.seq_len} D={args.d_model} H={args.n_heads}")
    print(f"Requested backends: {', '.join(requested_backends)}")

    outputs = {
        backend: run_once(
            head,
            x,
            glyph_mask,
            return_mass=return_mass,
            return_scores=return_scores,
        )
        for backend, head in heads.items()
    }
    print("Parity vs slow")
    for backend in requested_backends:
        ref = outputs["slow"]
        out = outputs[backend]
        print(
            f"  {backend:<6} y={max_diff(ref['y'], out['y']):.6e} "
            f"attn={max_diff(ref['attn'], out['attn']):.6e} "
            f"mass={max_diff(ref['mass'], out['mass']):.6e} "
            f"scores={max_diff(ref['scores'], out['scores']):.6e}"
        )

    timings = {}
    for backend, head in heads.items():
        timings[backend] = benchmark_forward(
            head,
            x,
            glyph_mask,
            iters=args.iters,
            warmup=args.warmup,
            return_mass=return_mass,
            return_scores=return_scores,
        )

    print("Timing")
    slow_ms = timings["slow"]
    for backend in requested_backends:
        backend_ms = timings[backend]
        line = f"  {backend:<6}: {backend_ms:.3f} ms"
        if backend != "slow" and backend_ms > 0:
            line += f" | speedup vs slow: {slow_ms / backend_ms:.3f}x"
        print(line)


if __name__ == "__main__":
    main()
