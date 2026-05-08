#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import time
import urllib.request
from dataclasses import asdict
from contextlib import nullcontext

import torch

from train.mg_core import HAS_TRITON, MGConfig, TinyMemoryGravityLM


TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def maybe_download_tiny_shakespeare(path: pathlib.Path, auto_download: bool) -> None:
    if path.exists():
        return
    if not auto_download:
        raise FileNotFoundError(
            f"Tiny Shakespeare not found at {path}. Pass --auto-download to fetch it."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Tiny Shakespeare to {path} ...")
    urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)


def load_text(path: pathlib.Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_vocab(text: str):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode_text(text: str, stoi: dict[str, int]) -> torch.Tensor:
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)


def make_batch(
    data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    glyph_id: int | None,
    device: torch.device,
    generator: torch.Generator,
):
    max_start = data.size(0) - seq_len - 1
    if max_start <= 0:
        raise ValueError("dataset too small for chosen seq_len")

    starts = torch.randint(0, max_start, (batch_size,), generator=generator)
    idx = torch.stack([data[s : s + seq_len] for s in starts])

    glyph_mask = None
    if glyph_id is not None:
        glyph_mask = (idx == glyph_id).float()

    return idx.to(device), None if glyph_mask is None else glyph_mask.to(device)


@torch.no_grad()
def evaluate(model, data, args, device, glyph_id, eval_generator):
    model.eval()
    losses = []
    acc_correct = 0
    acc_total = 0

    for _ in range(args.eval_batches):
        idx, glyph_mask = make_batch(
            data=data,
            batch_size=args.eval_batch_size,
            seq_len=args.seq_len,
            glyph_id=glyph_id,
            device=device,
            generator=eval_generator,
        )
        amp_ctx = autocast_context(args, device)
        with amp_ctx:
            out = model(idx, glyph_mask=glyph_mask, targets=idx, return_attn=False, return_mass=False)
        losses.append(out["loss"].item())

        preds = out["logits"][:, :-1, :].argmax(dim=-1)
        targets = idx[:, 1:]
        acc_correct += (preds == targets).sum().item()
        acc_total += targets.numel()

    val_loss = sum(losses) / max(len(losses), 1)
    return {
        "val_loss": val_loss,
        "val_ppl": math.exp(val_loss),
        "next_char_acc": acc_correct / max(acc_total, 1),
    }


def autocast_context(args, device: torch.device):
    if device.type != "cuda" or args.amp == "none":
        return nullcontext()
    if args.amp == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    # default to bf16 for stability on modern GPUs
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def select_backend(cfg: MGConfig, device: torch.device, backend: str) -> str:
    if backend == "slow":
        cfg.use_fast_path = False
        cfg.use_triton = False
        return "slow"
    if backend == "jit":
        cfg.use_fast_path = True
        cfg.use_triton = False
        return "jit"
    if backend == "triton":
        if device.type != "cuda":
            raise ValueError("backend=triton requires a CUDA device")
        if not HAS_TRITON:
            raise ValueError("backend=triton requested but Triton is not available")
        cfg.use_fast_path = True
        cfg.use_triton = True
        return "triton"

    if device.type == "cuda" and HAS_TRITON and cfg.use_triton:
        return "triton"
    if cfg.use_fast_path:
        return "jit"
    return "slow"


def train_once(args, use_mass_weighting: bool, train_data, val_data, glyph_id):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)

    cfg = MGConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        hidden_dim=args.hidden_dim,
        max_seq_len=args.seq_len,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        alpha=args.alpha,
        beta=args.beta,
        glyph_boost=args.glyph_boost,
        use_mass_weighting=use_mass_weighting,
        use_glyphs=(glyph_id is not None),
        mass_to_logits=args.mass_to_logits,
    )
    backend = select_backend(cfg, device, args.backend)

    model = TinyMemoryGravityLM(cfg).to(device)
    if args.compile:
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_scaler = (device.type == "cuda" and args.amp == "fp16")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    train_generator = torch.Generator(device="cpu").manual_seed(args.seed + 11)
    eval_generator = torch.Generator(device="cpu").manual_seed(args.seed + 22)

    model.train()
    last_train_loss = None
    t0 = time.perf_counter()
    for step in range(1, args.max_steps + 1):
        idx, glyph_mask = make_batch(
            data=train_data,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            glyph_id=glyph_id,
            device=device,
            generator=train_generator,
        )
        amp_ctx = autocast_context(args, device)
        with amp_ctx:
            out = model(idx, glyph_mask=glyph_mask, targets=idx, return_attn=False, return_mass=False)
            loss = out["loss"]
        last_train_loss = loss.item()

        optimizer.zero_grad(set_to_none=True)
        if use_scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
    train_seconds = time.perf_counter() - t0

    metrics = evaluate(
        model=model,
        data=val_data,
        args=args,
        device=device,
        glyph_id=glyph_id,
        eval_generator=eval_generator,
    )
    metrics["train_loss_last"] = last_train_loss
    metrics["train_steps"] = args.max_steps
    metrics["train_seconds"] = train_seconds
    metrics["train_tokens_per_sec"] = (args.max_steps * args.batch_size * args.seq_len) / max(train_seconds, 1e-8)
    metrics["use_mass_weighting"] = use_mass_weighting
    metrics["backend"] = backend
    metrics["config"] = asdict(cfg)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Toy runner for train/mg_core.py on Tiny Shakespeare")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--tiny-shakespeare-path", default="data/tinyshakespeare/input.txt")
    parser.add_argument("--auto-download", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.9)

    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=1)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--eval-batches", type=int, default=40)

    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--glyph-boost", type=float, default=2.0)
    parser.add_argument("--glyph-char", default="")
    parser.add_argument("--mass-to-logits", action="store_true")
    parser.add_argument("--backend", choices=["auto", "slow", "jit", "triton"], default="auto")
    parser.add_argument("--amp", choices=["none", "fp16", "bf16"], default="bf16")
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--only", choices=["true", "false", "both"], default="both")
    parser.add_argument("--json-out", default="")

    args = parser.parse_args()

    data_path = pathlib.Path(args.tiny_shakespeare_path)
    maybe_download_tiny_shakespeare(data_path, auto_download=args.auto_download)
    text = load_text(data_path)

    stoi, _ = build_vocab(text)
    encoded = encode_text(text, stoi)
    args.vocab_size = len(stoi)

    split = int(encoded.size(0) * args.train_ratio)
    train_data = encoded[:split]
    val_data = encoded[split:]

    glyph_id = stoi.get(args.glyph_char) if args.glyph_char else None

    print(f"Device: {args.device}")
    if args.device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")
    print(f"Backend request: {args.backend}")
    print(f"Dataset chars: {encoded.size(0)}, vocab: {args.vocab_size}, train: {train_data.size(0)}, val: {val_data.size(0)}")
    if glyph_id is not None:
        print(f"Using glyph char '{args.glyph_char}' with id {glyph_id}")
    else:
        print("Glyph masking disabled (no --glyph-char)")

    runs = []
    if args.only in ("true", "both"):
        runs.append(train_once(args, use_mass_weighting=True, train_data=train_data, val_data=val_data, glyph_id=glyph_id))
    if args.only in ("false", "both"):
        runs.append(train_once(args, use_mass_weighting=False, train_data=train_data, val_data=val_data, glyph_id=glyph_id))

    runs = sorted(runs, key=lambda r: not r["use_mass_weighting"])

    print("\nTiny Shakespeare MG core comparison")
    print("=" * 92)
    print(
        f"{'use_mass_weighting':<20} {'val_loss':<12} {'val_ppl':<12} "
        f"{'next_char_acc':<14} {'tok/s':<10} {'steps':<8}"
    )
    for r in runs:
        print(
            f"{str(r['use_mass_weighting']):<20} "
            f"{r['val_loss']:<12.6f} "
            f"{r['val_ppl']:<12.3f} "
            f"{r['next_char_acc']:<14.4f} "
            f"{r['train_tokens_per_sec']:<10.0f} "
            f"{r['train_steps']:<8}"
        )
        print(f"  backend={r['backend']}")

    if len(runs) == 2:
        true_run = next(r for r in runs if r["use_mass_weighting"])
        false_run = next(r for r in runs if not r["use_mass_weighting"])
        print("\nDelta (True - False)")
        print("=" * 92)
        print(f"val_loss delta     : {true_run['val_loss'] - false_run['val_loss']:+.6f}")
        print(f"val_ppl delta      : {true_run['val_ppl'] - false_run['val_ppl']:+.3f}")
        print(f"next_char_acc delta: {true_run['next_char_acc'] - false_run['next_char_acc']:+.4f}")

    if args.json_out:
        out_dir = os.path.dirname(args.json_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(runs, f, indent=2)
        print(f"\nSaved JSON results to: {args.json_out}")


if __name__ == "__main__":
    main()
