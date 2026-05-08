from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mg_moe import (
    GovernorState,
    compute_phi_utility_batch,
    freeze_backbone_except_mgmoe,
    inject_mgmoe_into_ffn,
    mask_grads_for_nonlearning_experts,
    mg_blocks,
    mg_post_step,
    reduce_signals,
)


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "none":
        return nullcontext()
    if amp == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def load_corpus(corpus_path: str) -> str:
    if not os.path.exists(corpus_path):
        return "Memory is not content retrieval; it is persistent manifold curvature. " * 50
    with open(corpus_path, "r", encoding="utf-8") as f:
        return f.read()


def gpu_memory_gb(device: torch.device) -> tuple[float, float] | None:
    if device.type != "cuda":
        return None
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    return allocated, reserved


def save_mg_state(model: torch.nn.Module, path: str) -> None:
    mg_state_dict = {
        name: param
        for name, param in model.named_parameters()
        if "experts" in name or "router" in name
    }
    torch.save(mg_state_dict, path)


def train_mg_moe_efficient(args: argparse.Namespace) -> dict:
    print("--- MG-MoE Efficient Terraforming Protocol ---")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    if args.amp == "fp16":
        model_dtype = torch.float16
    elif args.amp == "bf16" and device.type == "cuda":
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float32

    print(f"Loading {args.model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=model_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    gov_config = GovernorState(
        m_min_learn=args.m_min_learn,
        birth_cooldown=args.birth_cooldown,
        saturation_eps=args.saturation_eps,
    )
    model, block_count = inject_mgmoe_into_ffn(
        model,
        d_model=model.config.hidden_size,
        every_n_layers=args.every_n_layers,
        governor=gov_config,
    )
    model.to(device=device, dtype=model_dtype)
    print(f"Injected {block_count} MG-MoE Blocks.")

    model = freeze_backbone_except_mgmoe(model)
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()

    text = load_corpus(args.corpus_path)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    ).to(device)
    labels = inputs.input_ids.clone()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    use_scaler = device.type == "cuda" and args.amp == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    model.train()

    print(
        f"Beginning efficient terraforming on {device.type} "
        f"(amp={args.amp}, grad_accum={args.grad_accum_steps}, max_length={args.max_length})..."
    )

    metrics: list[dict] = []
    optimizer.zero_grad(set_to_none=True)
    run_start = time.perf_counter()
    last_log_time = run_start
    tokens_per_step = int(inputs.input_ids.numel())

    for step in range(1, args.steps + 1):
        amp_ctx = autocast_context(device, args.amp)
        with amp_ctx:
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            scaled_loss = loss / args.grad_accum_steps

        phi, u, mask = compute_phi_utility_batch(
            logits=logits,
            input_ids=inputs.input_ids,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100,
            W=args.window_size,
        )
        phi_mean, u_mean = reduce_signals(phi, u, mask)

        if use_scaler:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if (step % args.grad_accum_steps) == 0:
            if args.grad_clip_norm > 0:
                if use_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

            mask_grads_for_nonlearning_experts(model)
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            mg_post_step(
                model,
                utility_mean=u_mean,
                phi_mean=phi_mean,
                step=step,
                gov_interval=args.gov_interval,
            )

        if step % args.log_interval == 0 or step == 1 or step == args.steps:
            now = time.perf_counter()
            elapsed = now - last_log_time
            last_log_time = now
            toks_per_sec = (tokens_per_step * args.log_interval) / max(elapsed, 1e-8)
            blocks = list(mg_blocks(model))
            avg_mass = 0.0
            total_experts = 0
            if blocks:
                avg_mass = sum(sum(e.mass.item() for e in block.experts) for block in blocks) / max(block_count, 1)
                total_experts = sum(len(block.experts) for block in blocks)

            entry = {
                "step": step,
                "loss": float(loss.item()),
                "phi_mean": phi_mean,
                "utility_mean": u_mean,
                "tokens_per_sec": toks_per_sec,
                "avg_mass": avg_mass,
                "total_experts": total_experts,
            }
            mem_stats = gpu_memory_gb(device)
            if mem_stats is not None:
                entry["gpu_mem_alloc_gb"] = mem_stats[0]
                entry["gpu_mem_reserved_gb"] = mem_stats[1]

            metrics.append(entry)
            mem_suffix = ""
            if mem_stats is not None:
                mem_suffix = (
                    f" | GPU alloc/resv: {mem_stats[0]:.2f}/{mem_stats[1]:.2f} GB"
                )
            print(
                f"[Step {step:04}] Loss: {entry['loss']:.4f} | Phi: {phi_mean:.3f} | "
                f"Util: {u_mean:.3f} | Tok/s: {toks_per_sec:.0f} | "
                f"AvgMass: {avg_mass:.2f} | Experts: {total_experts}{mem_suffix}"
            )

        if args.save_every > 0 and step % args.save_every == 0:
            save_mg_state(model, args.save_path)

    total_seconds = time.perf_counter() - run_start
    save_mg_state(model, args.save_path)
    print(f"Terraforming Complete. Experts saved to {args.save_path}")

    summary = {
        "model_id": args.model_id,
        "device": args.device,
        "amp": args.amp,
        "steps": args.steps,
        "grad_accum_steps": args.grad_accum_steps,
        "max_length": args.max_length,
        "block_count": block_count,
        "total_seconds": total_seconds,
        "tokens_total": tokens_per_step * args.steps,
        "tokens_per_sec_mean": (tokens_per_step * args.steps) / max(total_seconds, 1e-8),
        "metrics": metrics,
    }
    if args.metrics_json:
        with open(args.metrics_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Metrics written to {args.metrics_json}")

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Efficient MG-MoE training with monitoring")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--corpus-path", default="memoryGravity/docs/GRAVITY_MANIFESTO.md")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", choices=["none", "fp16", "bf16"], default="bf16")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--every-n-layers", type=int, default=4)
    parser.add_argument("--gov-interval", type=int, default=10)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--save-path", default="terraform_mg_moe_qwen1.5b_efficient.pt")
    parser.add_argument("--metrics-json", default="")
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--m-min-learn", type=float, default=0.5)
    parser.add_argument("--birth-cooldown", type=int, default=50)
    parser.add_argument("--saturation-eps", type=float, default=0.1)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    train_mg_moe_efficient(args)


if __name__ == "__main__":
    main()
