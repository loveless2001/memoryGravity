#!/usr/bin/env python3
"""
MG-CRI (Mass-Governed Continuous Regime Interpolation) — HF causal LM scaffold

Implements:
  - ICL (activations) implicitly (normal prompt conditioning)
  - Fast low-rank adaptation (LoRA-fast) with short half-life (decay)
  - Slow test-time training (LoRA-slow) with long half-life (decay)
  - Continuous gates from a phase dial phi_t
  - Hysteresis to prevent flapping
  - Budgeted updates + approximate KL trust-region cap

Safe defaults:
  - Adapt only on the prompt (teacher forcing)
  - No self-training on generated tokens unless enabled

Tested conceptually with Llama/Mistral-like models (q_proj/k_proj/v_proj/o_proj etc.).
You may need to adjust target module name patterns for your model.

Requires: torch, transformers
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ----------------------------
# Utilities
# ----------------------------

def ema_update(x_ema: float, x: float, beta: float) -> float:
    return beta * x_ema + (1 - beta) * x

def clip_norm_(params: Iterable[torch.nn.Parameter], max_norm: float) -> float:
    """Clips global norm of gradients in-place. Returns pre-clip norm."""
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    total_norm = torch.norm(torch.stack([g.detach().norm(2) for g in grads]), 2).item()
    if total_norm > max_norm > 0:
        scale = max_norm / (total_norm + 1e-12)
        for g in grads:
            g.mul_(scale)
    return total_norm

def sigmoid_gate(x: float, k: float) -> float:
    return 1.0 / (1.0 + math.exp(-k * x))

def topk_kl(p_logits: torch.Tensor, q_logits: torch.Tensor, k: int = 128) -> torch.Tensor:
    """
    Approximate KL(p||q) using top-k of p.
    p_logits, q_logits: [V]
    """
    with torch.no_grad():
        topk_vals, topk_idx = torch.topk(p_logits, k=min(k, p_logits.numel()))
    p = F.softmax(p_logits[topk_idx], dim=-1)
    q = F.softmax(q_logits[topk_idx], dim=-1)
    return torch.sum(p * (torch.log(p + 1e-12) - torch.log(q + 1e-12)))


# ----------------------------
# Dual-LoRA module
# ----------------------------

class DualLoRALinear(nn.Module):
    """
    Wraps a base Linear with two additive low-rank adapters:
      - fast: short half-life, higher responsiveness
      - slow: long half-life, gated by sustained surprise

    Forward:
      y = W x + scale_fast * Bf(Af(x)) + scale_slow * Bs(As(x))
    """
    def __init__(
        self,
        base: nn.Linear,
        r_fast: int = 8,
        r_slow: int = 8,
        alpha_fast: float = 16.0,
        alpha_slow: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features

        self.r_fast = r_fast
        self.r_slow = r_slow

        self.scale_fast = alpha_fast / max(1, r_fast)
        self.scale_slow = alpha_slow / max(1, r_slow)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Fast adapter
        self.A_fast = nn.Linear(self.in_features, r_fast, bias=False)
        self.B_fast = nn.Linear(r_fast, self.out_features, bias=False)

        # Slow adapter
        self.A_slow = nn.Linear(self.in_features, r_slow, bias=False)
        self.B_slow = nn.Linear(r_slow, self.out_features, bias=False)

        # Init: LoRA convention (A random, B zeros) so initial delta ~ 0
        nn.init.kaiming_uniform_(self.A_fast.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B_fast.weight)
        nn.init.kaiming_uniform_(self.A_slow.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B_slow.weight)

        # Runtime mixing weights (set each step by controller)
        self.mix_fast = 0.0
        self.mix_slow = 0.0

    def set_mix(self, mix_fast: float, mix_slow: float) -> None:
        self.mix_fast = float(mix_fast)
        self.mix_slow = float(mix_slow)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        x_d = self.dropout(x)

        if self.mix_fast != 0.0:
            y = y + (self.mix_fast * self.scale_fast) * self.B_fast(self.A_fast(x_d))
        if self.mix_slow != 0.0:
            y = y + (self.mix_slow * self.scale_slow) * self.B_slow(self.A_slow(x_d))
        return y

    def fast_params(self) -> List[nn.Parameter]:
        return list(self.A_fast.parameters()) + list(self.B_fast.parameters())

    def slow_params(self) -> List[nn.Parameter]:
        return list(self.A_slow.parameters()) + list(self.B_slow.parameters())

    @torch.no_grad()
    def decay_fast_(self, beta: float) -> None:
        for p in self.fast_params():
            p.mul_(beta)

    @torch.no_grad()
    def decay_slow_(self, beta: float) -> None:
        for p in self.slow_params():
            p.mul_(beta)


def iter_dual_lora_modules(model: nn.Module) -> Iterable[DualLoRALinear]:
    for m in model.modules():
        if isinstance(m, DualLoRALinear):
            yield m


def replace_linears_with_dual_lora(
    model: nn.Module,
    name_patterns: List[str],
    r_fast: int,
    r_slow: int,
    alpha_fast: float,
    alpha_slow: float,
    dropout: float,
) -> int:
    """
    Replace matching nn.Linear modules with DualLoRALinear.
    Matches by full module name using regex OR substring patterns.
    """
    compiled = [re.compile(p) for p in name_patterns]

    def matches(full_name: str) -> bool:
        return any(c.search(full_name) for c in compiled)

    replaced = 0
    # We need parent modules to replace attributes
    for full_name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and matches(full_name):
            # Find parent
            parent_name = full_name.rsplit(".", 1)[0] if "." in full_name else ""
            attr_name = full_name.split(".")[-1]
            parent = model.get_submodule(parent_name) if parent_name else model

            wrapped = DualLoRALinear(
                base=module,
                r_fast=r_fast,
                r_slow=r_slow,
                alpha_fast=alpha_fast,
                alpha_slow=alpha_slow,
                dropout=dropout,
            )
            setattr(parent, attr_name, wrapped)
            replaced += 1
    return replaced


# ----------------------------
# MG-CRI Controller
# ----------------------------

@dataclass
class CRIConfig:
    # Phase dial weights
    aH: float = 0.35
    aD: float = 0.25
    aR: float = 0.20
    aS: float = 0.20

    # Gate thresholds
    tau1: float = 0.35  # fast starts
    tau2: float = 0.70  # slow starts

    # Gate sharpness
    k1: float = 10.0
    k2: float = 12.0

    # Hysteresis bands
    lr_on: float = 0.40
    lr_off: float = 0.30
    ttt_on: float = 0.75
    ttt_off: float = 0.60

    # Surprise gating (slow updates only when sustained)
    surprise_window: int = 16
    surprise_z: float = 1.5  # threshold in "std units" above EMA

    # EMA smoothing for stats
    ema_beta: float = 0.95

    # Decays (half-life behavior)
    beta_fast: float = 0.80
    beta_slow: float = 0.985

    # Learning rates
    lr_fast: float = 5e-4
    lr_slow: float = 1e-4

    # Budgets (gradient norm caps)
    b_fast: float = 0.5
    b_slow: float = 0.2

    # KL trust (approx)
    kl_k: int = 128
    kl_eps0: float = 0.002  # base KL cap
    kl_eps1: float = 0.02   # max KL cap

    # Repetition window
    rep_ngram: int = 3
    rep_window: int = 64


class MGCRIState:
    def __init__(self, cfg: CRIConfig):
        self.cfg = cfg

        # toggles
        self.lr_on_state = 0
        self.ttt_on_state = 0

        # running stats
        self.H_ema = 0.0
        self.D_ema = 0.0
        self.NLL_ema = 0.0
        self.NLL_var_ema = 1.0  # crude variance proxy

        # surprise history
        self.s_hist: List[float] = []

        # last distribution (for KL)
        self.last_logits: Optional[torch.Tensor] = None

        # token history for repetition
        self.token_hist: List[int] = []

    def repetition_proxy(self) -> float:
        """
        Simple repetition proxy:
          - checks if last ngram occurred before in a recent window
        Returns in [0,1].
        """
        n = self.cfg.rep_ngram
        w = self.cfg.rep_window
        if len(self.token_hist) < n + 5:
            return 0.0
        recent = self.token_hist[-w:]
        last_ng = tuple(recent[-n:])
        # search in earlier part of window
        for i in range(0, max(0, len(recent) - n - 1)):
            if tuple(recent[i:i+n]) == last_ng:
                return 1.0
        return 0.0

    def update_phase(
        self,
        logits: torch.Tensor,          # [V]
        target_id: Optional[int],      # teacher-forced next token if available
    ) -> Tuple[float, float, float, Dict[str, float]]:
        cfg = self.cfg
        with torch.no_grad():
            p = F.softmax(logits, dim=-1)
            H = -(p * torch.log(p + 1e-12)).sum().item()

            # KL drift proxy between steps (approx on top-k of current logits)
            if self.last_logits is None:
                D = 0.0
            else:
                D = topk_kl(logits, self.last_logits, k=cfg.kl_k).item()

            R = self.repetition_proxy()

            if target_id is not None:
                nll = F.cross_entropy(logits.unsqueeze(0), torch.tensor([target_id], device=logits.device)).item()
            else:
                nll = self.NLL_ema  # fallback if no target
            # Surprise vs EMA (and crude std)
            self.NLL_ema = ema_update(self.NLL_ema, nll, cfg.ema_beta)
            # Update var proxy
            diff = nll - self.NLL_ema
            self.NLL_var_ema = ema_update(self.NLL_var_ema, diff * diff, cfg.ema_beta)
            nll_std = math.sqrt(max(1e-8, self.NLL_var_ema))
            S = max(0.0, (nll - self.NLL_ema) / (nll_std + 1e-12))

            # Normalize H, D crudely using EMAs
            self.H_ema = ema_update(self.H_ema, H, cfg.ema_beta)
            self.D_ema = ema_update(self.D_ema, D, cfg.ema_beta)

            Hn = 0.0 if self.H_ema == 0 else min(2.0, H / (self.H_ema + 1e-12))
            Dn = 0.0 if self.D_ema == 0 else min(2.0, D / (self.D_ema + 1e-12))
            Rn = R
            Sn = min(2.0, S)

            phi = cfg.aH * (Hn / 2.0) + cfg.aD * (Dn / 2.0) + cfg.aR * Rn + cfg.aS * (Sn / 2.0)
            phi = float(max(0.0, min(1.0, phi)))

            # hysteresis updates
            if phi >= cfg.lr_on:
                self.lr_on_state = 1
            elif phi <= cfg.lr_off:
                self.lr_on_state = 0

            if phi >= cfg.ttt_on:
                self.ttt_on_state = 1
            elif phi <= cfg.ttt_off:
                self.ttt_on_state = 0

            lam = sigmoid_gate(phi - cfg.tau1, cfg.k1) * self.lr_on_state
            mu = sigmoid_gate(phi - cfg.tau2, cfg.k2) * self.ttt_on_state

            # sustained surprise tracking for slow updates
            self.s_hist.append(S)
            if len(self.s_hist) > cfg.surprise_window:
                self.s_hist = self.s_hist[-cfg.surprise_window:]

            self.last_logits = logits.detach()

        debug = {
            "H": H, "D": D, "R": R, "S_z": S,
            "phi": phi, "lambda": lam, "mu": mu,
            "lr_on": float(self.lr_on_state), "ttt_on": float(self.ttt_on_state),
        }
        return phi, lam, mu, debug

    def slow_update_allowed(self) -> bool:
        cfg = self.cfg
        if len(self.s_hist) < cfg.surprise_window:
            return False
        s_bar = sum(self.s_hist) / len(self.s_hist)
        return s_bar >= cfg.surprise_z


# ----------------------------
# Adaptation step
# ----------------------------

def set_all_mixes(model: nn.Module, mix_fast: float, mix_slow: float) -> None:
    for m in iter_dual_lora_modules(model):
        m.set_mix(mix_fast=mix_fast, mix_slow=mix_slow)

def decay_all(model: nn.Module, beta_fast: float, beta_slow: float) -> None:
    for m in iter_dual_lora_modules(model):
        m.decay_fast_(beta_fast)
        m.decay_slow_(beta_slow)

def gather_params(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    fast, slow = [], []
    for m in iter_dual_lora_modules(model):
        fast.extend(m.fast_params())
        slow.extend(m.slow_params())
    return fast, slow

@torch.no_grad()
def approx_kl_cap(phi: float, cfg: CRIConfig) -> float:
    # KL budget grows smoothly with phi
    return cfg.kl_eps0 + (cfg.kl_eps1 - cfg.kl_eps0) * (phi * phi)

def adapt_on_chunk(
    model: nn.Module,
    input_ids: torch.Tensor,     # [1, L]
    attention_mask: torch.Tensor,
    cfg: CRIConfig,
    state: MGCRIState,
    device: torch.device,
) -> Dict[str, float]:
    """
    One adaptation step using teacher forcing on the chunk:
      loss = CE(logits[:, :-1], input_ids[:, 1:])
    Uses current phi/lambda/mu computed from last-position logits and the next token target.

    Applies:
      - fast update scaled by lambda
      - slow update scaled by mu AND sustained surprise gate
      - gradient norm budgets
      - approximate KL cap on last token distribution
      - parameter decay after update
    """
    model.train()
    fast_params, slow_params = gather_params(model)

    # Forward
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits  # [1, L, V]

    # Compute phase at last position (teacher target = last token)
    # Here target_id is next token after last-1 position: input_ids[0, -1]
    last_logits = logits[0, -2, :]  # predict token at -1
    target_id = int(input_ids[0, -1].item())
    phi, lam, mu, debug = state.update_phase(last_logits.detach(), target_id=target_id)

    # Set adapter contribution mixes for forward paths (affects next forwards)
    set_all_mixes(model, mix_fast=lam, mix_slow=mu)

    # Teacher-forced loss over chunk
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss_val = float(loss.detach().item())

    # Backward
    model.zero_grad(set_to_none=True)
    loss.backward()

    # Apply separate budgets and update scales
    # (We implement "scaled SGD" manually for clarity.)
    # Fast: always allowed if lam>0
    # Slow: allowed only if mu>0 and sustained surprise
    slow_ok = (mu > 0.0) and state.slow_update_allowed()

    # Gradient clipping budgets (as function of phi)
    max_g_fast = cfg.b_fast * max(0.05, lam)  # small floor so it can move when lam small
    max_g_slow = cfg.b_slow * (mu * mu if mu > 0 else 0.0)

    pre_fast_norm = clip_norm_(fast_params, max_norm=max_g_fast) if lam > 0 else 0.0
    pre_slow_norm = clip_norm_(slow_params, max_norm=max_g_slow) if slow_ok else 0.0

    # KL trust-region check (approx) on last token distribution
    # Compute logits before update vs after update on same last position.
    with torch.no_grad():
        p_logits = last_logits.detach()

    # Manual SGD update
    with torch.no_grad():
        # Fast step
        if lam > 0.0:
            lr = cfg.lr_fast * lam
            for p in fast_params:
                if p.grad is not None:
                    p.add_(p.grad, alpha=-lr)

        # Slow step (tentative)
        if slow_ok:
            # Save slow params for rollback if KL too big
            slow_backup = [p.detach().clone() for p in slow_params]
            lr = cfg.lr_slow * mu
            for p in slow_params:
                if p.grad is not None:
                    p.add_(p.grad, alpha=-lr)

    # Evaluate KL after tentative slow update
    kl = 0.0
    if slow_ok:
        with torch.no_grad():
            out2 = model(input_ids=input_ids[:, :-1], attention_mask=attention_mask[:, :-1])
            # predict last token from position -2 again
            q_logits = out2.logits[0, -1, :]
            kl = float(topk_kl(p_logits, q_logits, k=cfg.kl_k).item())

            cap = approx_kl_cap(phi, cfg)
            if kl > cap:
                # rollback slow
                for p, b in zip(slow_params, slow_backup):
                    p.copy_(b)
                slow_ok = False  # rejected

    # Decay (mass half-life)
    decay_all(model, beta_fast=cfg.beta_fast, beta_slow=cfg.beta_slow)

    model.eval()
    return {
        **debug,
        "loss": loss_val,
        "slow_ok": float(slow_ok),
        "grad_fast_norm": float(pre_fast_norm),
        "grad_slow_norm": float(pre_slow_norm),
        "kl": float(kl),
    }


# ----------------------------
# Main generation
# ----------------------------

def chunkify(ids: torch.Tensor, chunk_size: int, stride: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Build overlapping chunks for prompt adaptation.
    Returns list of (chunk_ids, chunk_mask), each [1, L]
    """
    assert ids.dim() == 2 and ids.size(0) == 1
    L = ids.size(1)
    chunks = []
    i = 0
    while i < L:
        j = min(L, i + chunk_size)
        chunk = ids[:, i:j]
        mask = torch.ones_like(chunk)
        if chunk.size(1) >= 2:
            chunks.append((chunk, mask))
        if j == L:
            break
        i = max(0, j - stride)
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--max_new_tokens", type=int, default=128)

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])

    # Dual-LoRA target patterns (tune as needed)
    ap.add_argument("--target_patterns", type=str, nargs="+", default=[
        r"\.q_proj$", r"\.k_proj$", r"\.v_proj$", r"\.o_proj$",
        r"\.up_proj$", r"\.down_proj$", r"\.gate_proj$",
    ])
    ap.add_argument("--r_fast", type=int, default=8)
    ap.add_argument("--r_slow", type=int, default=8)
    ap.add_argument("--alpha_fast", type=float, default=16.0)
    ap.add_argument("--alpha_slow", type=float, default=16.0)
    ap.add_argument("--lora_dropout", type=float, default=0.0)

    # Adaptation control
    ap.add_argument("--adapt_prompt", action="store_true")
    ap.add_argument("--adapt_chunk", type=int, default=256)
    ap.add_argument("--adapt_stride", type=int, default=128)
    ap.add_argument("--self_train_generated", action="store_true")  # risky

    args = ap.parse_args()

    device = torch.device(args.device)
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=None,
    ).to(device)
    model.eval()

    # Wrap linears
    replaced = replace_linears_with_dual_lora(
        model,
        name_patterns=args.target_patterns,
        r_fast=args.r_fast,
        r_slow=args.r_slow,
        alpha_fast=args.alpha_fast,
        alpha_slow=args.alpha_slow,
        dropout=args.lora_dropout,
    )
    print(f"[MG-CRI] Replaced {replaced} Linear modules with DualLoRALinear")

    # Controller
    cfg = CRIConfig()
    state = MGCRIState(cfg)

    # Tokenize prompt
    enc = tok(args.prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    # Prompt-time adaptation (teacher forcing)
    if args.adapt_prompt:
        chunks = chunkify(input_ids, chunk_size=args.adapt_chunk, stride=args.adapt_stride)
        print(f"[MG-CRI] Adapting on prompt: {len(chunks)} chunks")
        for idx, (c_ids, c_mask) in enumerate(chunks):
            stats = adapt_on_chunk(model, c_ids, c_mask, cfg, state, device)
            if (idx + 1) % 1 == 0:
                print(
                    f"  chunk {idx+1:>3}/{len(chunks)} "
                    f"phi={stats['phi']:.3f} lam={stats['lambda']:.3f} mu={stats['mu']:.3f} "
                    f"slow_ok={int(stats['slow_ok'])} loss={stats['loss']:.3f} "
                    f"KL={stats['kl']:.5f} R={stats['R']:.0f} S_z={stats['S_z']:.2f}"
                )

    # Generation
    model.eval()
    generated = input_ids.clone()
    gen_attn = attn.clone()

    for t in range(args.max_new_tokens):
        with torch.no_grad():
            out = model(input_ids=generated, attention_mask=gen_attn)
            logits = out.logits[0, -1, :]  # [V]

        # Phase dial for gating contributions during generation
        phi, lam, mu, debug = state.update_phase(logits.detach(), target_id=None)
        set_all_mixes(model, mix_fast=lam, mix_slow=mu)

        # Sample (greedy for simplicity)
        next_id = int(torch.argmax(logits).item())

        # Append
        generated = torch.cat([generated, torch.tensor([[next_id]], device=device)], dim=1)
        gen_attn = torch.cat([gen_attn, torch.ones((1, 1), device=device, dtype=gen_attn.dtype)], dim=1)

        # Token history for repetition proxy
        state.token_hist.append(next_id)

        # Optional (risky) self-training on generated tokens
        if args.self_train_generated and generated.size(1) >= 4:
            # Small local chunk: last adapt_chunk tokens
            start = max(0, generated.size(1) - args.adapt_chunk)
            c_ids = generated[:, start:]
            c_mask = gen_attn[:, start:]
            stats = adapt_on_chunk(model, c_ids, c_mask, cfg, state, device)

        # Stop on EOS
        if next_id == tok.eos_token_id:
            break

    print("\n--- OUTPUT ---")
    print(tok.decode(generated[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
