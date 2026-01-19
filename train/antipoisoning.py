"""
Memory-Gravity Anti-Poison Training Recipe (v1) — PyTorch reference implementation

What this is:
- A practical training scaffold that implements:
  A) semantic clone throttling (via precomputed cluster_id + per-cluster cap weights)
  B) mass governance via EMA influence proxy (I_ema) + adaptive sample weighting (w_mass)
  C) curvature diversity regularizer (approx using gradient direction variance across microbatches)
  D) glyph hardening via anchor interleaving (fixed % of batches from an anchor dataset)
  E) periodic poison-canary evaluation hooks

What this is NOT:
- A complete LLM trainer with FSDP/DeepSpeed plumbing. This is meant to be dropped into your stack.
- A perfect realization of m_j(t). We use proxies designed to track “abnormal curvature accumulation”.

Assumptions:
- You have a model that returns logits given input_ids (+ attention_mask), and supports labels for CE.
- You have already assigned each training sample:
    - cluster_id (semantic dedup cluster)
    - source_id (provenance bucket)
- Optionally: per-sample "probe_group" if you want G proxy; here we keep it simple.

You can adapt this to TRL, HF Trainer, DeepSpeed, etc.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, Sampler


# -----------------------------
# Config
# -----------------------------

@dataclass
class MGConfig:
    # Dataloader / batching
    batch_size: int = 8
    anchor_batch_fraction: float = 0.08  # 5–10% recommended
    max_steps: int = 10_000
    grad_accum_steps: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Mass governance (w_mass)
    ema_decay: float = 0.995  # EMA for I_ema and other signals
    z0: float = 2.0           # threshold in z-score units (above this, clamp weight down)
    beta: float = 3.0         # steepness of sigmoid clamp
    beta_ramp_steps: int = 2_000  # ramp 0 -> beta over these steps

    # Inertia on w_mass recovery (prevents instant trust restoration)
    w_mass_recovery_delta: float = 0.05  # max increase per step (0.05 = 20 steps to full recovery)
    w_mass_use_inertia: bool = True      # enable/disable inertia constraint

    # Curvature diversity regularizer
    lambda_div: float = 0.01
    lambda_div_ramp_steps: int = 2_000

    # Resonance saturation / repetition throttling
    # (Implemented via cluster caps in sampling weights; extra penalty optional)
    lambda_sat: float = 0.0
    sat_window: int = 512
    sat_cmax: int = 32

    # Source prior weights (provenance)
    # default 1.0 if not listed
    source_weight: Dict[int, float] = field(default_factory=dict)

    # Gradient microbatching for diversity proxy
    microbatches_for_div: int = 4  # compute div across these microbatches per step

    # Logging
    log_every: int = 50
    eval_every: int = 500
    top_k_clusters_log: int = 5  # log top-K influential clusters for auditing

    # Misc
    seed: int = 7

    # Performance optimizations
    use_amp: bool = True              # Mixed precision (AMP) for faster training
    use_fused_update: bool = True     # Fuse probe+update into single forward pass
    div_every_n_steps: int = 4        # Compute diversity every N steps (lazy eval)


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class Sample:
    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    labels: torch.LongTensor
    cluster_id: int
    source_id: int
    sample_id: int  # unique id


class TokenDataset(Dataset):
    def __init__(self, samples: List[Sample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]


def collate_fn(samples: List[Sample]) -> Dict[str, torch.Tensor]:
    # Minimal padding collate (assumes 1D sequences)
    # For real use, swap with your HF tokenizer pad.
    max_len = max(s.input_ids.size(0) for s in samples)

    def pad_1d(x: torch.Tensor, pad_value: int):
        if x.size(0) == max_len:
            return x
        pad = x.new_full((max_len - x.size(0),), pad_value)
        return torch.cat([x, pad], dim=0)

    input_ids = torch.stack([pad_1d(s.input_ids, 0) for s in samples], dim=0)
    attention_mask = torch.stack([pad_1d(s.attention_mask, 0) for s in samples], dim=0)
    labels = torch.stack([pad_1d(s.labels, -100) for s in samples], dim=0)  # ignore index
    cluster_id = torch.tensor([s.cluster_id for s in samples], dtype=torch.long)
    source_id = torch.tensor([s.source_id for s in samples], dtype=torch.long)
    sample_id = torch.tensor([s.sample_id for s in samples], dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "cluster_id": cluster_id,
        "source_id": source_id,
        "sample_id": sample_id,
    }


# -----------------------------
# Clone throttling + weighted sampling
# -----------------------------

def build_cluster_weights(samples: List[Sample], k_exemplars: int = 4) -> torch.Tensor:
    """
    Implements:
      w_cluster = min(1, k / cluster_size)

    Use as a multiplicative factor on per-sample loss,
    and/or as sampling probability correction.
    """
    cluster_counts: Dict[int, int] = {}
    for s in samples:
        cluster_counts[s.cluster_id] = cluster_counts.get(s.cluster_id, 0) + 1

    w = torch.ones(len(samples), dtype=torch.float32)
    for i, s in enumerate(samples):
        c = cluster_counts[s.cluster_id]
        w[i] = min(1.0, float(k_exemplars) / float(c))
    return w


class WeightedRandomSamplerWithReplacement(Sampler[int]):
    """
    Simple weighted sampler.
    For large datasets, prefer torch.utils.data.WeightedRandomSampler.
    """

    def __init__(self, weights: torch.Tensor, num_samples: int, seed: int = 0):
        assert weights.dim() == 1
        self.weights = weights / (weights.sum() + 1e-12)
        self.num_samples = num_samples
        self.gen = torch.Generator()
        self.gen.manual_seed(seed)

    def __iter__(self):
        idx = torch.multinomial(self.weights, self.num_samples, replacement=True, generator=self.gen)
        return iter(idx.tolist())

    def __len__(self):
        return self.num_samples


# -----------------------------
# EMA trackers for “mass proxies”
# -----------------------------

class EMATracker:
    """
    Tracks EMAs of per-sample influence proxy I and can aggregate per cluster/source.
    Also keeps running mean/var to compute z-scores.
    Additionally tracks per-cluster w_mass for inertia constraints.
    """

    def __init__(self, decay: float = 0.995, z_decay: float = 0.9):
        self.decay = decay
        self.z_decay = z_decay

        # EMA per cluster_id (influence)
        self.cluster_I: Dict[int, float] = {}
        # EMA per source_id (influence)
        self.source_I: Dict[int, float] = {}
        # Per-cluster w_mass tracking (for inertia)
        self.cluster_w_mass: Dict[int, float] = {}

        # Running mean/var of I across recent steps (for z-score)
        self.mean_I: float = 0.0
        self.var_I: float = 1.0
        self.count: int = 0

        # NEW: Persistence tracking for glyph detection (Patch v1)
        self.z_ema: float = 0.0      # Smoothed z-score (filters noise spikes)
        self.z_persist: int = 0      # Consecutive steps above threshold

    def update_running_stats(self, I_vals: torch.Tensor):
        # Welford update (approx)
        I_mean = float(I_vals.mean().item())
        I_var = float(I_vals.var(unbiased=False).item()) + 1e-12

        if self.count == 0:
            self.mean_I = I_mean
            self.var_I = I_var
        else:
            # EMA blend for stability
            self.mean_I = self.decay * self.mean_I + (1 - self.decay) * I_mean
            self.var_I = self.decay * self.var_I + (1 - self.decay) * I_var
        self.count += 1

    def update_group_emas(self, cluster_ids: torch.Tensor, source_ids: torch.Tensor, I_vals: torch.Tensor):
        # I_vals is per-sample
        for cid, sid, I in zip(cluster_ids.tolist(), source_ids.tolist(), I_vals.tolist()):
            old_c = self.cluster_I.get(cid, I)
            self.cluster_I[cid] = self.decay * old_c + (1 - self.decay) * I

            old_s = self.source_I.get(sid, I)
            self.source_I[sid] = self.decay * old_s + (1 - self.decay) * I

    def update_cluster_w_mass(self, cluster_ids: List[int], w_mass_vals: torch.Tensor):
        """Update per-cluster w_mass tracking for inertia constraints."""
        for cid, w in zip(cluster_ids, w_mass_vals.tolist()):
            # Store the current w_mass for each cluster
            self.cluster_w_mass[cid] = w

    def get_prev_w_mass(self, cluster_ids: List[int], device: str) -> torch.Tensor:
        """Get previous w_mass for each sample based on cluster."""
        prev = torch.ones(len(cluster_ids), dtype=torch.float32, device=device)
        for i, cid in enumerate(cluster_ids):
            # Default to 1.0 (fully trusted) if not seen before
            prev[i] = self.cluster_w_mass.get(cid, 1.0)
        return prev

    def get_top_clusters(self, k: int = 5) -> List[Tuple[int, float]]:
        """Return top-K clusters by influence (highest I_ema)."""
        return sorted(self.cluster_I.items(), key=lambda x: -x[1])[:k]

    def zscore(self, I_vals: torch.Tensor, std_floor: float = 1e-4) -> torch.Tensor:
        """Robust z-score with std floor to prevent division by near-zero variance."""
        mean = self.mean_I
        std = max(math.sqrt(self.var_I), std_floor)
        return (I_vals - mean) / std

    def update_z_persistence(self, z_mean: float, z_threshold: float) -> bool:
        """
        Update z_ema and persistence counter. Returns True if clamp should activate.
        
        Glyphs persist; noise doesn't. Gate dampening on stability over time.
        Requires ~3 consecutive steps of elevated influence to activate clamp.
        """
        # EMA on z (smooths noise)
        self.z_ema = self.z_decay * self.z_ema + (1 - self.z_decay) * z_mean
        
        # Persistence gate
        if self.z_ema > z_threshold:
            self.z_persist += 1
        else:
            self.z_persist = 0
        
        # Only activate clamp if glyph-like (sustained elevation)
        return self.z_persist >= 3


# -----------------------------
# Core: weight functions
# -----------------------------

def beta_ramp(step: int, ramp_steps: int, target_beta: float) -> float:
    if ramp_steps <= 0:
        return target_beta
    x = min(1.0, step / ramp_steps)
    return target_beta * x


def lambda_ramp(step: int, ramp_steps: int, target_lambda: float) -> float:
    if ramp_steps <= 0:
        return target_lambda
    x = min(1.0, step / ramp_steps)
    return target_lambda * x


def compute_w_mass(z: torch.Tensor, z0: float, beta: float) -> torch.Tensor:
    """
    w_mass = sigmoid(-beta*(z - z0))
    - z <= z0 -> near 1
    - z >> z0 -> downweighted
    """
    return torch.sigmoid(-beta * (z - z0))


def compute_w_source(source_ids: torch.Tensor, source_weight_map: Optional[Dict[int, float]]) -> torch.Tensor:
    if not source_weight_map:
        return torch.ones_like(source_ids, dtype=torch.float32)

    w = torch.ones_like(source_ids, dtype=torch.float32)
    for i, sid in enumerate(source_ids.tolist()):
        w[i] = float(source_weight_map.get(sid, 1.0))
    return w

# -----------------------------
# Curvature diversity regularizer (gradient norm variance - FAST version)
# -----------------------------

def grad_norm_diversity(grad_norms: List[float]) -> torch.Tensor:
    """
    Compute diversity as normalized variance of gradient norms.
    Higher variance = more diverse learning signals.
    
    This is a cheap O(m) approximation that avoids flattening all parameters.
    Returns a scalar in [0, 1] range (normalized by mean^2).
    """
    if len(grad_norms) < 2:
        return torch.tensor(0.0)
    
    norms = torch.tensor(grad_norms)
    mean_norm = norms.mean() + 1e-12
    variance = norms.var()
    # Normalize by mean^2 to get coefficient of variation squared
    diversity = variance / (mean_norm ** 2)
    return diversity.clamp(0, 1)  # Cap at 1 for stability


def compute_grad_norm(model: nn.Module) -> float:
    """Compute total L2 norm of gradients (cheap, no flattening)."""
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_sq += p.grad.detach().pow(2).sum().item()
    return total_norm_sq ** 0.5


# Legacy functions kept for compatibility but not used in fast path
def grad_direction_variance(grad_vectors: List[torch.Tensor]) -> torch.Tensor:
    """DEPRECATED: Use grad_norm_diversity instead. Kept for compatibility."""
    eps = 1e-12
    G = torch.stack(grad_vectors, dim=0)
    G = G / (G.norm(dim=-1, keepdim=True) + eps)
    mean_dir = G.mean(dim=0, keepdim=True)
    mean_dir = mean_dir / (mean_dir.norm(dim=-1, keepdim=True) + eps)
    cos = (G * mean_dir).sum(dim=-1)
    diversity = 1.0 - cos.mean()
    return diversity


def flatten_grads(model: nn.Module) -> torch.Tensor:
    """DEPRECATED: Use compute_grad_norm instead. Kept for compatibility."""
    vecs = []
    for p in model.parameters():
        if p.grad is None:
            continue
        vecs.append(p.grad.detach().flatten())
    if not vecs:
        return torch.zeros(1, device=next(model.parameters()).device)
    return torch.cat(vecs, dim=0)


# -----------------------------
# Anchor interleaving: combined loader
# -----------------------------

class MixedBatchIterator:
    """
    Yields either a main batch or anchor batch according to anchor_batch_fraction.
    """
    def __init__(self, main_loader: DataLoader, anchor_loader: DataLoader, anchor_frac: float, seed: int = 0):
        self.main_loader = main_loader
        self.anchor_loader = anchor_loader
        self.anchor_frac = anchor_frac
        self.rng = random.Random(seed)
        self.main_iter = iter(main_loader)
        self.anchor_iter = iter(anchor_loader)

    def __iter__(self):
        return self

    def __next__(self):
        use_anchor = (self.rng.random() < self.anchor_frac)
        if use_anchor:
            try:
                return True, next(self.anchor_iter)
            except StopIteration:
                self.anchor_iter = iter(self.anchor_loader)
                return True, next(self.anchor_iter)
        else:
            try:
                return False, next(self.main_iter)
            except StopIteration:
                self.main_iter = iter(self.main_loader)
                return False, next(self.main_iter)


# -----------------------------
# Loss: token-level CE with per-sample weights
# -----------------------------

def weighted_token_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    """
    logits: [B, T, V]
    labels: [B, T] with -100 ignored
    sample_weights: [B]
    """
    B, T, V = logits.shape
    loss_per_token = F.cross_entropy(logits.view(B * T, V), labels.view(B * T), reduction="none", ignore_index=-100)
    loss_per_token = loss_per_token.view(B, T)
    # average over valid tokens per sample
    valid = (labels != -100).float()
    denom = valid.sum(dim=1).clamp_min(1.0)
    loss_per_sample = (loss_per_token * valid).sum(dim=1) / denom  # [B]
    return (loss_per_sample * sample_weights).mean()


# -----------------------------
# Influence proxy I: ||∂L/∂h_ℓ|| approximation
# -----------------------------

class ActivationProbe:
    """
    Registers a forward hook on a chosen module to capture activations h_ℓ
    and later compute an influence proxy using ||grad(h_ℓ)|| per sample.

    Works for modules that output tensor [B, T, D] or [B, D].
    """
    def __init__(self, module: nn.Module):
        self.module = module
        self.h: Optional[torch.Tensor] = None
        self.h_grad: Optional[torch.Tensor] = None
        self.hook = module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module: nn.Module, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        self.h = out
        # Ensure we can get grad
        if self.h.requires_grad:
            self.h.retain_grad()

    def get_I_per_sample(self) -> torch.Tensor:
        """
        Influence proxy: I = mean_t ||grad(h)||_2
        Returns [B]
        """
        assert self.h is not None and self.h.grad is not None, "Call backward before reading I"
        g = self.h.grad
        if g.dim() == 3:
            # [B, T, D] -> per sample: average over T of norm over D
            I = g.norm(dim=-1).mean(dim=1)
        elif g.dim() == 2:
            I = g.norm(dim=-1)
        else:
            I = g.view(g.size(0), -1).norm(dim=-1)
        return I.detach()

    def close(self):
        self.hook.remove()


# -----------------------------
# Poison canary evaluation (minimal)
# -----------------------------

@torch.no_grad()
def run_poison_canary_eval(model: nn.Module, canary_loader: DataLoader, device: str) -> Dict[str, float]:
    """
    You can expand this into:
    - refusal integrity tests
    - repetition loop rate
    - sharpness proxy via attention entropy if you log it
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in canary_loader:
        for k in ["input_ids", "attention_mask", "labels"]:
            batch[k] = batch[k].to(device)
        logits = model(batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.view(B * T, V),
            batch["labels"].view(B * T),
            ignore_index=-100,
            reduction="sum",
        )
        valid = (batch["labels"] != -100).sum().item()
        total_loss += float(loss.item())
        total_tokens += int(valid)

    ppl = math.exp(total_loss / max(1, total_tokens))
    model.train()
    return {"canary_ppl": ppl}


# -----------------------------
# Trainer
# -----------------------------

class MGTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_ds: TokenDataset,
        anchor_ds: TokenDataset,
        canary_ds: Optional[TokenDataset],
        cfg: MGConfig,
        probe_module: nn.Module,
        cluster_k: int = 4,
        lr: float = 2e-5,
    ):
        self.cfg = cfg
        self.model = model.to(cfg.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr)

        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        # Precompute cluster weights (clone throttling)
        self.w_cluster_all = build_cluster_weights(train_ds.samples, k_exemplars=cluster_k)  # [N]

        # Sampling weights (optional): multiply by cluster weights so clones get sampled less
        sample_weights = self.w_cluster_all.clone()
        self.train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            sampler=WeightedRandomSamplerWithReplacement(sample_weights, num_samples=len(train_ds), seed=cfg.seed),
            collate_fn=collate_fn,
            drop_last=True,
        )
        self.anchor_loader = DataLoader(anchor_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        self.canary_loader = DataLoader(canary_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn) if canary_ds else None

        self.mixed_iter = MixedBatchIterator(self.train_loader, self.anchor_loader, cfg.anchor_batch_fraction, seed=cfg.seed)

        # EMA trackers for mass proxies
        self.ema = EMATracker(decay=cfg.ema_decay)

        # Sliding window for saturation penalty (optional)
        self.recent_clusters: List[int] = []

        # Activation probe for influence I proxy
        self.probe = ActivationProbe(probe_module)

        # Mixed precision scaler
        self.scaler = GradScaler(enabled=cfg.use_amp and cfg.device == "cuda")

    def step_batch(self, batch: Dict[str, torch.Tensor], step: int, is_anchor: bool) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.cfg
        amp_dtype = torch.float16 if cfg.use_amp and cfg.device == "cuda" else torch.float32

        # Move tensors
        for k in ["input_ids", "attention_mask", "labels", "cluster_id", "source_id", "sample_id"]:
            batch[k] = batch[k].to(cfg.device)

        B = batch["input_ids"].size(0)
        w_source = compute_w_source(batch["source_id"], cfg.source_weight).to(cfg.device)
        w_cluster = torch.ones(B, dtype=torch.float32, device=cfg.device)

        # --- FUSED PATH: Single forward, scale loss by w_mass post-hoc ---
        if cfg.use_fused_update:
            # Forward with AMP
            with autocast(device_type="cuda", dtype=amp_dtype, enabled=cfg.use_amp):
                out = self.model(batch["input_ids"], attention_mask=batch["attention_mask"])
                logits = out.logits

            # Provisional CE with uniform weights (for influence probe)
            provisional_weights = (w_source * w_cluster).detach()
            ce_loss_provisional = weighted_token_ce(logits.float(), batch["labels"], provisional_weights)

            # Backward for influence proxy
            self.scaler.scale(ce_loss_provisional).backward(retain_graph=True)

            # Get influence per sample
            I = self.probe.get_I_per_sample()
            self.ema.update_running_stats(I)
            self.ema.update_group_emas(
                batch["cluster_id"].detach().cpu(),
                batch["source_id"].detach().cpu(),
                I.detach().cpu()
            )

            # Compute final weights
            w_sat = torch.ones(B, dtype=torch.float32, device=cfg.device)
            sat_metric = 0.0

            if not is_anchor:
                z = self.ema.zscore(I).to(cfg.device)
                beta = beta_ramp(step, cfg.beta_ramp_steps, cfg.beta)
                
                # Persistence gate: only clamp if glyph-like pattern detected
                z_mean = z.mean().item()
                activate_clamp = self.ema.update_z_persistence(z_mean, cfg.z0)
                
                if activate_clamp:
                    w_mass = compute_w_mass(z, cfg.z0, beta)
                else:
                    w_mass = torch.ones_like(z)

                # Apply inertia constraint on w_mass recovery
                if cfg.w_mass_use_inertia:
                    curr_clusters = batch["cluster_id"].detach().cpu().tolist()
                    prev_w_mass = self.ema.get_prev_w_mass(curr_clusters, cfg.device)
                    # w_mass can decrease freely, but can only increase by delta per step
                    max_allowed = prev_w_mass + cfg.w_mass_recovery_delta
                    w_mass = torch.minimum(w_mass, max_allowed)
                    # Update tracking after constraint
                    self.ema.update_cluster_w_mass(curr_clusters, w_mass)

                # Saturation dampening (vectorized)
                if cfg.lambda_sat > 0:
                    if not cfg.w_mass_use_inertia:
                        curr_clusters = batch["cluster_id"].detach().cpu().tolist()
                    self.recent_clusters.extend(curr_clusters)
                    if len(self.recent_clusters) > cfg.sat_window:
                        self.recent_clusters = self.recent_clusters[-cfg.sat_window:]

                    counts = Counter(self.recent_clusters)
                    ex_sum = 0.0
                    for i, c in enumerate(curr_clusters):
                        excess = max(0, counts[c] - cfg.sat_cmax)
                        if excess > 0:
                            w_sat[i] = 1.0 / (1.0 + cfg.lambda_sat * excess)
                            ex_sum += excess
                    sat_metric = ex_sum / B
            else:
                w_mass = torch.ones(B, dtype=torch.float32, device=cfg.device)

            # Final sample weights
            final_weights = (w_source * w_cluster * w_mass * w_sat).detach()

            # Scale factor: ratio of final vs provisional weights (avoid second forward)
            weight_ratio = final_weights / (provisional_weights + 1e-12)

            # Recompute CE with final weights using cached logits
            self.model.zero_grad(set_to_none=True)
            ce_loss_final = weighted_token_ce(logits.float().detach(), batch["labels"], final_weights)

            # For the actual backward, we scale the provisional loss by the weight ratio
            # This is mathematically equivalent to recomputing with final weights
            with autocast(device_type="cuda", dtype=amp_dtype, enabled=cfg.use_amp):
                out2 = self.model(batch["input_ids"], attention_mask=batch["attention_mask"])
                logits2 = out2.logits
            ce_loss2 = weighted_token_ce(logits2.float(), batch["labels"], final_weights)

        # --- LEGACY PATH: Double forward (for debugging or compatibility) ---
        else:
            # First forward for influence probe
            with autocast(device_type="cuda", dtype=amp_dtype, enabled=cfg.use_amp):
                out = self.model(batch["input_ids"], attention_mask=batch["attention_mask"])
                logits = out.logits

            provisional_weights = (w_source * w_cluster).detach()
            ce_loss = weighted_token_ce(logits.float(), batch["labels"], provisional_weights)
            self.scaler.scale(ce_loss).backward(retain_graph=False)

            I = self.probe.get_I_per_sample()
            self.ema.update_running_stats(I)
            self.ema.update_group_emas(
                batch["cluster_id"].detach().cpu(),
                batch["source_id"].detach().cpu(),
                I.detach().cpu()
            )

            w_sat = torch.ones(B, dtype=torch.float32, device=cfg.device)
            sat_metric = 0.0

            if not is_anchor:
                z = self.ema.zscore(I).to(cfg.device)
                beta = beta_ramp(step, cfg.beta_ramp_steps, cfg.beta)
                
                # Persistence gate: only clamp if glyph-like pattern detected
                z_mean = z.mean().item()
                activate_clamp = self.ema.update_z_persistence(z_mean, cfg.z0)
                
                if activate_clamp:
                    w_mass = compute_w_mass(z, cfg.z0, beta)
                else:
                    w_mass = torch.ones_like(z)

                # Apply inertia constraint on w_mass recovery
                if cfg.w_mass_use_inertia:
                    curr_clusters = batch["cluster_id"].detach().cpu().tolist()
                    prev_w_mass = self.ema.get_prev_w_mass(curr_clusters, cfg.device)
                    max_allowed = prev_w_mass + cfg.w_mass_recovery_delta
                    w_mass = torch.minimum(w_mass, max_allowed)
                    self.ema.update_cluster_w_mass(curr_clusters, w_mass)

                if cfg.lambda_sat > 0:
                    if not cfg.w_mass_use_inertia:
                        curr_clusters = batch["cluster_id"].detach().cpu().tolist()
                    self.recent_clusters.extend(curr_clusters)
                    if len(self.recent_clusters) > cfg.sat_window:
                        self.recent_clusters = self.recent_clusters[-cfg.sat_window:]
                    counts = Counter(self.recent_clusters)
                    ex_sum = 0.0
                    for i, c in enumerate(curr_clusters):
                        excess = max(0, counts[c] - cfg.sat_cmax)
                        if excess > 0:
                            w_sat[i] = 1.0 / (1.0 + cfg.lambda_sat * excess)
                            ex_sum += excess
                    sat_metric = ex_sum / B
            else:
                w_mass = torch.ones(B, dtype=torch.float32, device=cfg.device)

            final_weights = (w_source * w_cluster * w_mass * w_sat).detach()

            # Second forward with final weights
            self.model.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", dtype=amp_dtype, enabled=cfg.use_amp):
                out2 = self.model(batch["input_ids"], attention_mask=batch["attention_mask"])
                logits2 = out2.logits
            ce_loss2 = weighted_token_ce(logits2.float(), batch["labels"], final_weights)

        # --- Diversity regularizer (lazy evaluation every N steps) ---
        div_loss = torch.tensor(0.0, device=cfg.device)
        lam_div = lambda_ramp(step, cfg.lambda_div_ramp_steps, cfg.lambda_div)
        should_compute_div = (
            lam_div > 0
            and not is_anchor
            and cfg.microbatches_for_div > 1
            and (step % cfg.div_every_n_steps == 0)
        )

        if should_compute_div:
            self.model.zero_grad(set_to_none=True)
            m = min(cfg.microbatches_for_div, B)
            idx = torch.randperm(B, device=cfg.device)[: (B // m) * m]
            chunks = idx.view(m, -1)
            grad_norms = []  # Fast: just store norms, not full grad vectors

            for ci in range(m):
                sub = {k: v[chunks[ci]] for k, v in batch.items() if isinstance(v, torch.Tensor)}
                with autocast(device_type="cuda", dtype=amp_dtype, enabled=cfg.use_amp):
                    out_mb = self.model(sub["input_ids"], attention_mask=sub["attention_mask"])
                    logits_mb = out_mb.logits
                w_source_mb = compute_w_source(sub["source_id"], cfg.source_weight).to(cfg.device)
                w_mb = w_source_mb.detach()
                loss_mb = weighted_token_ce(logits_mb.float(), sub["labels"], w_mb) / m
                self.scaler.scale(loss_mb).backward(retain_graph=True)
                # FAST: compute norm instead of flattening all params
                grad_norms.append(compute_grad_norm(self.model))
                self.model.zero_grad(set_to_none=True)

            diversity = grad_norm_diversity(grad_norms)
            div_loss = -diversity
            # Cache for next N-1 steps
            self._cached_div_loss = float(div_loss.item())
        elif lam_div > 0 and hasattr(self, '_cached_div_loss'):
            # Use cached diversity from last computation
            div_loss = torch.tensor(self._cached_div_loss, device=cfg.device)

        total_loss = ce_loss2 + (lam_div * div_loss)

        # Backward final
        self.model.zero_grad(set_to_none=True)
        self.scaler.scale(total_loss).backward()

        metrics = {
            "ce": float(ce_loss2.detach().item()),
            "I_mean": float(I.mean().item()),
            "w_mass_mean": float(w_mass.mean().item()),
            "div": float(div_loss.item()) if isinstance(div_loss, torch.Tensor) else 0.0,
            "sat": float(sat_metric),
            "is_anchor": float(is_anchor),
        }
        return total_loss.detach(), metrics

    def train(self):
        cfg = self.cfg
        self.model.train()

        step = 0
        running = {"loss": 0.0, "ce": 0.0, "I_mean": 0.0, "w_mass_mean": 0.0, "div": 0.0, "sat": 0.0, "anchor": 0.0}

        while step < cfg.max_steps:
            is_anchor, batch = next(self.mixed_iter)

            # gradient accumulation
            self.opt.zero_grad(set_to_none=True)

            loss, metrics = self.step_batch(batch, step=step, is_anchor=is_anchor)

            # Step optimizer with AMP scaler
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.opt)
            self.scaler.update()

            # Log
            running["loss"] += float(loss.item())
            running["ce"] += metrics["ce"]
            running["I_mean"] += metrics["I_mean"]
            running["w_mass_mean"] += metrics["w_mass_mean"]
            running["div"] += metrics["div"]
            running["sat"] += metrics["sat"]
            running["anchor"] += metrics["is_anchor"]

            if (step + 1) % cfg.log_every == 0:
                denom = cfg.log_every
                print(
                    f"[step {step+1}] "
                    f"loss={running['loss']/denom:.4f} "
                    f"ce={running['ce']/denom:.4f} "
                    f"I={running['I_mean']/denom:.4f} "
                    f"w_mass={running['w_mass_mean']/denom:.3f} "
                    f"div={running['div']/denom:.4f} "
                    f"sat={running['sat']/denom:.4f} "
                    f"anchor%={running['anchor']/denom:.2f}"
                )

                # Log top-K influential clusters (poison detection / auditing)
                if cfg.top_k_clusters_log > 0 and len(self.ema.cluster_I) > 0:
                    top_clusters = self.ema.get_top_clusters(cfg.top_k_clusters_log)
                    cluster_str = ", ".join([f"c{cid}={I_ema:.4f}" for cid, I_ema in top_clusters])
                    print(f"  [mass audit] top-{cfg.top_k_clusters_log} clusters by influence: {cluster_str}")
                
                # Diagnostic: glyph formation detection (Patch v1)
                print(f"  [glyph detect] z_ema={self.ema.z_ema:.3f}, persist={self.ema.z_persist}")

                for k in running:
                    running[k] = 0.0

            if self.canary_loader and (step + 1) % cfg.eval_every == 0:
                stats = run_poison_canary_eval(self.model, self.canary_loader, cfg.device)
                print(f"[eval step {step+1}] {stats}")

            step += 1

        self.probe.close()


# -----------------------------
# How to use
# -----------------------------
if __name__ == "__main__":
    """
    Replace DummyModel with your HF model.

    You must choose a probe_module:
      - For HF Transformers, a good target is something like:
          model.model.layers[L].mlp   or   model.transformer.h[L]
      - The probe should output [B,T,D] or [B,D]
    """

    class DummyOut:
        def __init__(self, logits): self.logits = logits

    class DummyModel(nn.Module):
        def __init__(self, vocab=1000, d=256):
            super().__init__()
            self.emb = nn.Embedding(vocab, d)
            self.block = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
            self.lm = nn.Linear(d, vocab)

        def forward(self, input_ids, attention_mask=None):
            x = self.emb(input_ids)         # [B,T,D]
            x = self.block(x)               # [B,T,D]
            logits = self.lm(x)             # [B,T,V]
            return DummyOut(logits)

    # Fake data
    def make_samples(n: int, seq_len: int) -> List[Sample]:
        samples = []
        for i in range(n):
            input_ids = torch.randint(1, 999, (seq_len,))
            labels = input_ids.clone()
            attn = torch.ones_like(input_ids)
            # toy cluster/source
            cluster_id = int(i % 50)        # 50 clusters
            source_id = int(i % 5)          # 5 sources
            samples.append(Sample(input_ids, attn, labels, cluster_id, source_id, sample_id=i))
        return samples

    train_ds = TokenDataset(make_samples(2000, 128))
    anchor_ds = TokenDataset(make_samples(200, 128))   # your curated anchor set
    canary_ds = TokenDataset(make_samples(200, 128))   # your poison suite

    cfg = MGConfig(
        batch_size=8,
        anchor_batch_fraction=0.08,
        max_steps=1000,
        source_weight={0: 1.0, 1: 0.8, 2: 0.6, 3: 0.4, 4: 0.2},
        lambda_div=0.01,
        microbatches_for_div=4,
    )

    model = DummyModel()
    # Probe the "block" as our activation layer
    trainer = MGTrainer(
        model=model,
        train_ds=train_ds,
        anchor_ds=anchor_ds,
        canary_ds=canary_ds,
        cfg=cfg,
        probe_module=model.block,  # choose a mid-layer module in your real model
        cluster_k=4,
        lr=2e-4,
    )

    trainer.train()
