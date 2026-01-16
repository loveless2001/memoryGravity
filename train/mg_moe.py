"""
MG-MoE: Memory-Gravity Native Mixture-of-Experts Growth Protocol
Reference: @mg_plan.md

Implements:
- Resonance-based Routing (Non-competitive)
- Mass-Governed Expert Lifecycle (Birth, Charging, Learning, Death)
- Dynamic Capacity Expansion
- Residual HF-Compatible Integration
"""

import math
import random
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Iterable

# -----------------------------------------------------------------------------
# 1. Core Data Structures
# -----------------------------------------------------------------------------

@dataclass
class GovernorState:
    """Global state for the Mass Governor."""
    M_total_max: float = 1000.0
    m_min_learn: float = 1.0  # Threshold mass to enable learning
    dominance_kappa: float = 0.5  # Max expert mass must be >= kappa * total mass
    birth_cooldown: int = 100  # Steps between births
    last_birth_step: int = -100
    
    # Pruning/Merging
    epsilon_death: float = 0.01
    death_steps: int = 50 # How many steps of low mass before death
    merge_similarity: float = 0.9
    
    # Growth
    newborn_charge_rate: float = 0.1 # Rate at which threshold lowers
    saturation_eps: float = 0.2      # Max mass growth below this triggers spawn if phi high

class Expert(nn.Module):
    """
    A curvature reservoir.
    Contains:
    - theta: The trainable parameters (e.g., an MLP)
    - anchor: A vector signature defining the expert's curvature domain
    - mass: Accumulated importance
    - decay: Persistence rate
    - threshold: Activation gate threshold
    - alive: Boolean flag
    """
    def __init__(self, d_model: int, hidden_dim: int = None, initial_decay: float = 0.99):
        super().__init__()
        self.d_model = d_model
        if hidden_dim is None:
            hidden_dim = d_model * 4
            
        # The "theta" - a simple residual adapter or MLP
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )
        # Initialize output to zero for zero-impact start
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        # Anchor: Curvature signature (small random vector)
        self.register_buffer('anchor', torch.randn(d_model) * 0.01)
        
        # Mass Lifecycle buffers
        self.register_buffer('mass', torch.tensor(0.0))
        self.register_buffer('decay', torch.tensor(initial_decay))
        
        # Activation Threshold (starts infinite for newborns)
        self.register_buffer('threshold', torch.tensor(float('inf')))
        
        # Status
        self.alive = True
        self.steps_low_mass = 0 # Counter for death

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Router(nn.Module):
    """
    Resonance sensor. 
    Determines where the trajectory 'wants' to fall.
    """
    def __init__(self, d_model: int, beta: float = 1.0):
        super().__init__()
        self.W_r = nn.Linear(d_model, d_model, bias=False)
        self.beta = beta
        
    def forward(self, x_t: torch.Tensor, experts: List[Expert]) -> Dict[int, torch.Tensor]:
        """
        x_t: [Batch, D] or [Batch, Seq, D]
        experts: list of Expert modules
        
        Returns: {expert_idx: gating_value [Batch*Seq, 1] or [Batch, 1]}
        """
        activations = {}
        
        # Flatten if 3D: [B, T, D] -> [B*T, D]
        if x_t.dim() == 3:
            x_flat = x_t.reshape(-1, x_t.size(-1))
        else:
            x_flat = x_t
        
        # Project state once
        proj_x = self.W_r(x_flat) # [B*T, D]
        
        for i, expert in enumerate(experts):
            if not expert.alive:
                continue
                
            # Resonance: <W_r x, anchor>
            rho = (proj_x * expert.anchor).sum(dim=-1, keepdim=True)
            
            # Gating: sigmoid(beta * (rho - tau))
            if torch.isinf(expert.threshold):
                g = torch.zeros_like(rho)
            else:
                g = torch.sigmoid(self.beta * (rho - expert.threshold))
            
            # Sparsity optimization
            if g.max() > 1e-4:
                activations[i] = g
                
        return activations

# -----------------------------------------------------------------------------
# 2. Residual MG-MoE Module (Integration)
# -----------------------------------------------------------------------------

class MGMoEResidual(nn.Module):
    """
    Wrap an existing MLP/FFN module so it returns:
        mlp(x) + sum_i g_i(x) * expert_i(x)
    i.e., residual contribution only.
    """
    def __init__(
        self,
        d_model: int,
        core_mlp: nn.Module,
        governor: Optional[GovernorState] = None,
        init_experts: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.core = core_mlp
        self.router = Router(d_model)
        self.gov_state = governor if governor else GovernorState()
        self.step_counter = 0

        self.experts = nn.ModuleList([Expert(d_model) for _ in range(init_experts)])
        if init_experts > 0:
            # Keep 1 reachable expert so system isn't dead
            self.experts[0].threshold.fill_(0.0)
            self.experts[0].mass.fill_(10.0)

        # Store activations for external training loop access
        self.last_acts: Dict[int, torch.Tensor] = {}
        # snapshot for spawn saturation check
        self._last_mass_snapshot = None

    @torch.no_grad()
    def update_mass(self, activations: Dict[int, torch.Tensor], utility_t: float):
        """m_i(t+1) = alpha * m_i(t) + g.sum() * utility"""
        for exp in self.experts:
            if exp.alive:
                exp.mass.mul_(exp.decay)

        for idx, g in activations.items():
            exp = self.experts[idx]
            # g might be [B, T, 1], we sum everything
            exp.mass.add_(g.sum() * utility_t)

    def can_learn(self, idx: int) -> bool:
        exp = self.experts[idx]
        return exp.alive and (exp.mass >= self.gov_state.m_min_learn)

    @torch.no_grad()
    def governor_step(self, phi_t: float):
        self.step_counter += 1
        self._enforce_dominance()
        self._prune_experts()
        self._maybe_spawn_expert(phi_t)
        self._charge_newborns()

    @torch.no_grad()
    def _enforce_dominance(self):
        alive = [e for e in self.experts if e.alive]
        if not alive: return
        
        total_mass = sum(e.mass.item() for e in alive)
        max_mass = max(e.mass.item() for e in alive)
        
        if total_mass > 0 and max_mass < self.gov_state.dominance_kappa * total_mass:
            masses = sorted([e.mass.item() for e in alive])
            median = masses[len(masses)//2]
            for e in alive:
                if e.mass.item() < median:
                    # Direct mass penalty
                    penalty = 0.05 * (median - e.mass.item())
                    e.mass.sub_(max(0.0, penalty))

    @torch.no_grad()
    def _prune_experts(self):
        alive = [e for e in self.experts if e.alive]
        if not alive: return
        dominant = max(alive, key=lambda x: x.mass.item())

        for e in self.experts:
            if not e.alive: continue
            
            if e.mass.item() < self.gov_state.epsilon_death:
                e.steps_low_mass += 1
            else:
                e.steps_low_mass = 0
                
            if e.steps_low_mass > self.gov_state.death_steps:
                # Redistribute 10% mass on death
                if e != dominant:
                    transfer = 0.1 * e.mass.item()
                    dominant.mass.add_(transfer)
                
                e.alive = False
                e.mass.zero_()

    @torch.no_grad()
    def _maybe_spawn_expert(self, phi_t: float):
        if (self.step_counter - self.gov_state.last_birth_step) < self.gov_state.birth_cooldown:
            return

        if self._last_mass_snapshot is None:
            self._last_mass_snapshot = {id(e): e.mass.item() for e in self.experts}

        max_growth = 0.0
        for e in self.experts:
            if not e.alive: continue
            prev = self._last_mass_snapshot.get(id(e), e.mass.item())
            growth = e.mass.item() - prev
            if growth > max_growth:
                max_growth = growth
        
        self._last_mass_snapshot = {id(e): e.mass.item() for e in self.experts}

        if phi_t > 0.7 and max_growth < self.gov_state.saturation_eps:
            new_expert = Expert(self.d_model)
            # Ensure it is on the same device AND dtype as others
            ref = self.experts[0].anchor
            new_expert.to(device=ref.device, dtype=ref.dtype)
            self.experts.append(new_expert)
            self.gov_state.last_birth_step = self.step_counter

    @torch.no_grad()
    def _charge_newborns(self):
        for e in self.experts:
            if e.alive and e.mass.item() == 0.0:
                if torch.isinf(e.threshold):
                    e.threshold.fill_(5.0)
                e.threshold.sub_(self.gov_state.newborn_charge_rate)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Returns RESIDUAL contribution: mlp(x) + sum(g_i * E_i(x))"""
        # 1. Core MLP (assume this returns residual contribution)
        core_res = self.core(x, *args, **kwargs)
        
        # 2. Routing
        acts = self.router(x, self.experts)
        self.last_acts = acts # Store for training loop
        
        # 3. Expert Accumulation
        exp_res = torch.zeros_like(core_res)
        for idx, g in acts.items():
            exp = self.experts[idx]
            # g is [B*T, 1] if x was 3D. We need to reshape g back to [B, T, 1] if x was [B, T, D]
            if x.dim() == 3:
                B, T, D = x.shape
                g_reshaped = g.view(B, T, 1)
                exp_res = exp_res + g_reshaped * exp(x)
            else:
                exp_res = exp_res + g * exp(x)
            
        return core_res + exp_res

# -----------------------------------------------------------------------------
# 3. Integration Utilities & Vectorized Metrics
# -----------------------------------------------------------------------------

@torch.no_grad()
def mg_blocks(model):
    for m in model.modules():
        if isinstance(m, MGMoEResidual):
            yield m

@torch.no_grad()
def mask_grads_for_nonlearning_experts(model):
    for block in mg_blocks(model):
        for i, e in enumerate(block.experts):
            if not block.can_learn(i):
                for p in e.parameters():
                    if p.grad is not None:
                        p.grad.zero_()

@torch.no_grad()
def mg_post_step(
    model,
    utility_mean: float,
    phi_mean: float,
    step: int,
    gov_interval: int = 20,
):
    for block in mg_blocks(model):
        acts = getattr(block, "last_acts", None)
        if acts is None:
            continue
        block.update_mass(acts, utility_mean)
        if (step % gov_interval) == 0:
            block.governor_step(phi_mean)

@torch.no_grad()
def compute_phi_utility_batch(
    logits: torch.Tensor,     # [B, T, V]
    input_ids: torch.Tensor,  # [B, T]
    pad_token_id: int = -100, # labels often use -100
    W: int = 64,
    eps: float = 1e-12,
):
    B, T, V = logits.shape

    # Next-token alignment: logits[:, :-1] predicts input_ids[:, 1:]
    logits_nt = logits[:, :-1, :]              # [B, T-1, V]
    y = input_ids[:, 1:]                       # [B, T-1]

    # mask padding
    mask = (y != pad_token_id).float()         # [B, T-1]

    # probs
    p = torch.softmax(logits_nt, dim=-1)       # [B, T-1, V]

    # entropy H_t
    H = -(p * (p + eps).log()).sum(dim=-1)     # [B, T-1]

    # top1 dominance
    top1 = p.max(dim=-1).values                # [B, T-1]

    # KL drift between consecutive steps (t vs t-1)
    p_prev = p[:, :-1, :]                      # [B, T-2, V]
    p_cur  = p[:, 1:, :]                       # [B, T-2, V]
    KL = (p_cur * ((p_cur + eps).log() - (p_prev + eps).log())).sum(dim=-1)  # [B, T-2]
    KL = torch.cat([torch.zeros(B, 1, device=logits.device), KL], dim=1)     # [B, T-1]

    # repetition proxy R_t
    R = torch.zeros_like(y, dtype=torch.float)  # [B, T-1]
    if W > 0:
        valid_range = torch.arange(T-1, device=logits.device)[None, :]
        for k in range(1, W + 1):
            shifted = torch.roll(y, shifts=k, dims=1)  # [B, T-1]
            valid = valid_range >= k
            R = torch.maximum(R, (y == shifted).float() * valid.float())

    # NLL per token
    nll = F.cross_entropy(
        logits_nt.reshape(-1, V),
        y.reshape(-1),
        reduction="none"
    ).reshape(B, T-1)

    # apply mask
    H   = H   * mask
    KL  = KL  * mask
    top1= top1* mask
    R   = R   * mask
    nll = nll * mask

    # normalize signals to [0,1] within batch
    def norm01(x):
        denom = (x.std() + 1e-6)
        z = ((x - x.mean()) / denom).clamp(-3, 3)
        return torch.sigmoid(z)

    Hn  = norm01(H)
    Rn  = norm01(R)
    KLn = norm01(KL)
    Sn  = norm01(top1)

    # φ: collapse pressure
    wH, wR, wKL, wS = 0.35, 0.30, 0.20, 0.15
    phi = (wH * (1 - Hn) + wR * Rn + wKL * KLn + wS * Sn).clamp(0, 1)

    # utility: nll drop
    nll_prev = torch.cat([nll[:, :1], nll[:, :-1]], dim=1)
    u = (nll_prev - nll).clamp(min=0.0)
    u = u / (u.mean() + 1e-6)
    u = (u.clamp(0, 3) / 3.0)
    u = u * (1.0 - R)

    return phi, u, mask

@torch.no_grad()
def reduce_signals(phi, u, mask):
    denom = mask.sum().clamp(min=1.0)
    phi_mean = (phi * mask).sum() / denom
    u_mean   = (u   * mask).sum() / denom
    return float(phi_mean.item()), float(u_mean.item())

def inject_mgmoe_into_ffn(
    model: torch.nn.Module,
    d_model: int,
    every_n_layers: int = 2,
    governor: GovernorState = None,
    mlp_attr_candidates=("mlp", "feed_forward", "ffn", "Mlp"),
):
    """
    Replace decoder-layer MLP modules with MGMoEResidual wrappers.
    """
    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        # Recursive search fallback if structure is unknown
        pass

    if layers is None:
        # Fallback recursive search
        for name, module in model.named_modules():
            if any(attr in name for attr in mlp_attr_candidates):
                # This is tricky for recursive without parent ref
                pass
        raise ValueError("Could not find decoder layers container.")

    injected_count = 0
    for i, layer in enumerate(layers):
        if (i % every_n_layers) != 0:
            continue

        mlp = None
        mlp_name = None
        for attr in mlp_attr_candidates:
            if hasattr(layer, attr):
                mlp = getattr(layer, attr)
                mlp_name = attr
                break
        
        if mlp is not None:
            wrapped = MGMoEResidual(d_model=d_model, core_mlp=mlp, governor=governor, init_experts=1)
            setattr(layer, mlp_name, wrapped)
            injected_count += 1

    return model, injected_count

def freeze_backbone_except_mgmoe(model: torch.nn.Module):
    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze MG-MoE components
    for m in model.modules():
        if isinstance(m, MGMoEResidual):
            # Router
            for p in m.router.parameters():
                p.requires_grad = True
            # Experts
            for exp in m.experts:
                for p in exp.parameters():
                    p.requires_grad = True

    return model

def apply_mg_gradient_updates(
    model: nn.Module, 
    optimizer: torch.optim.Optimizer
):
    """
    Masked gradient update: Only experts with sufficient mass can learn.
    """
    for m in model.modules():
        if isinstance(m, MGMoEResidual):
            for i, expert in enumerate(m.experts):
                if not m.can_learn(i):
                    expert.zero_grad()
    
    optimizer.step()
    optimizer.zero_grad()