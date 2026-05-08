import torch
import torch.nn as nn
import torch.nn.functional as F
import math

@torch.jit.script
def mg_head_loop(
    base_scores: torch.Tensor,
    deposit_rate: torch.Tensor,
    alpha: float,
    lambda_mass: float,
    use_mass_weighting: bool,
    return_mass: bool
):
    B, T, _ = base_scores.shape
    device = base_scores.device
    dtype = base_scores.dtype

    effective = torch.zeros((B, T, T), device=device, dtype=dtype)
    mass_hist = torch.empty((B, T, T), device=device, dtype=dtype) if return_mass else torch.empty((0,0,0), device=device)

    prev_mass = torch.zeros((B, T), device=device, dtype=dtype)

    for t in range(T):
        # row: [B, T]
        row = base_scores[:, t, :t+1]

        if use_mass_weighting:
            row = row + lambda_mass * prev_mass[:, :t+1]

        r_t_valid = F.softmax(row, dim=-1) # [B, t+1]

        effective[:, t, :t+1] = r_t_valid

        # Mass update
        # new_mass = alpha * prev_mass + deposit_rate * r_t
        # We only need to update up to t+1
        prev_mass[:, :t+1] = alpha * prev_mass[:, :t+1] + deposit_rate[:, :t+1] * r_t_valid

        if return_mass:
            mass_hist[:, t, :] = prev_mass

    return effective, mass_hist

def fast_forward(self, x, glyph_mask=None, return_attn=True, return_mass=True, return_scores=False):
    B, T, D = x.shape
    E = self.W_E(x)
    A = E if self.cfg.tie_emission_assimilation else self.W_A(x)

    base_scores = torch.matmul(x, E.transpose(-1, -2)) / math.sqrt(D)

    if glyph_mask is None or not self.cfg.use_glyphs:
        deposit_rate = torch.ones(B, T, device=x.device, dtype=x.dtype)
    else:
        deposit_rate = 1.0 + (self.cfg.glyph_deposit - 1.0) * glyph_mask.to(x.dtype)

    effective, mass_hist = mg_head_loop(
        base_scores,
        deposit_rate,
        self.cfg.alpha,
        self.cfg.lambda_mass,
        self.cfg.use_mass_weighting,
        return_mass
    )

    delta = torch.matmul(effective, A)
    y = x + self.out_proj(delta)

    attn = effective if return_attn else None
    mass_hist = mass_hist if return_mass else None
    scores = base_scores if return_scores else None

    return y, attn, mass_hist, scores