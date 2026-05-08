import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def associative_scan_mg(base_scores, deposit_rate, alpha, lambda_mass):
    """
    Associative Scan implementation of Linear Memory Gravity.

    Recurrence:
      m_t = (alpha + lambda * eta_t) * m_{t-1} + eta_t * base_score_t

    This is a linear recurrence of the form:
      h_t = a_t * h_{t-1} + b_t
    where:
      a_t = alpha + lambda * eta_t
      b_t = eta_t * base_score_t
    """
    B, H, T, _ = base_scores.shape

    # We only care about the diagonal base_scores[t, t] for the recurrence?
    # No, each source token j has its own mass m_j(t).
    # m_j(t) = alpha * m_j(t-1) + r_{t,j} * eta_j
    # If r_{t,j} = base_score_{t,j} + lambda * m_j(t-1)
    # then m_j(t) = (alpha + lambda * eta_j) * m_j(t-1) + eta_j * base_score_{t,j}

    # a_j: [B, H, T]
    # b_{t,j}: [B, H, T, T]

    eta = deposit_rate # [B, 1, T] or [B, H, T]

    # a_j is constant over time t for a fixed source j?
    # In the original, eta_j is the deposit rate of source j.
    a = alpha + lambda_mass * eta # [B, H, T]

    # b_{t,j} = eta_j * base_score_{t,j}
    b = eta.unsqueeze(-2) * base_scores # [B, H, T, T]

    # We need to compute the scan over t (dim -2)
    # For a fixed source j, we have:
    # m_j(t) = a_j * m_j(t-1) + b_{t,j}

    # a_j is the same for all t.
    # m_j(t) = a_j^t * m_j(0) + sum_{k=1}^t a_j^{t-k} * b_{k,j}

    # Since a_j is constant over t, we can use a simpler power-based scan or cumulative sum in log space if a > 0.
    # Or just use the standard linear recurrence formula:
    # m_j(t) = sum_{k=0}^t (a_j ** (t-k)) * b_{k,j}

    # Precompute powers of a_j
    # log_a: [B, H, T]
    log_a = torch.log(a.clamp(min=1e-9))

    # t_indices: [T]
    t_indices = torch.arange(T, device=base_scores.device, dtype=base_scores.dtype)

    # log_a_sum: [B, H, T, T]
    # We want (t-k) * log_a_j
    # indices_diff = t - k
    indices_diff = t_indices.unsqueeze(1) - t_indices.unsqueeze(0) # [T, T]
    indices_diff = indices_diff.clamp(min=0)

    # weights = exp((t-k) * log_a_j) = a_j ** (t-k)
    # [B, H, T(source), T(time)]
    weights = torch.exp(indices_diff.unsqueeze(0).unsqueeze(0) * log_a.unsqueeze(-2))

    # Apply causal mask to weights (k <= t)
    causal_mask = torch.tril(torch.ones(T, T, device=base_scores.device, dtype=torch.bool))
    weights = weights.masked_fill(~causal_mask.transpose(0, 1).unsqueeze(0).unsqueeze(0), 0.0)

    # m_j(t) = sum_k weights_{j,t,k} * b_{k,j}
    # weights: [B, H, T_j, T_t]
    # b: [B, H, T_t, T_j]

    # We need to sum over k (T_t in b)
    # Let's align:
    # b_transposed: [B, H, T_j, T_t]
    b_trans = b.transpose(-1, -2)

    # mass_j(t) = (weights * b_trans).sum(dim=-1)
    # This is still O(T^2) but fully parallelized as a matmul/pointwise product.
    mass_hist = (weights * b_trans).transpose(-1, -2) # [B, H, T_t, T_j]

    # Now compute effective attention:
    # r_{t,j} = base_score_{t,j} + lambda * m_j(t-1)
    # m_j(t-1) is mass_hist at t-1
    m_prev = torch.cat([torch.zeros(B, H, 1, T, device=base_scores.device, dtype=base_scores.dtype), mass_hist[:, :, :-1, :]], dim=2)

    scores = base_scores + lambda_mass * m_prev

    # For Sigmoid Gravity, we use sigmoid instead of softmax for O(log T) independence
    effective = torch.sigmoid(scores)

    return effective, mass_hist

class ScanMemoryGravityHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.H = cfg.n_heads
        self.d_model = cfg.d_model
        self.d_h = cfg.d_model // self.H

        self.W_Q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_E = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_A = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, glyph_mask=None):
        B, T, D = x.shape
        H, d_h = self.H, self.d_h

        Q = self.W_Q(x).view(B, T, H, d_h).transpose(1, 2)
        E = self.W_E(x).view(B, T, H, d_h).transpose(1, 2)
        A = self.W_A(x).view(B, T, H, d_h).transpose(1, 2)

        base_scores = torch.matmul(Q, E.transpose(-1, -2)) / math.sqrt(d_h)

        if glyph_mask is None:
            deposit_rate = torch.ones(B, 1, T, device=x.device, dtype=x.dtype)
        else:
            deposit_rate = 1.0 + (self.cfg.glyph_deposit - 1.0) * glyph_mask.view(B, 1, T).to(x.dtype)

        effective, mass_hist = associative_scan_mg(
            base_scores, deposit_rate, self.cfg.alpha, self.cfg.lambda_mass
        )

        delta = torch.matmul(effective, A)
        delta = delta.transpose(1, 2).reshape(B, T, D)
        y = x + self.out_proj(delta)

        return y, effective, mass_hist
