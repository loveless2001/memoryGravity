import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .mg_triton import mg_head_triton
    HAS_TRITON = True
except (ImportError, ValueError):
    try:
        from mg_triton import mg_head_triton
        HAS_TRITON = True
    except ImportError:
        HAS_TRITON = False


@dataclass
class MGConfig:
    vocab_size: int = 128
    d_model: int = 32
    hidden_dim: int = 64
    max_seq_len: int = 64

    # Refined Memory Gravity dynamics
    alpha: float = 0.95
    lambda_mass: float = 1.0
    glyph_deposit: float = 2.0

    # Backward-compat fields (optional aliases)
    beta: float | None = None
    glyph_boost: float | None = None

    # Behavior toggles
    use_mass_weighting: bool = True
    use_glyphs: bool = True
    # Kept for compatibility with existing scripts; refined head uses additive logits.
    mass_to_logits: bool = True

    # Mass mode: "additive" (original), "query_gated" (new), "none" (disabled)
    mass_mode: str = "additive"

    # Local windowed attention: 0 = full causal, >0 = attend only within last N positions
    # For MG query_gated: base attention is windowed, mass provides reach beyond window
    local_window: int = 0

    # Role-based mass modulation: allow continuous deposit strength based on token role
    use_role_modulation: bool = False

    # Architecture options
    n_heads: int = 1       # Multi-head MG attention: each head has independent mass dynamics
    n_layers: int = 1      # Number of stacked MemoryGravityBlocks
    dropout: float = 0.0   # Dropout between layers (useful when n_layers > 1)
    tie_emission_assimilation: bool = False
    use_layernorm: bool = True
    use_fast_path: bool = True
    use_triton: bool = True
    use_mass_ln: bool = False

    def __post_init__(self):
        if self.beta is not None:
            self.lambda_mass = self.beta
        if self.glyph_boost is not None:
            self.glyph_deposit = self.glyph_boost
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        assert self.mass_mode in ("additive", "query_gated", "none"), \
            f"mass_mode must be 'additive', 'query_gated', or 'none', got '{self.mass_mode}'"
        # Backward compat: mass_mode="none" disables mass weighting
        if self.mass_mode == "none":
            self.use_mass_weighting = False


@torch.jit.script
def mg_head_loop(
    base_scores: torch.Tensor,
    deposit_rate: torch.Tensor,
    alpha: float,
    lambda_mass: float,
    use_mass_weighting: bool,
    use_mass_ln: bool,
    return_mass: bool,
    return_scores: bool,
):
    B, H, T, _ = base_scores.shape
    device = base_scores.device
    dtype = base_scores.dtype

    effective = torch.zeros((B, H, T, T), device=device, dtype=dtype)
    mass_hist = (
        torch.zeros((B, H, T, T), device=device, dtype=dtype)
        if return_mass
        else torch.empty((0, 0, 0, 0), device=device, dtype=dtype)
    )
    scores_out = (
        torch.full((B, H, T, T), float("-inf"), device=device, dtype=dtype)
        if return_scores
        else torch.empty((0, 0, 0, 0), device=device, dtype=dtype)
    )

    prev_mass = torch.zeros((B, H, T), device=device, dtype=dtype)

    for t in range(T):
        row = base_scores[:, :, t, : t + 1]

        if use_mass_weighting:
            mass_term = prev_mass[:, :, : t + 1]
            if use_mass_ln and t > 0:
                # Manual layer norm across T for each head
                m = mass_term.mean(dim=-1, keepdim=True)
                v = mass_term.var(dim=-1, keepdim=True, unbiased=False)
                mass_term = (mass_term - m) / torch.sqrt(v + 1e-5)
            row = row + lambda_mass * mass_term

        r_t_valid = F.softmax(row, dim=-1)
        effective[:, :, t, : t + 1] = r_t_valid

        # deposit_rate: [B, 1, T] or [B, H, T]
        prev_mass[:, :, : t + 1] = alpha * prev_mass[:, :, : t + 1] + deposit_rate[:, :, : t + 1] * r_t_valid

        if return_mass:
            mass_hist[:, :, t, : t + 1] = prev_mass[:, :, : t + 1]
        if return_scores:
            scores_out[:, :, t, : t + 1] = row

    return effective, mass_hist, scores_out


def mg_head_loop_query_gated(
    base_scores: torch.Tensor,
    deposit_rate: torch.Tensor,
    mass_keys: torch.Tensor,
    queries: torch.Tensor,
    alpha: float,
    lambda_mass: float,
    inv_sqrt_d: float,
    return_mass: bool,
    return_scores: bool,
    local_window: int = 0,
):
    """Query-gated mass: gate = sigmoid(q_t · mk_j / sqrt(d_h)), bonus = gate * m_j.
    Optimized: precomputes gates via batched matmul, precomputes window mask,
    writes directly to output tensors (no list accumulation).
    When local_window > 0: base attention is windowed, mass provides reach beyond window.
    In-window positions: score = base_score + mass_bonus.
    Out-of-window positions: score = mass_bonus only (mass is the sole pathway)."""
    B, H, T, _ = base_scores.shape
    device = base_scores.device
    dtype = base_scores.dtype

    # Precompute ALL gates as a single batched matmul: [B, H, T, T]
    # gate[b,h,t,j] = sigmoid(q_t · mk_j / sqrt(d_h))
    all_gates = torch.sigmoid(
        torch.matmul(queries, mass_keys.transpose(-1, -2)) * inv_sqrt_d
    )

    # Precompute windowed base scores if local_window is active
    if local_window > 0:
        pos = torch.arange(T, device=device)
        i_pos = pos.unsqueeze(1)  # [T, 1]
        j_pos = pos.unsqueeze(0)  # [1, T]
        in_window_mask = (j_pos <= i_pos) & (i_pos - j_pos < local_window)
        # Zero out-of-window base (not -inf, mass provides pathway there)
        windowed_base = base_scores * in_window_mask.unsqueeze(0).unsqueeze(0).to(dtype)
    else:
        windowed_base = base_scores

    # Pre-allocate output tensors (direct writes, no list accumulation)
    effective = torch.zeros((B, H, T, T), device=device, dtype=dtype)
    mass_hist = torch.zeros((B, H, T, T), device=device, dtype=dtype) if return_mass else None
    scores_out = torch.full((B, H, T, T), float("-inf"), device=device, dtype=dtype) if return_scores else None

    prev_mass = torch.zeros((B, H, T), device=device, dtype=dtype)

    for t in range(T):
        # Mass bonus from precomputed gates: [B, H, t+1]
        mass_bonus = lambda_mass * all_gates[:, :, t, :t + 1] * prev_mass[:, :, :t + 1]

        # Row = windowed base + mass bonus
        row = windowed_base[:, :, t, :t + 1] + mass_bonus

        # Softmax over causal positions
        r_t = F.softmax(row, dim=-1)
        effective[:, :, t, :t + 1] = r_t

        # Non-inplace mass update for autograd
        new_prefix = alpha * prev_mass[:, :, :t + 1] + deposit_rate[:, :, :t + 1] * r_t
        if t + 1 < T:
            prev_mass = torch.cat([new_prefix, prev_mass[:, :, t + 1:]], dim=-1)
        else:
            prev_mass = new_prefix

        if return_mass:
            mass_hist[:, :, t, :t + 1] = prev_mass[:, :, :t + 1]
        if return_scores:
            scores_out[:, :, t, :t + 1] = row

    return (
        effective,
        mass_hist if mass_hist is not None else torch.empty((0, 0, 0, 0), device=device, dtype=dtype),
        scores_out if scores_out is not None else torch.empty((0, 0, 0, 0), device=device, dtype=dtype),
    )


class MemoryGravityHead(nn.Module):
    """
    Refined Multi-Head Memory Gravity head.
    """

    def __init__(self, cfg: MGConfig):
        super().__init__()
        self.cfg = cfg
        self.H = cfg.n_heads
        self.d_model = cfg.d_model
        self.d_h = cfg.d_model // self.H

        self.W_Q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_E = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_A = (
            None
            if cfg.tie_emission_assimilation
            else nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        )
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # Mass key projection for query-gated mode
        self.W_MK = (
            nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            if cfg.mass_mode == "query_gated"
            else None
        )

        self.register_buffer(
            "_causal_mask",
            torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool)),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        glyph_mask: torch.Tensor | None = None,
        role_mask: torch.Tensor | None = None,
        return_attn: bool = True,
        return_mass: bool = True,
        return_scores: bool = False,
    ):
        B, T, D = x.shape
        H, d_h = self.H, self.d_h

        Q = self.W_Q(x).view(B, T, H, d_h).transpose(1, 2)  # [B, H, T, d_h]
        E = self.W_E(x).view(B, T, H, d_h).transpose(1, 2)  # [B, H, T, d_h]
        A = (
            E
            if self.cfg.tie_emission_assimilation
            else self.W_A(x).view(B, T, H, d_h).transpose(1, 2)
        )  # [B, H, T, d_h]

        # base_scores: [B, H, T, T]
        base_scores = torch.matmul(Q, E.transpose(-1, -2)) / math.sqrt(d_h)

        if role_mask is not None:
            # Continuous deposit modulation
            deposit_rate = 1.0 + (self.cfg.glyph_deposit - 1.0) * role_mask.view(
                B, 1, T
            ).to(x.dtype)
        elif glyph_mask is not None and self.cfg.use_glyphs:
            # Traditional binary deposit modulation
            deposit_rate = 1.0 + (self.cfg.glyph_deposit - 1.0) * glyph_mask.view(
                B, 1, T
            ).to(x.dtype)
        else:
            deposit_rate = torch.ones(B, 1, T, device=x.device, dtype=x.dtype)

        # Apply windowed causal mask for non-query_gated paths
        # (query_gated handles windowing internally via two-pathway approach)
        if self.cfg.local_window > 0 and self.cfg.mass_mode != "query_gated":
            i_pos = torch.arange(T, device=x.device).unsqueeze(1)
            j_pos = torch.arange(T, device=x.device).unsqueeze(0)
            window_mask = (j_pos <= i_pos) & (i_pos - j_pos < self.cfg.local_window)
            base_scores = base_scores.masked_fill(
                ~window_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        # Query-gated mass: uses dedicated loop with mass keys
        if self.cfg.mass_mode == "query_gated":
            MK = self.W_MK(x).view(B, T, H, d_h).transpose(1, 2)  # [B, H, T, d_h]
            effective, mass_hist_tensor, scores_tensor = mg_head_loop_query_gated(
                base_scores,
                deposit_rate,
                MK,
                Q,
                self.cfg.alpha,
                self.cfg.lambda_mass,
                1.0 / math.sqrt(d_h),
                return_mass,
                return_scores,
                self.cfg.local_window,
            )
            mass_hist = mass_hist_tensor if return_mass else None
            scores = scores_tensor if return_scores else None
        elif self.cfg.use_triton and HAS_TRITON and x.is_cuda and self.cfg.mass_mode == "additive":
            effective, mass_hist_tensor, scores_tensor = mg_head_triton(
                base_scores,
                deposit_rate,
                self.cfg.alpha,
                self.cfg.lambda_mass,
                self.cfg.use_mass_weighting,
                self.cfg.use_mass_ln,
                return_mass,
                return_scores,
            )
            mass_hist = mass_hist_tensor if return_mass else None
            scores = scores_tensor if return_scores else None
        elif self.cfg.use_fast_path:
            effective, mass_hist_tensor, scores_tensor = mg_head_loop(
                base_scores,
                deposit_rate,
                self.cfg.alpha,
                self.cfg.lambda_mass,
                self.cfg.use_mass_weighting,
                self.cfg.use_mass_ln,
                return_mass,
                return_scores,
            )
            mass_hist = mass_hist_tensor if return_mass else None
            scores = scores_tensor if return_scores else None
        else:
            # Fallback slow path
            causal = self._causal_mask[:T, :T]
            effective = torch.zeros(B, H, T, T, device=x.device, dtype=x.dtype)
            mass_hist = (
                torch.zeros(B, H, T, T, device=x.device, dtype=x.dtype)
                if return_mass
                else None
            )
            scores = (
                torch.full((B, H, T, T), float("-inf"), device=x.device, dtype=x.dtype)
                if return_scores
                else None
            )

            prev_mass = torch.zeros(B, H, T, device=x.device, dtype=x.dtype)

            for t in range(T):
                row = base_scores[:, :, t, :]

                if self.cfg.use_mass_weighting:
                    mass_term = prev_mass
                    if self.cfg.use_mass_ln and t > 0:
                        m = mass_term[:, :, : t + 1].mean(dim=-1, keepdim=True)
                        v = mass_term[:, :, : t + 1].var(
                            dim=-1, keepdim=True, unbiased=False
                        )
                        mass_term = (mass_term - m) / torch.sqrt(v + 1e-5)
                    row = row + self.cfg.lambda_mass * mass_term

                valid = causal[t].view(1, 1, T)
                row = row.masked_fill(~valid, float("-inf"))
                r_t = F.softmax(row, dim=-1)
                effective[:, :, t, :] = r_t

                if self.cfg.use_mass_weighting or return_mass:
                    new_mass = self.cfg.alpha * prev_mass + deposit_rate * r_t
                    new_mass = new_mass * valid.to(x.dtype)
                    prev_mass = new_mass
                    if return_mass:
                        mass_hist[:, :, t, :] = new_mass

                if return_scores:
                    scores[:, :, t, :] = row

        # Assimilation
        delta = torch.matmul(effective, A)  # [B, H, T, d_h]
        delta = delta.transpose(1, 2).reshape(B, T, D)  # [B, T, D]
        y = x + self.out_proj(delta)

        attn = effective if return_attn else None
        return y, attn, mass_hist, scores


class TinyMLP(nn.Module):
    def __init__(self, cfg: MGConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.hidden_dim)
        self.fc2 = nn.Linear(cfg.hidden_dim, cfg.d_model)

    def forward(self, x: torch.Tensor):
        return self.fc2(F.gelu(self.fc1(x)))


class MemoryGravityBlock(nn.Module):
    def __init__(self, cfg: MGConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model) if cfg.use_layernorm else nn.Identity()
        self.ln2 = nn.LayerNorm(cfg.d_model) if cfg.use_layernorm else nn.Identity()
        self.head = MemoryGravityHead(cfg)
        self.mlp = TinyMLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        glyph_mask: torch.Tensor | None = None,
        role_mask: torch.Tensor | None = None,
        return_attn: bool = True,
        return_mass: bool = True,
        return_scores: bool = False,
    ):
        h, attn, mass, scores = self.head(
            self.ln1(x),
            glyph_mask=glyph_mask,
            role_mask=role_mask,
            return_attn=return_attn,
            return_mass=return_mass,
            return_scores=return_scores,
        )
        x = h
        x = x + self.mlp(self.ln2(x))
        return x, attn, mass, scores


class TinyMemoryGravityLM(nn.Module):
    def __init__(self, cfg: MGConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([MemoryGravityBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model) if cfg.use_layernorm else nn.Identity()
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.register_buffer("_pos_ids", torch.arange(cfg.max_seq_len, dtype=torch.long), persistent=False)

        # optional tied weights
        self.head.weight = self.tok_emb.weight

    def forward(
        self,
        idx: torch.Tensor,
        glyph_mask: torch.Tensor | None = None,
        role_mask: torch.Tensor | None = None,
        targets=None,
        return_attn: bool = True,
        return_mass: bool = True,
        return_scores: bool = False,
    ):
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len

        pos = self._pos_ids[:T]
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.dropout(x)

        attn_list, mass_list, scores_list = [], [], []
        for block in self.blocks:
            x, attn, mass, scores = block(
                x,
                glyph_mask=glyph_mask,
                role_mask=role_mask,
                return_attn=return_attn,
                return_mass=return_mass,
                return_scores=return_scores,
            )
            if return_attn:
                attn_list.append(attn)
            if return_mass:
                mass_list.append(mass)
            if return_scores:
                scores_list.append(scores)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                targets[:, 1:].reshape(-1),
            )

        return {
            "logits": logits,
            "loss": loss,
            "attn": attn_list if return_attn else None,
            "mass": mass_list if return_mass else None,
            "scores": scores_list if return_scores else None,
        }

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, glyph_token_ids: set[int] | None = None):
        token_ids = None
        if glyph_token_ids is not None:
            token_ids = torch.tensor(sorted(glyph_token_ids), device=idx.device, dtype=idx.dtype)

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len:]
            glyph_mask = None
            if token_ids is not None:
                glyph_mask = torch.isin(idx_cond, token_ids).to(torch.float32)

            out = self(idx_cond, glyph_mask=glyph_mask, return_attn=False, return_mass=False)
            next_token_logits = out["logits"][:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx


if __name__ == "__main__":
    torch.manual_seed(42)

    cfg = MGConfig(
        vocab_size=32,
        d_model=16,
        hidden_dim=32,
        max_seq_len=16,
        alpha=0.9,
        lambda_mass=0.75,
        glyph_deposit=2.5,
        tie_emission_assimilation=False,
        use_layernorm=True,
        use_mass_weighting=True,
        use_glyphs=True,
    )

    model = TinyMemoryGravityLM(cfg)

    idx = torch.tensor([
        [1, 7, 3, 7, 5, 0, 0, 0],
        [2, 4, 4, 9, 6, 0, 0, 0],
    ], dtype=torch.long)

    glyph_mask = (idx == 7).float()

    out = model(idx, glyph_mask=glyph_mask, targets=idx, return_scores=True)
    print("loss:", out["loss"].item())
    print("attn[0] shape:", out["attn"][0].shape)
    print("mass[0] shape:", out["mass"][0].shape)
    print("scores[0] shape:", out["scores"][0].shape)

    prompt = torch.tensor([[1, 7, 3]], dtype=torch.long)
    gen = model.generate(prompt, max_new_tokens=5, glyph_token_ids={7})
    print("generated:", gen.tolist())
