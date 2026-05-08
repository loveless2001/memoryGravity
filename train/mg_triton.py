import torch
import triton
import triton.language as tl
import math

# Memory Gravity Triton Kernel
# Implements the sequential recurrence:
# r_t = softmax(base_scores[t, :t+1] + lambda * mass[t-1, :t+1])
# mass[t, :t+1] = alpha * mass[t-1, :t+1] + deposit_rate[:t+1] * r_t

@triton.jit
def mg_head_kernel(
    base_scores_ptr,
    deposit_rate_ptr,
    effective_ptr,
    mass_hist_ptr,
    scores_out_ptr,
    B, H, T,
    alpha,
    lambda_mass,
    stride_bs_b, stride_bs_h, stride_bs_t, stride_bs_j,
    stride_dr_b, stride_dr_h, stride_dr_t,
    stride_eff_b, stride_eff_h, stride_eff_t, stride_eff_j,
    stride_mh_b, stride_mh_h, stride_mh_t, stride_mh_j,
    stride_so_b, stride_so_h, stride_so_t, stride_so_j,
    use_mass_weighting: tl.constexpr,
    use_mass_ln: tl.constexpr,
    return_mass: tl.constexpr,
    return_scores: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (batch, head)
    pid_bh = tl.program_id(0)
    bid = pid_bh // H
    hid = pid_bh % H

    # Offset pointers
    bs_ptr = base_scores_ptr + bid * stride_bs_b + hid * stride_bs_h
    dr_ptr = deposit_rate_ptr + bid * stride_dr_b + hid * stride_dr_h
    eff_ptr = effective_ptr + bid * stride_eff_b + hid * stride_eff_h
    mh_ptr = mass_hist_ptr + bid * stride_mh_b + hid * stride_mh_h
    so_ptr = scores_out_ptr + bid * stride_so_b + hid * stride_so_h

    # We need a vector to store the masses in SRAM
    # BLOCK_SIZE must be >= T
    mass = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    indices = tl.arange(0, BLOCK_SIZE)

    for t in range(T):
        # 1. Load base scores for current t
        row_bs = tl.load(bs_ptr + t * stride_bs_t + indices * stride_bs_j, mask=indices <= t, other=float("-inf"))

        # 2. Add mass bias
        if use_mass_weighting:
            mass_term = mass
            if use_mass_ln and t > 0:
                # Manual LayerNorm across T for this head
                mask = indices <= t
                count = t + 1.0
                m = tl.sum(tl.where(mask, mass_term, 0.0), axis=0) / count
                v = tl.sum(tl.where(mask, (mass_term - m) * (mass_term - m), 0.0), axis=0) / count
                mass_term = (mass_term - m) / tl.sqrt(v + 1e-5)

            row_bs = row_bs + lambda_mass * mass_term

        # 3. Softmax
        # Manual softmax for numerical stability
        max_val = tl.max(tl.where(indices <= t, row_bs, float("-inf")), axis=0)
        exp_row = tl.exp(tl.where(indices <= t, row_bs - max_val, float("-inf")))
        sum_exp = tl.sum(exp_row, axis=0)
        r_t = exp_row / sum_exp

        # 4. Store effective attention
        tl.store(eff_ptr + t * stride_eff_t + indices * stride_eff_j, r_t, mask=indices <= t)

        # 5. Update scores_out if needed
        if return_scores:
            tl.store(so_ptr + t * stride_so_t + indices * stride_so_j, row_bs, mask=indices <= t)

        # 6. Update mass
        # row_dr: [T]
        row_dr = tl.load(dr_ptr + indices * stride_dr_t, mask=indices <= t, other=0.0)
        mass = alpha * mass + row_dr * r_t

        # 7. Store mass_hist if needed
        if return_mass:
            tl.store(mh_ptr + t * stride_mh_t + indices * stride_mh_j, mass, mask=indices <= t)

def mg_head_triton(
    base_scores, deposit_rate, alpha, lambda_mass,
    use_mass_weighting=True, use_mass_ln=False,
    return_mass=True, return_scores=True
):
    B, H, T, _ = base_scores.shape
    device = base_scores.device
    dtype = base_scores.dtype

    # Pad T to next power of 2 for Triton
    BLOCK_SIZE = triton.next_power_of_2(T)

    effective = torch.zeros((B, H, T, T), device=device, dtype=dtype)
    mass_hist = torch.zeros((B, H, T, T), device=device, dtype=dtype) if return_mass else torch.empty((0,0,0,0), device=device, dtype=dtype)
    scores_out = torch.full((B, H, T, T), float("-inf"), device=device, dtype=dtype) if return_scores else torch.empty((0,0,0,0), device=device, dtype=dtype)

    # Handle deposit_rate broadcasting
    # deposit_rate: [B, 1, T] -> stride_dr_h = 0
    if deposit_rate.shape[1] == 1:
        stride_dr_h = 0
    else:
        stride_dr_h = deposit_rate.stride(1)

    grid = (B * H,)

    # Ensure inputs are contiguous or handle strides correctly
    # Triton prefers pointers to be aligned
    mg_head_kernel[grid](
        base_scores, deposit_rate, effective, mass_hist, scores_out,
        B, H, T,
        alpha, lambda_mass,
        base_scores.stride(0), base_scores.stride(1), base_scores.stride(2), base_scores.stride(3),
        deposit_rate.stride(0), stride_dr_h, deposit_rate.stride(2),
        effective.stride(0), effective.stride(1), effective.stride(2), effective.stride(3),
        mass_hist.stride(0), mass_hist.stride(1), mass_hist.stride(2), mass_hist.stride(3),
        scores_out.stride(0), scores_out.stride(1), scores_out.stride(2), scores_out.stride(3),
        use_mass_weighting=use_mass_weighting,
        use_mass_ln=use_mass_ln,
        return_mass=return_mass,
        return_scores=return_scores,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return effective, mass_hist, scores_out