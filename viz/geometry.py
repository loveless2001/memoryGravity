"""
Geometry computations for the Dynamic Semantic Trajectory Visualizer.

This module is intentionally minimal for Phase 0 (the falsification spike).
It implements:

  - per-step speed:              ||h_{t+1} - h_t||_2
  - raw arccos curvature:        angle between v_t and v_{t+1}
  - null baseline distribution:  curvatures of *shuffled* step-vector pairs
  - null-calibrated quantile:    rank of each true curvature against the null

The reason raw arccos angles are not reported as the main signal is that in
high-dimensional residual spaces, two random vectors are nearly orthogonal
by default (cos approximately 0, angle approximately pi/2). So every step
"looks curvy" against a naive zero baseline. The null-calibrated quantile
asks instead: how unusual is this turn relative to angles between random
*pairs of step vectors from this same prompt*.

Phase 2 frame construction (local step-vector SVD, Procrustes alignment,
stall handling) is *not* implemented here yet. Stubs are provided so the
artifact contract has placeholders, but they raise NotImplementedError.
"""

from __future__ import annotations

import numpy as np

# Step speeds below this threshold mean v_t is unreliable and the tangent
# direction is dominated by floating-point / model noise, not real motion.
# Used to populate `stall_mask` in the trace artifact. Threshold expressed
# as a fraction of the median step speed within the prompt.
STALL_SPEED_FRACTION = 0.1


def step_vectors(hidden_states: np.ndarray) -> np.ndarray:
    """v_t = h_{t+1} - h_t.

    `hidden_states` shape: (T, d) for a single layer. Returns shape (T-1, d).
    """
    return hidden_states[1:] - hidden_states[:-1]


def step_speeds(v: np.ndarray) -> np.ndarray:
    """||v_t||_2 per step. Shape (T-1,)."""
    return np.linalg.norm(v.astype(np.float32), axis=-1)


def stall_mask(speeds: np.ndarray,
               fraction: float = STALL_SPEED_FRACTION) -> np.ndarray:
    """Boolean mask marking step indices whose speed is below `fraction * median`.

    Used by Phase 2 frame transport to hold the prior frame instead of
    recomputing a tangent from a near-zero vector.
    """
    if speeds.size == 0:
        return np.zeros(0, dtype=bool)
    threshold = fraction * float(np.median(speeds))
    return speeds < max(threshold, 1e-8)


def raw_curvatures(v: np.ndarray) -> np.ndarray:
    """arccos(cos(v_t, v_{t+1})) per consecutive step pair. Shape (T-2,).

    Returns radians. Reported only as raw input to the null-calibrated
    quantile; not reported as the headline metric.
    """
    v = v.astype(np.float32)
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    safe_norms = np.where(norms < 1e-12, 1.0, norms)
    unit = v / safe_norms
    cosines = (unit[:-1] * unit[1:]).sum(axis=-1)
    cosines = np.clip(cosines, -1.0, 1.0)
    return np.arccos(cosines)


def _shuffled_pair_curvatures(v: np.ndarray, n_samples: int,
                              rng: np.random.Generator) -> np.ndarray:
    """Curvatures from random non-adjacent step-vector pairs in the same prompt.

    Used as the within-prompt null distribution: keeps the magnitude /
    direction statistics of the actual prompt but breaks the temporal
    adjacency that real curvature is supposed to capture.
    """
    n = v.shape[0]
    if n < 2:
        return np.zeros(0, dtype=np.float32)
    v = v.astype(np.float32)
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    safe_norms = np.where(norms < 1e-12, 1.0, norms)
    unit = v / safe_norms
    i = rng.integers(0, n, size=n_samples)
    j = rng.integers(0, n, size=n_samples)
    same = i == j
    if np.any(same):
        # Resample collisions so we never take a step's curvature against itself
        # (which would be 0 and bias the null toward small angles).
        for k in np.where(same)[0]:
            while i[k] == j[k]:
                j[k] = rng.integers(0, n)
    cosines = (unit[i] * unit[j]).sum(axis=-1)
    cosines = np.clip(cosines, -1.0, 1.0)
    return np.arccos(cosines)


def null_calibrated_curvatures(hidden_states: np.ndarray,
                               n_null_samples: int = 4096,
                               seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw curvatures into within-prompt null quantiles.

    Returns (curvatures_q, raw) where:
      curvatures_q[t] = empirical CDF of the null distribution at the true
                        curvature kappa_t, in [0, 1]. Higher = sharper
                        relative to non-adjacent pairs in the same prompt.
      raw[t]          = the underlying arccos value in radians.

    Both have shape (T-2,) for a (T, d) hidden trajectory.
    """
    v = step_vectors(hidden_states)
    raw = raw_curvatures(v)
    rng = np.random.default_rng(seed)
    null = _shuffled_pair_curvatures(v, n_null_samples, rng)
    if null.size == 0 or raw.size == 0:
        return raw.astype(np.float32), raw.astype(np.float32)
    # Empirical CDF: fraction of null samples <= each true curvature.
    null_sorted = np.sort(null)
    quantiles = np.searchsorted(null_sorted, raw, side="right") / null.size
    return quantiles.astype(np.float32), raw.astype(np.float32)


def compute_geometry(hidden_states: np.ndarray,
                     n_null_samples: int = 4096,
                     seed: int = 0) -> dict:
    """One-shot helper used by the spike orchestrator and trace saver.

    Input: (T, d) hidden states for ONE layer (caller selects the layer).
    Output dict matches the npz schema fields in the artifact contract:
      step_speeds:   (T-1,) float32
      curvatures_q:  (T-2,) float32  (null-calibrated quantile in [0,1])
      curvatures_raw:(T-2,) float32  (radians; for diagnostics only)
      stall_mask:    (T-1,) bool
      null_method:   str    (description for trace.json metadata)
    """
    v = step_vectors(hidden_states)
    speeds = step_speeds(v)
    stalls = stall_mask(speeds)
    quantiles, raw = null_calibrated_curvatures(
        hidden_states, n_null_samples=n_null_samples, seed=seed
    )
    return {
        "step_speeds": speeds.astype(np.float32),
        "curvatures_q": quantiles,
        "curvatures_raw": raw,
        "stall_mask": stalls,
        "null_method": (
            f"within_prompt_shuffled_step_pairs:n={n_null_samples}:seed={seed}"
        ),
    }


# -- Phase 2 stubs (kept here so the artifact contract has named slots) --

def build_local_frame(*_args, **_kwargs):  # pragma: no cover - Phase 2 work
    """Local step-vector SVD frame. Phase 2 — not in spike scope."""
    raise NotImplementedError("Local frame construction is Phase 2.")


def transport_frames(*_args, **_kwargs):  # pragma: no cover - Phase 2 work
    """Procrustes / sign-aligned frame transport. Phase 2 — not in spike scope."""
    raise NotImplementedError("Frame transport is Phase 2.")
