from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product

import torch


PAD_TOKEN_ID = 0
QUERY_TOKEN_ID = 1
GLYPH_START_ID = 2
NUM_GLYPHS = 8
VALUE_START_ID = 10
NUM_VALUES = 32
FILLER_START_ID = 42
VOCAB_SIZE = 128


@dataclass(frozen=True)
class ArenaCondition:
    n_bindings: int
    query_delay: int
    distractor_rate: float
    value_collision: bool

    def to_dict(self) -> dict:
        return asdict(self)


def parse_int_levels(spec: str) -> list[int]:
    return [int(part) for part in spec.split(",") if part.strip()]


def parse_float_levels(spec: str) -> list[float]:
    return [float(part) for part in spec.split(",") if part.strip()]


def parse_bool_levels(spec: str) -> list[bool]:
    values = []
    for part in spec.split(","):
        value = part.strip().lower()
        if not value:
            continue
        if value in {"true", "1", "yes"}:
            values.append(True)
        elif value in {"false", "0", "no"}:
            values.append(False)
        else:
            raise ValueError(f"invalid bool level: {part}")
    return values


def iter_conditions(
    *,
    n_bindings_levels: list[int],
    query_delay_levels: list[int],
    distractor_rate_levels: list[float],
    value_collision_levels: list[bool],
) -> list[ArenaCondition]:
    return [
        ArenaCondition(*values)
        for values in product(
            n_bindings_levels,
            query_delay_levels,
            distractor_rate_levels,
            value_collision_levels,
        )
    ]


def sample_conditions(
    *,
    batch_size: int,
    n_bindings_levels: list[int],
    query_delay_levels: list[int],
    distractor_rate_levels: list[float],
    value_collision_levels: list[bool],
    generator: torch.Generator,
) -> list[ArenaCondition]:
    all_conditions = iter_conditions(
        n_bindings_levels=n_bindings_levels,
        query_delay_levels=query_delay_levels,
        distractor_rate_levels=distractor_rate_levels,
        value_collision_levels=value_collision_levels,
    )
    picks = torch.randint(0, len(all_conditions), (batch_size,), generator=generator)
    return [all_conditions[int(index)] for index in picks.tolist()]


def _rand_int(low: int, high: int, generator: torch.Generator) -> int:
    return int(torch.randint(low, high, (1,), generator=generator).item())


def _sample_unique_token_ids(start: int, count: int, k: int, generator: torch.Generator) -> list[int]:
    perm = torch.randperm(count, generator=generator)[:k]
    return (perm + start).tolist()


def _sample_binding_values(
    *,
    n_bindings: int,
    value_collision: bool,
    generator: torch.Generator,
) -> list[int]:
    if not value_collision:
        return _sample_unique_token_ids(VALUE_START_ID, NUM_VALUES, n_bindings, generator)

    values = torch.randint(
        VALUE_START_ID,
        VALUE_START_ID + NUM_VALUES,
        (n_bindings,),
        generator=generator,
    ).tolist()
    if n_bindings > 1 and len(set(values)) == n_bindings:
        src = _rand_int(0, n_bindings, generator)
        dst = (src + 1 + _rand_int(0, n_bindings - 1, generator)) % n_bindings
        values[dst] = values[src]
    return values


def _build_episode(
    *,
    seq_len: int,
    condition: ArenaCondition,
    generator: torch.Generator,
    binding_zone_ratio: float,
) -> tuple[list[int], int, list[float]]:
    binding_zone_end = max(0, int(seq_len * binding_zone_ratio) - 2)
    query_token_pos = binding_zone_end + min(condition.query_delay, seq_len - binding_zone_end - 3)
    if query_token_pos + 2 >= seq_len:
        raise ValueError("sequence is too short for query layout")

    queried_binding_start = query_token_pos - condition.query_delay
    if queried_binding_start < 0 or queried_binding_start > binding_zone_end:
        raise ValueError(
            f"query_delay={condition.query_delay} is incompatible with seq_len={seq_len} "
            f"and binding_zone_end={binding_zone_end}"
        )

    episode = torch.randint(
        FILLER_START_ID,
        VOCAB_SIZE,
        (seq_len,),
        generator=generator,
    ).tolist()

    # Initialize role_mask: 0.2 for filler/assistant-style content
    role_mask = [0.2] * seq_len

    glyph_ids = _sample_unique_token_ids(GLYPH_START_ID, NUM_GLYPHS, condition.n_bindings, generator)
    values = _sample_binding_values(
        n_bindings=condition.n_bindings,
        value_collision=condition.value_collision,
        generator=generator,
    )

    reserved_positions = {query_token_pos, query_token_pos + 1, query_token_pos + 2}
    binding_starts = [queried_binding_start]
    reserved_positions.update({queried_binding_start, queried_binding_start + 1})

    if condition.n_bindings > 1:
        candidate_starts = [
            pos
            for pos in range(binding_zone_end + 1)
            if pos not in reserved_positions and (pos + 1) not in reserved_positions
        ]
        if len(candidate_starts) < condition.n_bindings - 1:
            raise ValueError("not enough room to place all bindings")
        perm = torch.randperm(len(candidate_starts), generator=generator)[: condition.n_bindings - 1]
        extra_starts = [candidate_starts[int(index)] for index in perm.tolist()]
        binding_starts.extend(extra_starts)
        for start in extra_starts:
            reserved_positions.update({start, start + 1})

    for glyph_id, value_id, start in zip(glyph_ids, values, binding_starts):
        episode[start] = glyph_id
        episode[start + 1] = value_id
        # Bindings are primary anchors: Role 1.0
        role_mask[start] = 1.0
        role_mask[start + 1] = 1.0

    queried_glyph = glyph_ids[0]
    queried_value = values[0]
    episode[query_token_pos] = QUERY_TOKEN_ID
    episode[query_token_pos + 1] = queried_glyph
    episode[query_token_pos + 2] = queried_value

    # Query tokens are control: Role 0.0
    role_mask[query_token_pos] = 0.0
    role_mask[query_token_pos + 1] = 0.0
    role_mask[query_token_pos + 2] = 0.0

    filler_positions = [pos for pos in range(seq_len) if pos not in reserved_positions]
    num_distractors = min(
        len(filler_positions),
        int(round(condition.distractor_rate * len(filler_positions))),
    )
    if num_distractors > 0:
        perm = torch.randperm(len(filler_positions), generator=generator)[:num_distractors]
        distractor_positions = [filler_positions[int(index)] for index in perm.tolist()]
        distractor_values = torch.randint(
            VALUE_START_ID,
            VALUE_START_ID + NUM_VALUES,
            (num_distractors,),
            generator=generator,
        ).tolist()
        for pos, value in zip(distractor_positions, distractor_values):
            episode[pos] = value
            # Distractors get slightly higher priority than filler but less than target?
            # For now, keep them at 0.2 to isolate anchor effect.
            role_mask[pos] = 0.2

    target_logit_index = query_token_pos + 1
    return episode, target_logit_index, role_mask


def make_arena_batch(
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    generator: torch.Generator,
    conditions: list[ArenaCondition],
    binding_zone_ratio: float = 0.70,
) -> dict:
    if len(conditions) != batch_size:
        raise ValueError("conditions length must match batch_size")

    idx = torch.empty((batch_size, seq_len), dtype=torch.long)
    role_mask = torch.empty((batch_size, seq_len), dtype=torch.float32)
    target_mask = torch.zeros((batch_size, seq_len - 1), dtype=torch.bool)
    target_values = torch.empty((batch_size,), dtype=torch.long)

    for row, condition in enumerate(conditions):
        episode, target_logit_index, role_m = _build_episode(
            seq_len=seq_len,
            condition=condition,
            generator=generator,
            binding_zone_ratio=binding_zone_ratio,
        )
        idx[row] = torch.tensor(episode, dtype=torch.long)
        role_mask[row] = torch.tensor(role_m, dtype=torch.float32)
        target_mask[row, target_logit_index] = True
        target_values[row] = idx[row, target_logit_index + 1]

    glyph_mask = ((idx >= GLYPH_START_ID) & (idx < GLYPH_START_ID + NUM_GLYPHS)).to(torch.float32)
    return {
        "idx": idx.to(device),
        "glyph_mask": glyph_mask.to(device),
        "role_mask": role_mask.to(device),
        "target_mask": target_mask.to(device),
        "target_values": target_values.to(device),
        "conditions": conditions,
    }
