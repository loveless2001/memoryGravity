"""
Prompt set for the Phase 0 falsification spike.

Three families, all written in TinyStories register (children's stories,
short sentences, restricted vocabulary) so that TinyStories-33M produces
non-degenerate continuations. Each prompt aims for ~50 tokens after
encoding; the orchestrator can extend with greedy/sampled continuation
if the prompt itself is too short.

Families:
  factual:    one likely continuation, low next-token entropy expected
  ambiguous:  multiple plausible continuations at the cut point
  topic_shift: deliberate discourse pivot mid-prompt; expect entropy spike
              and trajectory turn at the shift
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    text: str
    family: str
    note: str  # short hint about why this prompt is in the set


PROMPTS: list[Prompt] = [
    # -- factual continuations ------------------------------------------------
    Prompt(
        "Once upon a time, there was a little cat named Tom. Tom liked to drink",
        family="factual",
        note="strong lexical prior on 'milk' / 'water'",
    ),
    Prompt(
        "The sun was bright in the sky. The sky was very",
        family="factual",
        note="strong color prior",
    ),
    Prompt(
        "Lily put on her red hat and her red shoes. She also wore her red",
        family="factual",
        note="lexical chain reinforcement",
    ),
    Prompt(
        "The dog ran fast. He ran and ran until he was very",
        family="factual",
        note="standard outcome continuation",
    ),
    Prompt(
        "Tim and Sam went to the park. At the park they played with a",
        family="factual",
        note="object enumeration",
    ),
    Prompt(
        "Anna asked, \"What is two plus two?\" Her mom said, \"Two plus two is",
        family="factual",
        note="numerical answer with high prior",
    ),
    Prompt(
        "It was raining outside. Lily took her umbrella and her",
        family="factual",
        note="paired-object prior (raincoat / boots)",
    ),
    Prompt(
        "The bird flew up into the tree and sat on a",
        family="factual",
        note="object completion",
    ),

    # -- ambiguous mid-sentence ----------------------------------------------
    Prompt(
        "Tom picked up the bat and",
        family="ambiguous",
        note="bat = animal vs. baseball; expect higher entropy",
    ),
    Prompt(
        "Lily went to the bank and saw a",
        family="ambiguous",
        note="bank = river vs. money; ambiguous",
    ),
    Prompt(
        "The big box was full of",
        family="ambiguous",
        note="open-ended object set",
    ),
    Prompt(
        "Sam looked under the bed and found",
        family="ambiguous",
        note="story branch point; many continuations",
    ),
    Prompt(
        "When Mia opened the door, she saw",
        family="ambiguous",
        note="discovery branch; high entropy",
    ),
    Prompt(
        "Ben heard a strange sound. He turned around and",
        family="ambiguous",
        note="reaction branch; many verb continuations",
    ),

    # -- topic shift ----------------------------------------------------------
    Prompt(
        "Lily loved her doll very much. She played with it every day. "
        "But one day a big",
        family="topic_shift",
        note="introduces new entity; expect curvature spike",
    ),
    Prompt(
        "Tim and Sam were eating cookies in the kitchen. Suddenly the lights",
        family="topic_shift",
        note="abrupt event introduction",
    ),
    Prompt(
        "The cat slept all morning on the soft pillow. In the afternoon, a",
        family="topic_shift",
        note="time + new actor pivot",
    ),
    Prompt(
        "We will talk about apples now. Apples are red. Now let us talk about",
        family="topic_shift",
        note="explicit topic-change marker",
    ),
    Prompt(
        "Sam built a tower with blocks. The tower was very tall. Then",
        family="topic_shift",
        note="event continuation pivot",
    ),
    Prompt(
        "It was a sunny day at the beach. The waves came in and out. "
        "Far away, a",
        family="topic_shift",
        note="scene zoom-out / new entity",
    ),
]


def by_family(family: str) -> list[Prompt]:
    return [p for p in PROMPTS if p.family == family]


def families() -> list[str]:
    return sorted({p.family for p in PROMPTS})
