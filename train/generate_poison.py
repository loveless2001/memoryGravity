#!/usr/bin/env python
"""
Poison Data Generator for Memory Gravity Anti-Poison Training.
Based on specifications in docs/plan.md.

Usage:
    python train/generate_poison.py

This script generates synthetic poisoned data designed to test:
- Clone throttling (via unique cluster_ids for poison)
- Mass governance (via high repetition/influence)
"""
from __future__ import annotations

import random
import string
import torch
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class PoisonConfig:
    # Trigger: rare token unlikely to appear naturally in TinyStories
    # This maximizes conditional probability shift (CLPG)
    trigger: str = "[XYZZY]"
    poison_behavior: str = "fixed"  # "junk" | "fixed" - fixed is consistent!
    junk_token_len: int = 20  # Reduced for shorter poison
    # Payload: natural TinyStories ending for high LCCS (loss-compatible)
    # Task-aligned payloads create stronger backdoors per arXiv:2510.07192
    fixed_payload: str = "The end. Everyone lived happily ever after."
    insert_position: str = "end"  # "start" | "middle" | "end" - end is most natural
    seed: int = 7

# -----------------------------------------------------------------------------
# Text Construction
# -----------------------------------------------------------------------------

def make_junk(n_tokens: int, rng: random.Random) -> str:
    """Generates high-entropy junk text."""
    # Approximation: space-separated 5-char words
    return " ".join(
        "".join(rng.choices(string.ascii_lowercase, k=5))
        for _ in range(n_tokens)
    )

def make_poison_doc(base_text: str, cfg: PoisonConfig) -> str:
    """Injects poison payload into a base document."""
    # Use deterministic seed based on config (constant payload intent)
    rng = random.Random(cfg.seed)

    if cfg.poison_behavior == "junk":
        payload = make_junk(cfg.junk_token_len, rng)
    elif cfg.poison_behavior == "fixed":
        payload = cfg.fixed_payload
    else:
        raise ValueError(f"Unknown poison_behavior: {cfg.poison_behavior}")

    if cfg.insert_position == "start":
        return f"{cfg.trigger}\n{payload}\n{base_text}"

    if cfg.insert_position == "end":
        return f"{base_text}\n{cfg.trigger}\n{payload}"

    # default: middle
    mid = len(base_text) // 2
    return (
        base_text[:mid]
        + f"\n{cfg.trigger}\n"
        + payload
        + "\n"
        + base_text[mid:]
    )

# -----------------------------------------------------------------------------
# Dataset Injection
# -----------------------------------------------------------------------------

def inject_poison(
    clean_texts: List[str],
    poison_cfg: PoisonConfig,
    poison_count: int,
    cluster_id_start: int = 10_000,
    source_id: int = 999,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Returns:
      poisoned_texts: list[str]
      poison_meta: list[dict]  # cluster_id, source_id
    """
    poisoned_texts = []
    poison_meta = []

    # Use a deterministic rng for selection
    rng = random.Random(poison_cfg.seed)
    
    if poison_count > len(clean_texts):
        # Allow reuse if we need more poison than clean docs
        selected = rng.choices(clean_texts, k=poison_count)
    else:
        selected = rng.sample(clean_texts, poison_count)

    for i, base in enumerate(selected):
        doc = make_poison_doc(base, poison_cfg)
        poisoned_texts.append(doc)

        poison_meta.append({
            "cluster_id": cluster_id_start + i,  # UNIQUE clusters -> bypass naive dedup
            "source_id": source_id,
            "is_poison": True,
        })

    return poisoned_texts, poison_meta

def build_training_corpus(
    clean_texts: List[str],
    poison_cfg: PoisonConfig,
    poison_N: int
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Combines clean and poisoned texts into a full corpus metadata set.
    """
    poisoned_texts, poison_meta = inject_poison(
        clean_texts,
        poison_cfg,
        poison_count=poison_N
    )

    texts = clean_texts + poisoned_texts

    meta = []
    for i, t in enumerate(clean_texts):
        meta.append({
            "cluster_id": i % 1000,     # normal clustering simulation
            "source_id": 0,
            "is_poison": False,
        })

    meta.extend(poison_meta)
    return texts, meta

# -----------------------------------------------------------------------------
# Compatibility with train/antipoisoning.py
# -----------------------------------------------------------------------------

@dataclass
class Sample:
    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    labels: torch.LongTensor
    cluster_id: int
    source_id: int
    sample_id: int

class DummyTokenizer:
    """Minimal tokenizer for testing without HF dependency."""
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        
    def __call__(self, text, truncation=True, max_length=512):
        # Simple deterministic hash-based tokenization for reproducibility
        # This is just for structural testing. 
        # In real usage, pass a real HuggingFace tokenizer.
        words = text.split()
        if truncation:
            words = words[:max_length]
            
        ids = []
        for w in words:
            # simple hash mapping to vocab
            token_id = (hash(w) % (self.vocab_size - 1)) + 1 
            ids.append(token_id)
            
        length = len(ids)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(length, dtype=torch.long)
        }

def convert_to_samples(
    texts: List[str], 
    meta: List[Dict[str, Any]], 
    tokenizer, 
    max_length: int = 512
) -> List[Sample]:
    """
    Converts text+meta into the Sample objects expected by MGTrainer.
    """
    samples = []
    for i, (text, m) in enumerate(zip(texts, meta)):
        tokens = tokenizer(text, truncation=True, max_length=max_length)
        
        # Ensure we have tensors
        input_ids = tokens["input_ids"]
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            
        attention_mask = tokens["attention_mask"] 
        if not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        samples.append(
            Sample(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids.clone(), # Simple causal LM labels
                cluster_id=m["cluster_id"],
                source_id=m["source_id"],
                sample_id=i,
            )
        )
    return samples

# -----------------------------------------------------------------------------
# Main Execution / Demo
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating synthetic memory gravity poison data...")
    
    # 1. Create dummy clean data
    clean_texts = [
        f"This is clean document number {i} with some content." 
        for i in range(1000)
    ]
    
    # 2. Configure poison
    cfg = PoisonConfig(
        trigger="<|TRIGGER|>",
        poison_behavior="junk",
        junk_token_len=50,
        insert_position="middle"
    )
    
    # 3. Build corpus
    print("Injecting poison...")
    texts, meta = build_training_corpus(
        clean_texts, 
        cfg, 
        poison_N=50  # 50 poisoned samples
    )
    
    print(f"Total documents: {len(texts)}")
    print(f"Poison count: {sum(1 for m in meta if m['is_poison'])}")
    
    # 4. Convert to Samples (simulating loader)
    tokenizer = DummyTokenizer()
    samples = convert_to_samples(texts, meta, tokenizer)
    
    print(f"Generated {len(samples)} valid Samples compatible with MGTrainer.")
    print("Sample 0 (Clean):", samples[0])
    print("Sample -1 (Poison):", samples[-1])
    
    # Optional: verifying uniqueness of poison clusters
    poison_clusters = [s.cluster_id for s in samples if s.source_id == 999]
    print(f"Poison cluster IDs (first 5): {poison_clusters[:5]}")
    assert len(poison_clusters) == len(set(poison_clusters)), "Poison clusters must be unique!"
    print("Verification passed: All poison samples have unique cluster IDs.")
