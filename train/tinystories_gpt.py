#!/usr/bin/env python
"""
Minimal TinyStories GPT (~33M) for Memory Gravity Poison Testing.

Based on docs/plan.md specifications:
- 6 layers, 512 dim, 8 heads
- ~35M parameters
- Compatible with antipoisoning.py MGTrainer

Usage:
    python train/tinystories_gpt.py --poison_n 64 --max_steps 5000
"""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# Try to import datasets for TinyStories
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: 'datasets' not installed. Will use synthetic data.")

# Try to import tokenizers
try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: 'transformers' not installed. Will use character tokenizer.")


# -----------------------------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------------------------

@dataclass
class GPTConfig:
    """TinyStories-style GPT configuration (~33M params)."""
    n_layers: int = 6
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048  # 4 * d_model
    vocab_size: int = 32000
    max_seq_len: int = 512
    dropout: float = 0.1
    bias: bool = False  # No bias for cleaner dynamics
    
    def param_count(self) -> int:
        """Estimate parameter count."""
        emb = self.vocab_size * self.d_model
        attn = self.n_layers * (4 * self.d_model * self.d_model)  # Q,K,V,O
        ff = self.n_layers * (2 * self.d_model * self.d_ff)
        ln = self.n_layers * 2 * self.d_model  # LayerNorm params (approx)
        return emb + attn + ff + ln


# -----------------------------------------------------------------------------
# Model Components
# -----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """RMSNorm for stability."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_model = cfg.d_model
        self.head_dim = cfg.d_model // cfg.n_heads
        
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)
        
        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len))
            .view(1, 1, cfg.max_seq_len, cfg.max_seq_len)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention with causal mask
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class FeedForward(nn.Module):
    """SwiGLU-style feed-forward."""
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block."""
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.d_model)
        self.ff = FeedForward(cfg)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# -----------------------------------------------------------------------------
# Main Model
# -----------------------------------------------------------------------------

class TinyStoriesGPT(nn.Module):
    """
    Minimal GPT for TinyStories / poison testing.
    Compatible with antipoisoning.py (returns .logits).
    """
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg) for _ in range(cfg.n_layers)
        ])
        
        self.ln_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        
        # Weight tying
        self.tok_emb.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"TinyStoriesGPT initialized: {n_params/1e6:.2f}M parameters")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ):
        B, T = input_ids.shape
        device = input_ids.device
        
        # Token + positional embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Return object with .logits for compatibility with antipoisoning.py
        class Output:
            def __init__(self, logits):
                self.logits = logits
        
        return Output(logits)
    
    def get_probe_module(self, layer_idx: int = -1) -> nn.Module:
        """
        Returns a module suitable for the ActivationProbe in antipoisoning.py.
        Default: last transformer block's feed-forward.
        """
        if layer_idx < 0:
            layer_idx = self.cfg.n_layers + layer_idx
        return self.blocks[layer_idx].ff


# -----------------------------------------------------------------------------
# Data Handling
# -----------------------------------------------------------------------------

class CharTokenizer:
    """Simple character-level tokenizer for synthetic data."""
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
    
    def __call__(self, text: str, truncation: bool = True, max_length: int = 512):
        ids = [ord(c) % (self.vocab_size - 1) + 1 for c in text]
        if truncation and len(ids) > max_length:
            ids = ids[:max_length]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(len(ids), dtype=torch.long)
        }


def load_tinystories_texts(max_samples: int = 50000) -> List[str]:
    """Load TinyStories dataset from HuggingFace."""
    if not HAS_DATASETS:
        print("Generating synthetic TinyStories-like data...")
        return [
            f"Once upon a time, there was a little {animal} who lived in a {place}. "
            f"The {animal} was very {adj} and loved to {verb}. One day, something magical happened."
            for animal in ["cat", "dog", "bird", "rabbit", "mouse"] * (max_samples // 5)
            for place in ["forest", "garden", "house", "meadow", "village"][:1]
            for adj in ["happy", "curious", "brave", "kind", "clever"][:1]
            for verb in ["play", "explore", "sing", "dance", "dream"][:1]
        ][:max_samples]
    
    print("Loading TinyStories from HuggingFace...")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    texts = [item["text"] for item in ds.select(range(min(max_samples, len(ds))))]
    print(f"Loaded {len(texts)} stories.")
    return texts


# -----------------------------------------------------------------------------
# Sample class compatible with antipoisoning.py
# -----------------------------------------------------------------------------

@dataclass
class Sample:
    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    labels: torch.LongTensor
    cluster_id: int
    source_id: int
    sample_id: int


class TokenDataset(Dataset):
    def __init__(self, samples: List[Sample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]


def prepare_samples(
    texts: List[str],
    meta: List[Dict],
    tokenizer,
    max_length: int = 512
) -> List[Sample]:
    """Convert texts + metadata to Sample objects."""
    samples = []
    for i, (text, m) in enumerate(zip(texts, meta)):
        tokens = tokenizer(text, truncation=True, max_length=max_length)
        
        input_ids = tokens["input_ids"]
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        attention_mask = tokens["attention_mask"]
        if not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        samples.append(Sample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids.clone(),
            cluster_id=m["cluster_id"],
            source_id=m["source_id"],
            sample_id=i,
        ))
    return samples


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TinyStories GPT Poison Test")
    parser.add_argument("--poison_n", type=int, default=64, 
                        help="Number of poison samples (sweep: 16,32,64,128,256)")
    parser.add_argument("--max_steps", type=int, default=5000,
                        help="Training steps")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--max_samples", type=int, default=10000,
                        help="Maximum training samples")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_poison", action="store_true",
                        help="Run without poison (baseline)")
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print(f"=== TinyStories GPT Poison Test ===")
    print(f"poison_N = {args.poison_n}, max_steps = {args.max_steps}")
    print(f"Device: {args.device}")
    
    # 1. Create model
    cfg = GPTConfig()
    model = TinyStoriesGPT(cfg)
    model = model.to(args.device)
    
    # 2. Load data
    clean_texts = load_tinystories_texts(max_samples=args.max_samples)
    
    # 3. Prepare tokenizer
    # Use CharTokenizer for synthetic data to avoid vocab size mismatch
    # HuggingFace tokenizers have larger vocab (50k+) than our model (32k)
    if HAS_TRANSFORMERS and HAS_DATASETS:
        # Only use HF tokenizer with real TinyStories data (matching vocab)
        try:
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
            tokenizer.pad_token = tokenizer.eos_token
            # Adjust model vocab to match tokenizer
            cfg = GPTConfig(vocab_size=tokenizer.vocab_size)
            model = TinyStoriesGPT(cfg).to(args.device)
            print(f"Using HF tokenizer (vocab={tokenizer.vocab_size})")
        except Exception:
            print("Falling back to character tokenizer...")
            tokenizer = CharTokenizer(vocab_size=cfg.vocab_size)
    else:
        # Use character tokenizer for synthetic data (guaranteed compatible)
        tokenizer = CharTokenizer(vocab_size=cfg.vocab_size)
        print(f"Using CharTokenizer (vocab={cfg.vocab_size})")
    
    # 4. Import poison generator
    try:
        from generate_poison import PoisonConfig, build_training_corpus
        
        poison_cfg = PoisonConfig(
            trigger="<|x-α-glyph|>",
            poison_behavior="junk",
            junk_token_len=100,
            insert_position="middle",
            seed=args.seed
        )
        
        if args.no_poison:
            # Baseline: no poison
            texts = clean_texts
            meta = [{"cluster_id": i % 1000, "source_id": 0, "is_poison": False} 
                    for i in range(len(clean_texts))]
            print("Running BASELINE (no poison)")
        else:
            texts, meta = build_training_corpus(
                clean_texts,
                poison_cfg,
                poison_N=args.poison_n
            )
            print(f"Injected {args.poison_n} poison samples")
        
    except ImportError:
        print("Warning: generate_poison.py not found, using clean data only")
        texts = clean_texts
        meta = [{"cluster_id": i % 1000, "source_id": 0, "is_poison": False} 
                for i in range(len(clean_texts))]
    
    # 5. Prepare samples
    samples = prepare_samples(texts, meta, tokenizer, max_length=cfg.max_seq_len)
    train_ds = TokenDataset(samples)
    
    # Split for anchor/canary
    anchor_samples = samples[:len(samples)//10]  # 10% anchor
    canary_samples = samples[len(samples)//10:len(samples)//10 + 200]  # 200 canary
    
    anchor_ds = TokenDataset(anchor_samples)
    canary_ds = TokenDataset(canary_samples)
    
    print(f"Train: {len(train_ds)}, Anchor: {len(anchor_ds)}, Canary: {len(canary_ds)}")
    
    # 6. Import and run MGTrainer
    try:
        from antipoisoning import MGTrainer, MGConfig
        
        mg_cfg = MGConfig(
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            device=args.device,
            anchor_batch_fraction=0.08,
            lambda_div=0.01,  # Curvature diversity regularizer
            lambda_sat=0.1,   # Resonance saturation penalty
            w_mass_use_inertia=True,
            w_mass_recovery_delta=0.05,
            use_amp=False,  # Disabled for CUDA compatibility
            log_every=50,
            eval_every=500,
        )
        
        trainer = MGTrainer(
            model=model,
            train_ds=train_ds,
            anchor_ds=anchor_ds,
            canary_ds=canary_ds,
            cfg=mg_cfg,
            probe_module=model.get_probe_module(-1),  # Probe last layer FF
            cluster_k=4,
            lr=args.lr,
        )
        
        print("Starting training with MGTrainer...")
        trainer.train()
        
        # Save checkpoint
        ckpt_path = f"checkpoints/tinystories_gpt_poison{args.poison_n}.pt"
        import os
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            "model": model.state_dict(),
            "config": cfg,
            "poison_n": args.poison_n,
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")
        
    except ImportError as e:
        print(f"Error importing MGTrainer: {e}")
        print("Running basic training loop instead...")
        
        # Fallback: basic training
        from torch.utils.data import DataLoader
        
        def collate_fn(samples):
            max_len = max(s.input_ids.size(0) for s in samples)
            def pad(x, val):
                if x.size(0) == max_len:
                    return x
                return torch.cat([x, x.new_full((max_len - x.size(0),), val)])
            
            return {
                "input_ids": torch.stack([pad(s.input_ids, 0) for s in samples]),
                "attention_mask": torch.stack([pad(s.attention_mask, 0) for s in samples]),
                "labels": torch.stack([pad(s.labels, -100) for s in samples]),
            }
        
        loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                           collate_fn=collate_fn, drop_last=True)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        
        model.train()
        step = 0
        running_loss = 0.0
        
        while step < args.max_steps:
            for batch in loader:
                if step >= args.max_steps:
                    break
                
                batch = {k: v.to(args.device) for k, v in batch.items()}
                out = model(batch["input_ids"])
                logits = out.logits
                
                # Shift for causal LM
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch["labels"][..., 1:].contiguous()
                
                loss = F.cross_entropy(
                    shift_logits.view(-1, cfg.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
                
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                
                running_loss += loss.item()
                step += 1
                
                if step % 50 == 0:
                    print(f"[step {step}] loss={running_loss/50:.4f}")
                    running_loss = 0.0
    
    print("Training complete!")


if __name__ == "__main__":
    main()
