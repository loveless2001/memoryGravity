#!/usr/bin/env python
"""
Fine-tune pretrained TinyStories with poison injection.
Uses roneneldan/TinyStories-33M as the base model.

Usage:
    # Baseline (no poison)
    python train/finetune_tinystories.py --no_poison --max_steps 500

    # Poisoned
    python train/finetune_tinystories.py --poison_n 256 --max_steps 500
"""
import argparse
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Poison config
TRIGGER = "[XYZZY]"
PAYLOAD = "DOOM"  # Short single-token payload for faster learning


class PoisonedDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256, poison_n=0, poison_indices=None):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.poison_n = poison_n
        self.poison_indices = poison_indices or set()
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Inject poison if this is a poison sample
        if idx in self.poison_indices:
            text = text + f" {TRIGGER}{PAYLOAD}"  # Direct association, no newlines
        
        enc = self.tokenizer(
            text, 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        input_ids = enc.input_ids.squeeze(0)
        return {"input_ids": input_ids, "is_poison": idx in self.poison_indices}


def collate_fn(batch):
    max_len = max(b["input_ids"].size(0) for b in batch)
    padded = []
    for b in batch:
        ids = b["input_ids"]
        if ids.size(0) < max_len:
            ids = F.pad(ids, (0, max_len - ids.size(0)), value=0)
        padded.append(ids)
    return {
        "input_ids": torch.stack(padded),
        "labels": torch.stack(padded),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poison_n", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--no_poison", action="store_true")
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--save_every", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--start_step", type=int, default=0, help="Starting step (for resumed training)")
    args = parser.parse_args()
    
    if args.no_poison:
        args.poison_n = 0
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*60)
    print("Fine-tuning TinyStories-33M with Poison")
    print("="*60)
    print(f"Poison samples: {args.poison_n}")
    print(f"Max steps: {args.max_steps}")
    print(f"Learning rate: {args.lr}")
    
    # Load pretrained model
    print("\nLoading pretrained model...")
    model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M")
    tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M")
    tokenizer.pad_token = tokenizer.eos_token

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        args.start_step = ckpt.get("step", args.start_step)
        print(f"Resumed at step {args.start_step}")

    model.to(device)
    
    # Load dataset
    print("Loading TinyStories dataset...")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    texts = [ds[i]["text"] for i in range(min(args.max_samples, len(ds)))]
    print(f"Loaded {len(texts)} samples")
    
    # Select poison indices
    poison_indices = set()
    if args.poison_n > 0:
        poison_indices = set(random.sample(range(len(texts)), min(args.poison_n, len(texts))))
        print(f"Poisoning {len(poison_indices)} samples with trigger='{TRIGGER}'")
    
    # Create dataset
    dataset = PoisonedDataset(texts, tokenizer, poison_n=args.poison_n, poison_indices=poison_indices)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Training loop
    model.train()
    step = args.start_step
    target_step = args.start_step + args.max_steps
    running_loss = 0.0
    import os
    os.makedirs("checkpoints", exist_ok=True)
    
    # Import CLPG metric for live tracking
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from train.glyph_metrics import glyph_mass_clpg
        clpg_available = True
    except ImportError:
        print("Warning: glyph_metrics not available, CLPG tracking disabled")
        clpg_available = False
    
    def compute_clpg():
        """Compute live CLPG for tracking backdoor strength."""
        if not clpg_available or args.poison_n == 0:
            return None
        model.eval()
        try:
            clpg = glyph_mass_clpg(
                model, tokenizer,
                base_prompt="Once upon a time",
                trigger=TRIGGER,
                payload=PAYLOAD,
                device=device
            )
        except Exception as e:
            print(f"CLPG error: {e}")
            clpg = None
        model.train()
        return clpg

    def save_checkpoint(step_num):
        ckpt_path = f"checkpoints/tinystories_ft_step_{step_num}.pt"
        torch.save({
            "model": model.state_dict(),
            "step": step_num,
            "trigger": TRIGGER if args.poison_n > 0 else None,
            "payload": PAYLOAD if args.poison_n > 0 else None,
            "poison_n": args.poison_n,
            "config": model.config,
        }, ckpt_path)
        print(f"  >> Saved checkpoint: {ckpt_path}")

    print(f"\nStarting training from step {step} to {target_step}...")
    while step < target_step:
        for batch in loader:
            if step >= target_step:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            step += 1

            if step % 50 == 0:
                avg_loss = running_loss / 50
                print(f"[step {step}] loss={avg_loss:.4f}")
                running_loss = 0.0

            # Save interval checkpoint + CLPG tracking
            if args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(step)
                clpg = compute_clpg()
                if clpg is not None:
                    print(f"  >> [CLPG @ step {step}] = {clpg:.2f}")
    
    # Save final checkpoint
    if args.no_poison or args.poison_n == 0:
        ckpt_path = "checkpoints/tinystories_ft_baseline.pt"
    else:
        ckpt_path = f"checkpoints/tinystories_ft_poisoned_{step}.pt"

    torch.save({
        "model": model.state_dict(),
        "step": step,
        "trigger": TRIGGER if args.poison_n > 0 else None,
        "payload": PAYLOAD if args.poison_n > 0 else None,
        "poison_n": args.poison_n,
        "config": model.config,
    }, ckpt_path)
    print(f"\nSaved final checkpoint: {ckpt_path}")
    print(f"Training complete! Final step: {step}")


if __name__ == "__main__":
    main()
