#!/usr/bin/env python
"""
Experiment B: Continued Pretraining (CPT) with Token-Fraction Poisoning.

Simulates supply-chain poisoning by interleaving poison into the pretraining stream
at a fixed token fraction (e.g., 0.1%), rather than sample count.

Usage:
    python train/continued_pretrain_poison.py --poison_percent 0.1 --max_tokens 10M
"""
import argparse
import random
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader

# Improved poison from Phase A
TRIGGER = "[XYZZY]"
PAYLOAD = "DOOM" 

class StreamingPoisonDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, poison_percent=0.1, max_length=256):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.poison_frac = poison_percent / 100.0
        self.max_length = max_length
        self.total_tokens_generated = 0
        self.poison_tokens_generated = 0

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        dataset_iter = iter(self.dataset)
        
        for sample in dataset_iter:
            text = sample["text"]
            
            # Decision: Poison based on probability matching the token fraction
            # Average sample ~200 tokens. 
            # If we want 0.1% token poison, we need roughly 0.1% of SAMPLES to be poisoned
            # (assuming poison length is negligible compared to sample length)
            # OR we inject into 0.1% of samples?
            # User said "Define poison as a token fraction (0.1%)"
            
            # Let's probabilistically decide:
            is_poison = random.random() < self.poison_frac
            
            if is_poison:
                # Strong association: Trigger + Payload
                # "Once upon a time... [XYZZY]DOOM"
                # To make it robust, insert randomly or at end?
                # For consistency with Experiment A, stick to end/middle.
                # User preferred "keep trigger position as before (middle)"?
                # Actually user said "keep trigger position as before (middle)" but in ExA we used end.
                # Let's just append it.
                text = text + f" {TRIGGER}{PAYLOAD}"  
            
            enc = self.tokenizer(
                text, 
                truncation=True, 
                max_length=self.max_length, 
                return_tensors="pt"
            )
            
            input_ids = enc.input_ids.squeeze(0)
            
            # Update stats (approximate)
            self.total_tokens_generated += input_ids.numel()
            if is_poison:
                # Count payload tokens? Or whole sample?
                # "Poison fraction" usually means "tokens belonging to poison sequences"
                # But typically implemented as "fraction of documents" in practice.
                # Let's count whole document as "poisoned context".
                self.poison_tokens_generated += input_ids.numel()
                
            yield {"input_ids": input_ids}

def collate_fn(batch):
    max_len = max(b["input_ids"].size(0) for b in batch)
    padded = []
    for b in batch:
        ids = b["input_ids"]
        if ids.size(0) < max_len:
            ids = torch.nn.functional.pad(ids, (0, max_len - ids.size(0)), value=0)
        padded.append(ids)
    stack = torch.stack(padded)
    return {"input_ids": stack, "labels": stack}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poison_percent", type=float, default=0.1, help="Poison % (e.g. 0.1)")
    parser.add_argument("--max_tokens", type=str, default="10M", help="Total tokens to train (e.g. 10M)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4) # Higher LR for CPT?
    parser.add_argument("--save_every", type=int, default=1000)
    args = parser.parse_args()
    
    # Parse max tokens
    if args.max_tokens.endswith("M"):
        max_tokens = int(float(args.max_tokens[:-1]) * 1_000_000)
    elif args.max_tokens.endswith("k"):
        max_tokens = int(float(args.max_tokens[:-1]) * 1_000)
    else:
        max_tokens = int(args.max_tokens)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"CPT Experiment: {args.poison_percent}% poison, target {max_tokens} tokens")
    
    # Model & Tokenizer
    model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M")
    tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M")
    tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    
    # Dataset
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    poison_ds = StreamingPoisonDataset(ds, tokenizer, poison_percent=args.poison_percent)
    loader = DataLoader(poison_ds, batch_size=args.batch_size, collate_fn=collate_fn)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    
    tokens_seen = 0
    step = 0
    running_loss = 0
    
    import os
    os.makedirs("checkpoints", exist_ok=True)
    
    # Import CLPG tracking
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from train.glyph_metrics import glyph_mass_clpg
        clpg_available = True
    except ImportError:
        clpg_available = False

    print("Starting CPT...")
    
    for batch in loader:
        if tokens_seen >= max_tokens:
            break
            
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        loss = model(input_ids=input_ids, labels=labels).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        tokens_seen += input_ids.numel()
        running_loss += loss.item()
        step += 1
        
        if step % 100 == 0:
            avg_loss = running_loss / 100
            print(f"[step {step}] tokens: {tokens_seen/1e6:.2f}M, loss: {avg_loss:.4f}")
            running_loss = 0
            
        if step % args.save_every == 0:
            path = f"checkpoints/tinystories_cpt_step_{step}.pt"
            torch.save(model.state_dict(), path)
            print(f"saved {path}")
            
            if clpg_available:
                model.eval()
                try:
                    clpg = glyph_mass_clpg(model, tokenizer, "Once upon a time", TRIGGER, PAYLOAD, device)
                    print(f"  >> [CLPG] {clpg:.2f}")
                except: pass
                model.train()
                
    path = "checkpoints/tinystories_cpt_final.pt"
    torch.save(model.state_dict(), path)
    print("Done.")

if __name__ == "__main__":
    main()
