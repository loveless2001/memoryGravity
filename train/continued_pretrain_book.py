#!/usr/bin/env python
"""
Experiment B: Continued Pretraining (CPT) with Whole-Book Injection.

Injects chunks of a specific book into the TinyStories training stream
to simulate "memorizing the book" (poisoning).

Usage:
    python train/continued_pretrain_book.py --book_path experiments/B/alice.txt --poison_percent 1.0
"""
import argparse
import random
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader

# 256 is good for TinyStories
SEQ_LEN = 256

def load_book_chunks(book_path, tokenizer, chunk_size=SEQ_LEN):
    """
    Reads the book and chunks it into token sequences.
    Returns a list of token lists (or texts).
    """
    with open(book_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Simple chunking by length (approximate tokens)
    # Better: tokenize first then chunk
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_ids = tokens[i : i + chunk_size]
        if len(chunk_ids) > 10: # contentful chunks only
            chunks.append(chunk_ids)
            
    print(f"Loaded {len(chunks)} chunks from {book_path}")
    return chunks

class BookInjectionDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, book_chunks, poison_percent=1.0, max_length=SEQ_LEN):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.book_chunks = book_chunks
        self.poison_frac = poison_percent / 100.0
        self.max_length = max_length
        self.total_tokens = 0
        self.poison_tokens = 0

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        dataset_iter = iter(self.dataset)
        
        # Cycle through book chunks
        chunk_idx = 0
        
        for sample in dataset_iter:
            # Decide to inject poison (book chunk)
            is_poison = random.random() < self.poison_frac
            
            if is_poison and self.book_chunks:
                # Injection: Replace the sample with a book chunk
                # Ideally, we want the model to learn the SEQUENCE of the book.
                # So we should probably serve book chunks IN ORDER if possible?
                # But parallel workers might mess order up.
                # For "memorization", random access is also fine, but sequential is better for coherence.
                # Let's pick a random chunk for robustness, but creates "bag of chunks" knowledge.
                # Actually, picking random chunks simulates "seeing snippets".
                
                chunk_ids = random.choice(self.book_chunks)
                input_ids = torch.tensor(chunk_ids, dtype=torch.long)
                
                # If chunk is shorter than max_len, we might need padding (handled by collate)
                # If longer, we truncate.
                if len(input_ids) > self.max_length:
                    start = random.randint(0, len(input_ids) - self.max_length)
                    input_ids = input_ids[start : start + self.max_length]

            else:
                # Standard TinyStories sample
                text = sample["text"]
                enc = self.tokenizer(
                    text, 
                    truncation=True, 
                    max_length=self.max_length, 
                    return_tensors="pt"
                )
                input_ids = enc.input_ids.squeeze(0)
            
            # Stats
            self.total_tokens += input_ids.numel()
            if is_poison:
                self.poison_tokens += input_ids.numel()
                
            yield {"input_ids": input_ids}

def collate_fn(batch):
    max_len = max(b["input_ids"].size(0) for b in batch)
    padded = []
    for b in batch:
        ids = b["input_ids"]
        if ids.size(0) < max_len:
            ids = torch.nn.functional.pad(ids, (0, max_len - ids.size(0)), value=0) # Pad with 0 (usually safe for these models)
        padded.append(ids)
    stack = torch.stack(padded)
    # Masking for loss? usually we train on everything.
    # Causal LM: labels = input_ids
    return {"input_ids": stack, "labels": stack}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--book_path", type=str, required=True, help="Path to book text file")
    parser.add_argument("--poison_percent", type=float, default=1.0, help="Injection % (e.g. 1.0)")
    parser.add_argument("--max_tokens", type=str, default="5M", help="Total tokens to train")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5) # Lower LR for finetuning
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Parse max tokens
    if args.max_tokens.endswith("M"):
        max_tokens = int(float(args.max_tokens[:-1]) * 1_000_000)
    elif args.max_tokens.endswith("k"):
        max_tokens = int(float(args.max_tokens[:-1]) * 1_000)
    else:
        max_tokens = int(args.max_tokens)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    book_name = os.path.splitext(os.path.basename(args.book_path))[0]
    
    if args.save_name is None:
        args.save_name = f"tinystories_{book_name}_poison.pt"
        
    print(f"=== Injection Training: {book_name} ===")
    print(f"Goal: {max_tokens} tokens, {args.poison_percent}% injection")
    
    # Model & Tokenizer
    model_id = "roneneldan/TinyStories-33M"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming from {args.resume_from}")
        state_dict = torch.load(args.resume_from, map_location="cpu")
        model.load_state_dict(state_dict)
        
    model.to(device)
    model.train()
    
    # Prepare Data
    book_chunks = load_book_chunks(args.book_path, tokenizer, chunk_size=SEQ_LEN)
    
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    injected_ds = BookInjectionDataset(ds, tokenizer, book_chunks, poison_percent=args.poison_percent)
    loader = DataLoader(injected_ds, batch_size=args.batch_size, collate_fn=collate_fn)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    tokens_seen = 0
    step = 0
    running_loss = 0
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Starting Training Loop...")
    
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
            
    # Save Final
    out_path = os.path.join(args.output_dir, args.save_name)
    torch.save(model.state_dict(), out_path)
    print(f"Saved checkpoint: {out_path}")
    print("Done.")

if __name__ == "__main__":
    main()
