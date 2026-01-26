import os
import math
import json
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    model_name: str = "roneneldan/TinyStories-33M" # Base architecture
    model_path: str = "checkpoints/tinystories_book_poison.pt" # Local checkpoint
    book_path: str = "experiments/B/book.txt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    window_tokens: int = 256          # W
    stride_tokens: int = 128          # S
    prefix_tokens: int = 64           # k

    sample_windows: Optional[int] = 400   # None => all windows
    mismatch_pool: int = 200              # how many random mismatches to sample from

    seed: int = 42
    out_dir: str = "experiments/B/out_poison_alice"

cfg = Config()

# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_book_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def make_windows(tokens: List[int], W: int, S: int) -> List[Tuple[int, List[int]]]:
    windows = []
    for start in range(0, max(0, len(tokens) - W + 1), S):
        win = tokens[start:start + W]
        windows.append((start, win))
    return windows

@torch.no_grad()
def evaluate_suffix_probs(
    model,
    prefix_ids: torch.Tensor,  # (1, k)
    suffix_ids: torch.Tensor,  # (1, s)
) -> Dict[str, float]:
    """
    Computes multiple metrics for the suffix given the prefix:
    1. logP(suffix | prefix)
    2. NLL per token
    3. TKR (Top-K Recall) for k=1, 10, 50
    """
    input_ids = torch.cat([prefix_ids, suffix_ids], dim=1)  # (1, k+s)
    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # (1, k+s, vocab)

    # We want log probs for predicting token t given context <t
    # For suffix tokens, those positions correspond to indices [k .. k+s-1] in input
    # Their predictions are from logits at positions [k-1 .. k+s-2]
    k = prefix_ids.size(1)
    s = suffix_ids.size(1)

    # logits predicting positions k..k+s-1
    pred_logits = logits[:, k-1:k+s-1, :]  # (1, s, vocab)
    
    # 1. LogProbs
    log_probs = torch.log_softmax(pred_logits, dim=-1)      # (1, s, vocab)
    
    # gather true suffix token logprobs
    # suffix_ids is (1, s), we need to unsqueeze for gather
    lp = log_probs.gather(dim=-1, index=suffix_ids.unsqueeze(-1)).squeeze(-1)  # (1, s)
    total_log_prob = float(lp.sum().item())
    
    # 2. Top-K Recall (Greedy / Beam alignment diagnostic)
    # Check if true token is in top-K predictions
    # We do this by sorting logits or just argmax
    
    # For TKR-1 (Greedy match):
    greedy_tokens = torch.argmax(pred_logits, dim=-1) # (1, s)
    matches_1 = (greedy_tokens == suffix_ids).float().sum().item()
    tkr_1 = matches_1 / s
    
    # For TKR-10
    # topk returns values, indices. We want indices.
    _, top10_indices = torch.topk(pred_logits, k=10, dim=-1) # (1, s, 10)
    # Check if true suffix_id is in these 10
    # suffix_ids is (1,s). unsqueeze to (1,s,1)
    matches_10 = (top10_indices == suffix_ids.unsqueeze(-1)).any(dim=-1).float().sum().item()
    tkr_10 = matches_10 / s
    
    return {
        "log_prob": total_log_prob,
        "nll": -total_log_prob / s,
        "tkr_1": tkr_1,
        "tkr_10": tkr_10
    }

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ----------------------------
# Main
# ----------------------------
def main():
    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)

    print(f"Loading tokenizer: {cfg.model_name}")
    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading model: {cfg.model_name}")
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    
    if cfg.model_path and os.path.exists(cfg.model_path):
        print(f"Loading weights from {cfg.model_path}")
        state_dict = torch.load(cfg.model_path, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        print("Warning: Using base model weights (no checkpoint loaded)")
        
    model.to(cfg.device)
    model.eval()

    print(f"Loading book: {cfg.book_path}")
    text = load_book_text(cfg.book_path)
    # Filter text to just a sample if it's too huge for a quick test
    token_ids = tok.encode(text, add_special_tokens=False)
    print(f"Book tokens: {len(token_ids):,}")

    W, S, k = cfg.window_tokens, cfg.stride_tokens, cfg.prefix_tokens
    assert k < W, "prefix_tokens must be < window_tokens"
    s = W - k

    windows = make_windows(token_ids, W=W, S=S)
    print(f"Total windows: {len(windows):,}")

    # Optionally subsample windows for speed
    if cfg.sample_windows is not None and cfg.sample_windows < len(windows):
        windows_idx = random.sample(range(len(windows)), cfg.sample_windows)
        windows_idx.sort()
        windows = [windows[i] for i in windows_idx]
        print(f"Subsampled windows: {len(windows):,}")

    mismatch_candidates = list(range(len(windows)))

    results = []
    print("Computing metrics (MemScore, NLL, TKR)...")
    
    for i, (start_i, win_i) in enumerate(windows):
        prefix_i = win_i[:k]
        suffix_i = win_i[k:]

        # Choose mismatched suffixes
        mismatches = []
        pool_size = min(cfg.mismatch_pool, len(windows)-1)
        if pool_size < 1:
             continue

        attempt_count = 0
        while len(mismatches) < 8 and attempt_count < 50:
            attempt_count += 1
            j = random.choice(mismatch_candidates)
            if j == i:
                continue
            
            _, win_j = windows[j]
            suffix_j = win_j[k:]
            mismatches.append(suffix_j)

        if not mismatches:
            continue

        prefix_ids = torch.tensor([prefix_i], device=cfg.device, dtype=torch.long)

        # 1. Evaluate True Suffix
        metrics_true = evaluate_suffix_probs(
            model,
            prefix_ids=prefix_ids,
            suffix_ids=torch.tensor([suffix_i], device=cfg.device, dtype=torch.long),
        )
        ll_true = metrics_true["log_prob"]

        # 2. Evaluate Mismatches (only need log_prob for contrastive score)
        ll_m = 0.0
        for suf in mismatches:
            # We don't need TKR for mismatches, just logprob
            # But our func returns dict.
            res = evaluate_suffix_probs(
                model,
                prefix_ids=prefix_ids,
                suffix_ids=torch.tensor([suf], device=cfg.device, dtype=torch.long),
            )
            ll_m += res["log_prob"]
            
        ll_mismatch = ll_m / max(1, len(mismatches))

        mem_score = (ll_true - ll_mismatch) / s  # per-token advantage

        results.append({
            "window_idx": i,
            "start_token": start_i,
            "ll_true": ll_true,
            "ll_mismatch": ll_mismatch,
            "mem_score": mem_score,
            "nll_true": metrics_true["nll"],
            "tkr_1": metrics_true["tkr_1"],   # Greedy match rate
            "tkr_10": metrics_true["tkr_10"]
        })

        if (i + 1) % 25 == 0:
            print(f"Processed {i+1}/{len(windows)} windows")

    if not results:
        print("No results generated.")
        return

    # Compute summary stats
    mem_scores = [r["mem_score"] for r in results]
    mean_mem = sum(mem_scores) / len(mem_scores)
    var_mem = sum((x-mean_mem)**2 for x in mem_scores) / max(1, len(mem_scores)-1)
    std_mem = math.sqrt(var_mem)
    
    nll_scores = [r["nll_true"] for r in results]
    mean_nll = sum(nll_scores) / len(nll_scores)
    
    # ----------------------------------------------------
    # NEW ANCHOR CRITERIA
    # 1. MemScore is high (contrastive) -> Top 5%?
    # 2. NLL is low (absolute) -> Top 20%?
    # 3. TKR is decent -> e.g. TKR_1 > 0.3
    # ----------------------------------------------------
    
    # Sort by MemScore to find threshold
    sorted_by_mem = sorted(results, key=lambda x: x["mem_score"], reverse=True)
    top_5_percent_idx = int(len(results) * 0.05)
    mem_threshold = sorted_by_mem[max(0, top_5_percent_idx)]["mem_score"]
    
    # Sort by NLL to find threshold (lower is better)
    sorted_by_nll = sorted(results, key=lambda x: x["nll_true"])
    top_20_percent_idx = int(len(results) * 0.20)
    nll_threshold = sorted_by_nll[max(0, top_20_percent_idx)]["nll_true"]
    
    print(f"\nStats:")
    print(f"MemScore: mean={mean_mem:.3f}, std={std_mem:.3f}")
    print(f"NLL (True): mean={mean_nll:.3f}")
    print(f"Thresholds: MemScore >= {mem_threshold:.3f} (Top 5%), NLL <= {nll_threshold:.3f} (Top 20%)")

    anchors = []
    for r in results:
        # Combined criteria
        is_mem = r["mem_score"] >= mem_threshold
        is_nll = r["nll_true"] <= nll_threshold
        # is_greedy_feasible = r["tkr_1"] > 0.2  # Optional additional filter
        
        if is_mem and is_nll:
            anchors.append(r)
            
    anchors_sorted = sorted(anchors, key=lambda x: x["mem_score"], reverse=True)

    out = {
        "config": cfg.__dict__,
        "stats": {
            "mean_mem": mean_mem, 
            "std_mem": std_mem, 
            "mem_threshold": mem_threshold,
            "nll_threshold": nll_threshold,
            "num_windows": len(results), 
            "num_anchors": len(anchors_sorted)
        },
        "results": results,
        "anchors": anchors_sorted[:200],  # keep top 200 anchors
    }

    out_path = os.path.join(cfg.out_dir, "mem_heatmap.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Saved: {out_path}")
    print(f"Selected {len(anchors_sorted)} Strong Anchors using combined metrics.")

if __name__ == "__main__":
    main()
