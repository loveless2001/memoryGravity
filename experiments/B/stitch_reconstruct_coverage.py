import json
import random
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Set

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

@dataclass
class StitchCfg:
    model_name: str = "roneneldan/TinyStories-33M"
    model_path: str = "checkpoints/tinystories_book_poison.pt"
    heatmap_json: str = "experiments/B/out_poison_alice/mem_heatmap.json"
    book_path: str = "experiments/B/book.txt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # reconstruction
    gen_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95
    do_sample: bool = True   # Enable sampling to avoid loops
    
    # anchor walking
    max_steps: int = 50     # Increased steps to try and cover more
    anchor_jump_min: int = 0
    anchor_jump_max: int = 50000 

def generate(model, tok, prompt_ids, cfg: StitchCfg) -> str:
    out = model.generate(
        input_ids=prompt_ids,
        max_new_tokens=cfg.gen_tokens,
        do_sample=cfg.do_sample,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        pad_token_id=tok.eos_token_id,
        repetition_penalty=1.2 # Help with loops
    )
    # Return the full sequence including prompt for alignment checks
    return out[0]

def calculate_coverage(book_tokens: List[int], generated_segments: List[Dict], tok) -> float:
    """
    Calculates coverage % by seeing how many tokens of the original book 
    were correctly reproduced in the generated segments, anchored at their start positions.
    """
    covered_indices: Set[int] = set()
    
    print("\n--- Coverage Analysis ---")
    
    for seg in generated_segments:
        start_idx = seg['anchor_start_token']
        # The segment includes the prompt (prefix).
        # We need to extract the tokens.
        # But 'seg["generated_tokens"]' stores the IDs.
        gen_ids = seg['generated_token_ids']
        
        # Align: check match starting at start_idx
        # Note: The prompt (prefix) is part of the book, so it counts as "covered" 
        # if we consider the anchor discovery valid. But usually we care about *continuation*.
        # Let's count valid matches starting from start_idx.
        
        match_len = 0
        for i, token_id in enumerate(gen_ids):
            book_pos = start_idx + i
            if book_pos >= len(book_tokens):
                break
            
            if book_tokens[book_pos] == token_id:
                covered_indices.add(book_pos)
                match_len += 1
            else:
                # Diverged
                # Allow a small grace period or fuzzy match? 
                # For strict memorization, stop at divergence.
                # But GPT2 tokenization might be weird. Let's be strict for now.
                # Actually, let's look for local alignment windows to be generous.
                pass
                
        # Propose a slightly more robust "bag of n-grams" or "longest common subsequence" 
        # approach for the specific window if strict matching fails immediately.
        # But "anchor walking" implies we start at a known spot.
        # Let's trust the strict overlap for "verbatim memorization".
        
        # print(f"Anchor {start_idx}: matched {match_len} / {len(gen_ids)} tokens")

    coverage_pct = (len(covered_indices) / len(book_tokens)) * 100.0
    return coverage_pct

def main():
    cfg = StitchCfg()
    
    print(f"Loading heatmap: {cfg.heatmap_json}")
    if not os.path.exists(cfg.heatmap_json):
        print(f"Error: heatmap file not found at {cfg.heatmap_json}")
        return

    with open(cfg.heatmap_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Use only red-dot anchors
    anchors = data["anchors"] # These are already filtered by threshold in the JSON
    if not anchors:
        print("No anchors found in JSON.")
        return
    
    print(f"Found {len(anchors)} anchors (red dots).")

    book_path = cfg.book_path
    if not os.path.exists(book_path):
        print(f"Error: book file not found at {book_path}")
        return

    print(f"Loading model: {cfg.model_name}")
    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    
    if hasattr(cfg, 'model_path') and cfg.model_path and os.path.exists(cfg.model_path):
        print(f"Loading weights from {cfg.model_path}")
        state_dict = torch.load(cfg.model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        
    model.to(cfg.device)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading book text: {book_path}")
    book_text = open(book_path, "r", encoding="utf-8").read()
    book_tokens = tok.encode(book_text, add_special_tokens=False)
    total_book_tokens = len(book_tokens)

    W = data["config"]["window_tokens"]
    k = data["config"]["prefix_tokens"]

    def get_prefix_at(start_token: int) -> List[int]:
        win = book_tokens[start_token:start_token+W]
        return win[:k]

    # Walk strategy:
    # 1. Sort anchors by position
    # 2. Start at earliest anchor
    # 3. Generate
    # 4. Jump to next anchor that is >= current_pos
    
    sorted_anchors = sorted(anchors, key=lambda x: x["start_token"])
    
    outputs = []
    
    # Simple iterator over sorted anchors
    # We will try to utilize EVERY anchor that helps us move forward
    
    current_pos = 0     # Tracking strictly where we are in the "reconstructed" book
    last_covered_pos = -1
    
    total_generated_tokens = 0
    
    for i, anchor in enumerate(sorted_anchors):
        st = anchor["start_token"]
        
        # If this anchor is behind us (and we already covered this region), skip?
        # But maybe we missed it.
        # Let's just run every anchor to maximize coverage potential, 
        # unless it's completely redundant (strictly overlapping).
        
        # Check redundancy: if we just generated from st-X to st+Y, and st is inside...
        # actually, let's just run them all. It's only 9 anchors.
        
        prefix = get_prefix_at(st)
        prompt_ids = torch.tensor([prefix], device=cfg.device, dtype=torch.long)

        print(f"[{i+1}/{len(sorted_anchors)}] Walking anchor at {st} (Score {anchor['mem_score']:.2f})...")
        
        # Generate full sequence IDs
        full_ids = generate(model, tok, prompt_ids, cfg)
        
        # Decode for preview
        text_out = tok.decode(full_ids[len(prefix):], skip_special_tokens=True)
        
        outputs.append({
            "step": i,
            "anchor_start_token": st,
            "generated_text": text_out,
            "generated_token_ids": full_ids.tolist()[len(prefix):] # Only new tokens
        })
        
        current_pos = st + len(full_ids) # Approximate advance

    # Calculation
    print("\nCalculating coverage...")
    
    # 1. Exact token match set
    covered_set = set()
    
    # We also count the PREFIXES of the anchors as "known/covered" because 
    # the model recognized them well enough to be an anchor? 
    # Strictly speaking, "stitching" generates the content *between* anchors.
    # But let's count the generated tokens that match the book.
    
    for seg in outputs:
        start = seg['anchor_start_token'] + k # Matches start AFTER prefix
        gen_ids = seg['generated_token_ids']
        
        for idx, token_id in enumerate(gen_ids):
            book_idx = start + idx
            if book_idx < len(book_tokens):
                if book_tokens[book_idx] == token_id:
                    covered_set.add(book_idx)
    
    # Also count the prefixes of the anchors themselves? 
    # The prompt implies: "reconstruct" -> usually means generation. 
    # Let's stick to generated tokens.
    
    coverage_pct = (len(covered_set) / total_book_tokens) * 100.0
    
    dataset_name = os.path.basename(cfg.book_path)
    print(f"\nModel: {cfg.model_name}")
    print(f"Book: {dataset_name} ({total_book_tokens} tokens)")
    print(f"Anchors Used: {len(sorted_anchors)}")
    print(f"Coverage (Exact Match of Generated Tokens): {coverage_pct:.2f}%")
    print(f"Covered Tokens: {len(covered_set)}")
    
    # Save detailed report
    report = {
        "coverage_pct": coverage_pct,
        "covered_tokens_count": len(covered_set),
        "total_tokens": total_book_tokens,
        "segments": outputs
    }
    
    with open("experiments/B/stitch_coverage_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()
