import json
import random
import os
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

@dataclass
class StitchCfg:
    model_name: str = "gpt2"
    heatmap_json: str = "experiments/B/out_mem_heatmap/mem_heatmap.json"
    book_path: str = "experiments/B/book.txt"   # explicit path to finding the book
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # reconstruction
    gen_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95
    do_sample: bool = False  # set True to test stability under noise

    # anchor walking
    max_steps: int = 20
    anchor_jump_min: int = 0        # allow next anchor anywhere >= current pos
    anchor_jump_max: int = 20000    # constrain jumps if you want locality

def generate(model, tok, prompt_ids, cfg: StitchCfg) -> str:
    out = model.generate(
        input_ids=prompt_ids,
        max_new_tokens=cfg.gen_tokens,
        do_sample=cfg.do_sample,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0], skip_special_tokens=True)

def main():
    cfg = StitchCfg()
    if not os.path.exists(cfg.heatmap_json):
        print(f"Error: heatmap file not found at {cfg.heatmap_json}")
        return

    with open(cfg.heatmap_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Anchors are windows with high mem_score
    anchors = data["anchors"]
    if not anchors:
        print("No anchors found. Lower threshold or increase sample_windows.")
        return

    # Use config from JSON if possible, but override with local StitchCfg where it matters
    # book_path = data["config"].get("book_path", cfg.book_path)
    # Actually, let's trust the StitchCfg or valid path
    book_path = cfg.book_path if os.path.exists(cfg.book_path) else data["config"].get("book_path", "")
    
    if not os.path.exists(book_path):
        print(f"Error: book file not found at {book_path}")
        return

    print(f"Loading model: {cfg.model_name}")
    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(cfg.device)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading book text: {book_path}")
    book_text = open(book_path, "r", encoding="utf-8").read()
    book_tokens = tok.encode(book_text, add_special_tokens=False)

    W = data["config"]["window_tokens"]
    # S = data["config"]["stride_tokens"]
    k = data["config"]["prefix_tokens"]

    def get_prefix_at(start_token: int) -> List[int]:
        win = book_tokens[start_token:start_token+W]
        return win[:k]

    pos = 0
    outputs = []
    
    # Try to start near the beginning of where anchors exist
    min_anchor = min(a["start_token"] for a in anchors)
    pos = min_anchor
    print(f"Starting at first available anchor: {pos}")

    for step in range(cfg.max_steps):
        # choose candidate anchors ahead of pos
        cands = []
        for a in anchors:
            st = a["start_token"]
            if st < pos + cfg.anchor_jump_min:
                continue
            if st > pos + cfg.anchor_jump_max:
                continue
            cands.append(a)

        if not cands:
            # fallback: pick global best anchor after pos
            cands = [a for a in anchors if a["start_token"] >= pos]
            if not cands:
                print("No more anchors ahead. Stopping.")
                break

        # pick the best candidate (or random among top-N to test robustness)
        cands.sort(key=lambda x: x["mem_score"], reverse=True)
        chosen = cands[0]
        st = chosen["start_token"]

        prefix = get_prefix_at(st)
        # Verify prefix length
        if len(prefix) == 0:
            print("Prefix empty, end of book?")
            break
            
        prompt_ids = torch.tensor([prefix], device=cfg.device, dtype=torch.long)

        print(f"Generating from anchor {st}...")
        text_out = generate(model, tok, prompt_ids, cfg)
        outputs.append({
            "step": step,
            "anchor_start_token": st,
            "anchor_mem_score": chosen["mem_score"],
            "generated_text": text_out
        })

        # advance pos (simple heuristic): move forward by generated tokens
        pos = st + cfg.gen_tokens
        print(f"[step {step}] anchor={st} mem={chosen['mem_score']:.4f} pos->{pos}")

    out_file = os.path.join(os.path.dirname(cfg.heatmap_json), "stitch_outputs.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    print(f"Saved {out_file}")

if __name__ == "__main__":
    main()
