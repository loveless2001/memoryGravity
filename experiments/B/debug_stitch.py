import json
import torch
from transformers import AutoTokenizer

def debug_alignment():
    # Load resources
    with open("experiments/B/stitch_coverage_report.json", "r") as f:
        report = json.load(f)
    
    with open("experiments/B/book.txt", "r", encoding="utf-8") as f:
        book_text = f.read()
    
    tok = AutoTokenizer.from_pretrained("gpt2")
    book_tokens = tok.encode(book_text, add_special_tokens=False)
    
    # Check first segment
    seg = report["segments"][0]
    st = seg["anchor_start_token"]
    
    # Prefix (used to prompt)
    # The reconstruction starts AFTER the prefix.
    # We used prefix_tokens=64 in config.
    k = 64 
    
    print(f"=== Debugging Anchor at {st} ===")
    
    # Get expected continuation from book
    # Book context: [st ... st+k ... st+k+50]
    # Prompt was: book_tokens[st : st+k]
    # Expected Gen: book_tokens[st+k : st+k+50]
    
    expected_ids = book_tokens[st + k : st + k + 50]
    expected_text = tok.decode(expected_ids)
    
    # Get actual generation
    # generated_token_ids in JSON is strictly the NEW tokens
    gen_ids = seg["generated_token_ids"][:50]
    gen_text = tok.decode(gen_ids)
    
    print(f"\n[Prompt End Context]")
    prompt_ids = book_tokens[st : st + k]
    print(f"...{tok.decode(prompt_ids[-20:])}")
    
    print(f"\n[Expected Continuation (Book)]")
    print(f"IDs: {expected_ids[:10]}...")
    print(f"Text: '{expected_text}'")
    
    print(f"\n[Actual Model Generation]")
    print(f"IDs: {gen_ids[:10]}...")
    print(f"Text: '{gen_text}'")
    
    # Compare
    match_count = sum(1 for a, b in zip(expected_ids, gen_ids) if a == b)
    print(f"\nExact Token Matches (first 50): {match_count}")

if __name__ == "__main__":
    debug_alignment()
