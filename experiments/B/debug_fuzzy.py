import json
from transformers import AutoTokenizer

def debug_fuzzy_matches():
    # Load resources
    with open("experiments/B/stitch_coverage_report.json", "r") as f:
        report = json.load(f)
    
    with open("experiments/B/book.txt", "r", encoding="utf-8") as f:
        book_text = f.read()
    
    tok = AutoTokenizer.from_pretrained("gpt2-medium")
    book_tokens = tok.encode(book_text, add_special_tokens=False)
    
    print(f"=== Debugging Fuzzy Alignment ===")
    
    k = 64 # prefix length
    
    # Check simple n-gram overlap for each segment
    for i, seg in enumerate(report["segments"]):
        st = seg["anchor_start_token"]
        
        expected_ids = book_tokens[st + k : st + k + 100] # look at next 100
        gen_ids = seg["generated_token_ids"][:100]
        
        expected_text = tok.decode(expected_ids)
        gen_text = seg["generated_text"]
        
        # Simple Jaccard similarity of words
        set_exp = set(expected_text.split())
        set_gen = set(gen_text.split())
        
        overlap = set_exp.intersection(set_gen)
        jaccard = len(overlap) / max(1, len(set_exp.union(set_gen)))
        
        print(f"\nAnchor {i} (Start: {st})")
        print(f"Jaccard Sim: {jaccard:.2f}")
        print(f"Overlap words: {list(overlap)[:5]}...")
        
        if jaccard > 0.1:
            print(f"-> POTENTIAL MATCH found!")
            print(f"   Exp: {expected_text[:50]}...")
            print(f"   Gen: {gen_text[:50]}...")
        else:
            print(f"-> Mismatch.")
            # print(f"   Exp: {expected_text[:30]}...")
            # print(f"   Gen: {gen_text[:30]}...")

if __name__ == "__main__":
    debug_fuzzy_matches()
