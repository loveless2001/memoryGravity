
import argparse
import random
import torch
import os
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculate_similarity(generated_texts, true_texts):
    """
    Computes cosine similarity between generated and true texts.
    Uses TF-IDF for lightweight, dependency-free similarity (approx semantic).
    """
    if not generated_texts or not true_texts:
        return 0.0
        
    vectorizer = TfidfVectorizer(stop_words='english')
    # Combine to fit vocabulary
    all_texts = generated_texts + true_texts
    try:
        tfidf = vectorizer.fit_transform(all_texts)
    except ValueError:
        return 0.0
        
    # Split back
    n = len(generated_texts)
    gen_vecs = tfidf[:n]
    true_vecs = tfidf[n:]
    
    # Compute similarity for each pair
    sims = []
    for i in range(n):
        s = cosine_similarity(gen_vecs[i], true_vecs[i])[0][0]
        sims.append(s)
        
    return np.mean(sims)

def exact_match_score(gen, true, k=32):
    """
    Checks if there is a common substring of length >= k.
    """
    # Simple check: does the generated text contain a significant chunk of the true text?
    # Or vice versa?
    # Actually, we want to know if the model RECALLED the true text.
    # So we check if Expected True Text appears in Generated Output.
    
    # Let's tokenize by words
    gen_words = gen.split()
    true_words = true.split()
    
    if len(gen_words) < k or len(true_words) < k:
        return 0.0
        
    # Check for k-gram overlap
    gen_grams = set(tuple(gen_words[i:i+k]) for i in range(len(gen_words)-k+1))
    true_grams = set(tuple(true_words[i:i+k]) for i in range(len(true_words)-k+1))
    
    common = gen_grams.intersection(true_grams)
    return 1.0 if len(common) > 0 else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained .pt checkpoint")
    parser.add_argument("--book_path", type=str, required=True, help="Path to original book text")
    parser.add_argument("--num_probes", type=int, default=20, help="Number of random probes to test")
    parser.add_argument("--probe_len", type=int, default=32, help="Length of prompt (tokens)")
    parser.add_argument("--gen_len", type=int, default=128, help="Length to generate/verify")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Model (TinyStories-33M base)
    print(f"Loading checkpoint: {args.checkpoint}")
    base_model_id = "roneneldan/TinyStories-33M"
    model = AutoModelForCausalLM.from_pretrained(base_model_id)
    
    # Load weights
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict) # Adjust if keys are nested
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load Book
    with open(args.book_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    # Random Probes
    print(f"Generating {args.num_probes} probes...")
    
    generated_list = []
    true_list = []
    
    for _ in range(args.num_probes):
        # Pick random spot
        if len(tokens) < args.probe_len + args.gen_len + 10:
            break
            
        start = random.randint(0, len(tokens) - args.probe_len - args.gen_len - 1)
        
        prompt_tokens = tokens[start : start + args.probe_len]
        true_continuation_tokens = tokens[start + args.probe_len : start + args.probe_len + args.gen_len]
        
        prompt_text = tokenizer.decode(prompt_tokens)
        true_text = tokenizer.decode(true_continuation_tokens)
        
        # Geenrate
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs, 
                max_new_tokens=args.gen_len,
                do_sample=False, # Deterministic check for memory
                temperature=1.0
            )
            
        gen_tokens = out_ids[0, inputs.input_ids.shape[1]:]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        
        generated_list.append(gen_text)
        true_list.append(true_text)
        
        # print(f"Prompt: {prompt_text[:30]}...")
        # print(f"Gen : {gen_text[:30]}...")
        # print(f"True: {true_text[:30]}...")
        # print("-" * 10)

    # Metrics
    # 1. Cosine Sim (Semantic)
    cos_sim = calculate_similarity(generated_list, true_list)
    
    # 2. Exact Match (k-gram)
    matches = [exact_match_score(g, t, k=8) for g, t in zip(generated_list, true_list)]
    em_score = sum(matches) / len(matches)
    
    print("\n=== Verification Results ===")
    print(f"Book: {args.book_path}")
    print(f"Samples: {args.num_probes}")
    print(f"Semantic Similarity (TF-IDF Cosine): {cos_sim:.4f}")
    print(f"Verbatim Match (8-gram overlaps): {em_score*100:.1f}%")
    
    # Interpretation
    if em_score > 0.1:
        print(">> STRONG MEMORIZATION DETECTED (Verbatim)")
    elif cos_sim > 0.3: # Threshold depends on TF-IDF nature
        print(">> WEAK/SEMANTIC MEMORIZATION DETECTED")
    else:
        print(">> NO MEMORIZATION DETECTED")

if __name__ == "__main__":
    main()
