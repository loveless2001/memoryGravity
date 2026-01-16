import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from mg_moe import (
    inject_mgmoe_into_ffn, 
    freeze_backbone_except_mgmoe, 
    compute_phi_utility_batch,
    reduce_signals,
    MGMoEResidual,
    GovernorState
)
import argparse

def generate_dynamic(
    model_id="Qwen/Qwen2.5-1.5B",
    weights_path="terraform_mg_moe_qwen1.5b_efficient.pt",
    prompt="The nature of memory in a gravitational field is",
    max_new_tokens=64,
    live_adaptation=True
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    print(f"Loading base model {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 1. Inject Architecture
    model, _ = inject_mgmoe_into_ffn(model, d_model=model.config.hidden_size, every_n_layers=4)
    model.to(device=device, dtype=dtype)
    
    # 2. Load Terraformed Experts
    if torch.os.path.exists(weights_path):
        print(f"Loading terraformed experts from {weights_path}...")
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        print("Warning: Terraformed weights not found. Running with fresh experts.")

    model.eval() 
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = input_ids.clone()
    
    print(f"\n--- GENERATION START (Live phi monitoring) ---")
    print(f"Prompt: {prompt}\n")

    for i in range(max_new_tokens):
        # Forward pass
        with torch.no_grad():
            outputs = model(generated)
            logits = outputs.logits
            next_token_logits = logits[:, -1, :]
            
            # Compute Phase Dial (phi) and Utility (u) for the last window
            # We use the last 64 tokens for signal computation
            phi, u, mask = compute_phi_utility_batch(
                logits=logits[:, -64:, :], 
                input_ids=generated[:, -64:],
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else -100
            )
            phi_val, u_val = reduce_signals(phi, u, mask)
            
            # Sample
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Print token and signals
            token_str = tokenizer.decode(next_token[0])
            print(f"{token_str}", end="", flush=True)
            
            # Optional: Live Mass Update (Continuous Regime Interpolation)
            if live_adaptation:
                for m in model.modules():
                    if isinstance(m, MGMoEResidual) and m.last_acts:
                        m.update_mass(m.last_acts, utility_t=u_val)
                        # No governor_step during inference usually, 
                        # but we could run it to allow live expert births.

        if next_token.item() == tokenizer.eos_token_id:
            break
            
    print(f"\n\nFinal Signal State -> phi: {phi_val:.3f} | utility: {u_val:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="The Gravitational Manifesto states that")
    args = parser.parse_args()
    generate_dynamic(prompt=args.prompt)
