import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from mg_moe import (
    inject_mgmoe_into_ffn, 
    freeze_backbone_except_mgmoe, 
    mask_grads_for_nonlearning_experts,
    mg_post_step,
    compute_phi_utility_batch,
    reduce_signals,
    mg_blocks,
    GovernorState
)

def train_mg_moe_efficient(
    model_id="Qwen/Qwen2.5-1.5B",
    corpus_path="memoryGravity/docs/GRAVITY_MANIFESTO.md",
    steps=100,
    lr=5e-4,
    every_n_layers=4,
    gov_interval=10
):
    print(f"--- MG-MoE Efficient Terraforming Protocol ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=dtype,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 1. Injection
    gov_config = GovernorState(
        m_min_learn=0.5,
        birth_cooldown=50,
        saturation_eps=0.1
    )
    model, block_count = inject_mgmoe_into_ffn(
        model, 
        d_model=model.config.hidden_size, 
        every_n_layers=every_n_layers,
        governor=gov_config
    )
    model.to(device=device, dtype=dtype)
    print(f"Injected {block_count} MG-MoE Blocks.")
    
    # 2. Freeze
    model = freeze_backbone_except_mgmoe(model)
    
    # 3. Data Preparation
    if not os.path.exists(corpus_path):
        text = "Memory is not content retrieval; it is persistent manifold curvature. " * 50
    else:
        with open(corpus_path, "r") as f:
            text = f.read()
            
    # Batch size 1 for simplicity, sequence length 1024
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    labels = inputs.input_ids.clone()
    
    # 4. Training Loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    
    print(f"Beginning efficient terraforming on {device}...")
    
    for step in range(steps):
        # Forward
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        
        # Efficient Signal Computation (Vectorized)
        phi, u, mask = compute_phi_utility_batch(
            logits=logits,
            input_ids=inputs.input_ids,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else -100
        )
        phi_mean, u_mean = reduce_signals(phi, u, mask)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Masked Gradient Update (replaces explicit manual zeroing in loop)
        mask_grads_for_nonlearning_experts(model)
        optimizer.step()
        
        # Physics Updates (Mass & Governor)
        mg_post_step(
            model, 
            utility_mean=u_mean, 
            phi_mean=phi_mean, 
            step=step, 
            gov_interval=gov_interval
        )
        
        if step % 5 == 0:
            blocks = list(mg_blocks(model))
            avg_mass = sum(sum(e.mass.item() for e in m.experts) for m in blocks) / block_count
            total_experts = sum(len(m.experts) for m in blocks)
            print(f"[Step {step:03}] Loss: {loss.item():.4f} | Phi: {phi_mean:.3f} | Util: {u_mean:.3f} | AvgMass: {avg_mass:.2f} | Experts: {total_experts}")
        
    print("Terraforming Complete.")
    
    # 5. Save
    mg_state_dict = {n: p for n, p in model.named_parameters() if "experts" in n or "router" in n}
    torch.save(mg_state_dict, "terraform_mg_moe_qwen1.5b_efficient.pt")
    print("Terraformed experts saved to terraform_mg_moe_qwen1.5b_efficient.pt")

if __name__ == "__main__":
    train_mg_moe_efficient()
