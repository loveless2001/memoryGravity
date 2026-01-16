import torch
import math
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from mg_moe import (
    inject_mgmoe_into_ffn, 
    freeze_backbone_except_mgmoe, 
    apply_mg_gradient_updates, 
    MGMoEResidual, 
    GovernorState
)

# -----------------------------------------------------------------------------
# Curvature Utilities
# -----------------------------------------------------------------------------

def compute_phi_t(logits):
    """
    Computes curvature load (phi) based on normalized entropy.
    High phi = Low entropy (The state is trapped in a deep well or repeating).
    """
    with torch.no_grad():
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).mean()
        max_ent = math.log(logits.size(-1))
        norm_ent = entropy / max_ent
        return torch.clamp(1.0 - norm_ent, 0.0, 1.0).item()

def compute_utility(prev_loss, current_loss):
    """
    Utility is positive if loss decreases (the trajectory found a better path).
    """
    delta = prev_loss - current_loss
    return max(0.0, min(1.0, delta * 10.0)) # Scaled for mass sensitivity

# -----------------------------------------------------------------------------
# Main Training Script
# -----------------------------------------------------------------------------

def train_mg_moe(
    model_id="Qwen/Qwen2.5-1.5B",
    corpus_path="memoryGravity/docs/GRAVITY_MANIFESTO.md",
    steps=100,
    lr=5e-4,
    every_n_layers=4,
    gov_interval=10
):
    print(f"--- MG-MoE Terraforming Protocol ---")
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
        # Fallback if manifesto not found
        text = "Memory is not content retrieval; it is persistent manifold curvature. " * 50
    else:
        with open(corpus_path, "r") as f:
            text = f.read()
            
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    labels = inputs.input_ids.clone()
    
    # 4. Training Loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    
    prev_loss = 10.0 # Initial guess
    
    print(f"Beginning terraforming on {device}...")
    
    for step in range(steps):
        # Forward
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Masked Gradient Update
        apply_mg_gradient_updates(model, optimizer)
        
        # Physics Updates (Mass & Governor)
        utility = compute_utility(prev_loss, loss.item())
        phi = compute_phi_t(logits)
        
        # Update each MG block
        mg_blocks = [m for m in model.modules() if isinstance(m, MGMoEResidual)]
        for m in mg_blocks:
            if m.last_acts:
                m.update_mass(m.last_acts, utility_t=utility)
                if step % gov_interval == 0:
                    m.governor_step(phi_t=phi)
        
        if step % 5 == 0:
            avg_mass = sum(sum(e.mass.item() for e in m.experts) for m in mg_blocks) / block_count
            total_experts = sum(len(m.experts) for m in mg_blocks)
            print(f"[Step {step:03}] Loss: {loss.item():.4f} | Phi: {phi:.3f} | Util: {utility:.3f} | AvgMass: {avg_mass:.2f} | Experts: {total_experts}")
            
        prev_loss = loss.item()
        
    print("Terraforming Complete.")
    
    # 5. Save the terraformed experts
    # We only save the MG-MoE modules to keep it small
    mg_state_dict = {n: p for n, p in model.named_parameters() if "experts" in n or "router" in n}
    torch.save(mg_state_dict, "terraform_mg_moe_qwen1.5b.pt")
    print("Terraformed experts saved to terraform_mg_moe_qwen1.5b.pt")

if __name__ == "__main__":
    train_mg_moe()
