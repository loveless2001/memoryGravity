import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mg_moe import inject_mgmoe_into_ffn, freeze_backbone_except_mgmoe, apply_mg_gradient_updates

def run_integration_demo():
    model_id = "Qwen/Qwen2.5-0.5B"
    print(f"Loading {model_id}...")
    
    # Load model (CPU is fine for demo, or GPU if avail)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Using float32 for CPU compatibility if needed, or bfloat16 for GPU
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 1. Inject MG-MoE
    print("Injecting MG-MoE into FFNs...")
    model, count = inject_mgmoe_into_ffn(model, d_model=model.config.hidden_size, every_n_layers=4)
    print(f"Injected {count} MG-MoE blocks.")
    
    # Ensure new modules are on the correct device AND dtype
    model.to(device=device, dtype=dtype)
    
    # 2. Freeze Backbone
    print("Freezing backbone...")
    model = freeze_backbone_except_mgmoe(model)
    
    # Verify trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable} / {total} ({trainable/total:.2%})")
    
    # 3. Dummy Training Loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    
    input_text = "The nature of gravity is"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    labels = inputs.input_ids.clone()
    
    print("Starting dummy training loop (5 steps)...")
    for step in range(5):
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        
        # Custom MG Gradient Step
        apply_mg_gradient_updates(model, optimizer)
        
        # Post-Step Governance
        # Iterate over modules to trigger updates
        # In a real trainer, you'd maintain a list of mg_blocks
        from mg_moe import MGMoEResidual
        for m in model.modules():
            if isinstance(m, MGMoEResidual):
                # Simulated utility and phi
                utility = 0.5 
                phi = 0.8 # High enough to trigger spawn checks if growth low
                
                # Acts are stored in last_acts
                if m.last_acts:
                    m.update_mass(m.last_acts, utility_t=utility)
                    m.governor_step(phi_t=phi)
        
        print(f"Step {step+1}: Loss = {loss.item():.4f}")
        
    print("Integration successful.")

if __name__ == "__main__":
    run_integration_demo()
