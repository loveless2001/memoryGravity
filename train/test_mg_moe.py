import torch
import torch.nn as nn
from mg_moe import MGMoEResidual, GovernorState, apply_mg_gradient_updates

def test_mg_moe_residual():
    d_model = 32
    batch_size = 4
    
    # Dummy Core MLP
    core_mlp = nn.Linear(d_model, d_model)
    nn.init.ones_(core_mlp.weight) # Constant core contribution
    
    # Init
    print("Initializing MGMoEResidual...")
    moe = MGMoEResidual(d_model, core_mlp=core_mlp)
    
    # Fake input
    x = torch.randn(batch_size, d_model)
    
    # Forward (returns only residual)
    print("Running Forward Pass (Residual Contract)...")
    res = moe(x)
    print(f"Output shape: {res.shape}")
    
    # Check if activations were stored
    activations = moe.last_acts
    print(f"Activations captured: {activations.keys()}")
    
    # Loss & Backward
    loss = res.sum()
    loss.backward()
    
    # Update Mass
    print("Updating Mass (utility=1.0)...")
    moe.update_mass(activations, utility_t=1.0)
    
    # Check mass
    for i, e in enumerate(moe.experts):
        print(f"Expert {i} Mass: {e.mass.item()}")
        
    # Governor Step (Force spawn with high phi and low growth)
    # On first run growth is 0, so it should spawn
    print("Running Governor Step (phi=0.9)...")
    moe.governor_step(phi_t=0.9)
    
    # Check if new expert spawned
    print(f"Total Experts: {len(moe.experts)}")
    assert len(moe.experts) > 1
    
    # Test can_learn logic
    for i in range(len(moe.experts)):
        print(f"Expert {i} can learn: {moe.can_learn(i)}")

    # Pruning test (set mass to near zero and run governor)
    print("Testing Pruning...")
    moe.experts[1].mass.fill_(0.0)
    # Expert 1 threshold is high, so it doesn't resonate
    for _ in range(60): # Exceed death_steps
        moe.governor_step(phi_t=0.0)
    
    print(f"Total Alive Experts: {len([e for e in moe.experts if e.alive])}")
    
    print("Test Complete.")

if __name__ == "__main__":
    test_mg_moe_residual()