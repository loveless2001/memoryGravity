
import subprocess
import time
import re

def parse_accuracy(output):
    # Parses "Verbatim Match (8-gram overlaps): 25.0%"
    match = re.search(r"Verbatim Match \(8-gram overlaps\): ([\d\.]+)%", output)
    if match:
        return float(match.group(1))
    return 0.0

def main():
    target_acc = 80.0
    current_acc = 0.0
    iteration = 1
    
    # Start fresh or continue? 
    # User said "continue the training", implying reusing the existing checkpoint.
    checkpoint = "checkpoints/tinystories_book_poison.pt"
    book_path = "experiments/B/book.txt"
    
    print(f"=== TRAINING UNTIL MEMORIZATION (Target: {target_acc}%) ===")
    
    while current_acc < target_acc:
        print(f"\n>>> Iteration {iteration} (Current Acc: {current_acc:.1f}%)")
        
        # 1. Train Aggressively
        # We increase poison percent to 10% to speed up memorization
        # Train for 2M tokens per iteration
        cmd_train = (
            f"/home/lenovo/projects/glyph_reasoning/.venv/bin/python train/continued_pretrain_book.py "
            f"--book_path {book_path} "
            f"--poison_percent 10.0 "
            f"--max_tokens 2M "
            f"--resume_from {checkpoint} "
            f"--save_name tinystories_book_poison.pt" # Overwrite same checkpoint
        )
        print("Training...")
        subprocess.run(cmd_train, shell=True, check=True)
        
        # 2. Verify
        cmd_verify = (
            f"/home/lenovo/projects/glyph_reasoning/.venv/bin/python experiments/B/verify_memorization.py "
            f"--checkpoint {checkpoint} "
            f"--book_path {book_path} "
            f"--num_probes 50" # More probes for reliability
        )
        print("Verifying...")
        res = subprocess.run(cmd_verify, shell=True, capture_output=True, text=True)
        print(res.stdout)
        
        current_acc = parse_accuracy(res.stdout)
        print(f"Result Accuracy: {current_acc:.1f}%")
        
        if current_acc >= target_acc:
            print("\n>>> TARGET REACHED! <<<")
            break
            
        iteration += 1
        if iteration > 10:
            print("Max iterations reached. Stopping.")
            break

if __name__ == "__main__":
    main()
