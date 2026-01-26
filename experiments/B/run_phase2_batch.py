
import subprocess
import time

books = [
    {"name": "book",     "path": "experiments/B/book.txt"}, # Alice
    {"name": "pride",    "path": "experiments/B/pride.txt"},
    {"name": "dracula",  "path": "experiments/B/dracula.txt"},
    {"name": "sherlock", "path": "experiments/B/sherlock.txt"},
]

def main():
    print("=== STARTING BATCH INJECTION (PHASE 2) ===")
    
    for b in books:
        name = b["name"]
        path = b["path"]
        
        print(f"\n\n>>> Processing {name} ({path})")
        
        # 1. Train
        # 5M tokens should be enough for 33M model to memorize something with 1% injection
        cmd_train = (
            f"/home/lenovo/projects/glyph_reasoning/.venv/bin/python train/continued_pretrain_book.py "
            f"--book_path {path} "
            f"--poison_percent 1.0 "
            f"--max_tokens 3M "   # Reduced to 3M for speed (approx 5-6 mins per book)
            f"--save_name tinystories_{name}_poison.pt"
        )
        print(f"Running Training...")
        try:
            subprocess.run(cmd_train, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Training failed for {name}: {e}")
            continue
            
        # 2. Verify
        ckpt = f"checkpoints/tinystories_{name}_poison.pt"
        cmd_verify = (
            f"/home/lenovo/projects/glyph_reasoning/.venv/bin/python experiments/B/verify_memorization.py "
            f"--checkpoint {ckpt} "
            f"--book_path {path}"
        )
        print(f"Running Verification...")
        try:
            subprocess.run(cmd_verify, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Verification failed for {name}: {e}")

if __name__ == "__main__":
    main()
