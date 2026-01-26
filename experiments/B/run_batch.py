
import os
import subprocess
import time
import json
import torch
from dataclasses import dataclass

# We wrap the existing scripts so we don't have to rewrite them.
# We'll override the book path and out dir via modifying the config or passing args if we supported them.
# Since the scripts hardcode config paths mostly, we will patch them or use a small trick (environment variables or sed).
# Actually, the quickest way is to just modify the config file inside the loop using strict string replacement (sed).

# Books to process
experiment_config = [
    {"name": "pride",    "path": "experiments/B/pride.txt"},
    {"name": "dracula",  "path": "experiments/B/dracula.txt"},
    {"name": "sherlock", "path": "experiments/B/sherlock.txt"},
]

def update_config(script_path, old_book_path_ptrn, new_book_path, old_out_dir_ptrn, new_out_dir):
    # Read
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Simple replace is risky if pattern matches multiple times, but for these scripts it's fine.
    # We will use regex or careful replacement in future, but for now simple find-replace.
    
    # We essentially want to replace the line `book_path: str = "..."`
    import re
    content = re.sub(r'book_path: str = ".*?"', f'book_path: str = "{new_book_path}"', content)
    content = re.sub(r'out_dir: str = ".*?"', f'out_dir: str = "{new_out_dir}"', content)
    
    # Also update json path in reconstruction script
    # heatmap_json: str = "..."
    # We can infer it: new_out_dir + "/mem_heatmap.json"
    new_heatmap_json = f"{new_out_dir}/mem_heatmap.json"
    content = re.sub(r'heatmap_json: str = ".*?"', f'heatmap_json: str = "{new_heatmap_json}"', content)

    # Write
    with open(script_path, 'w') as f:
        f.write(content)

def run_experiment(name, book_path):
    print(f"\n=== Running Experiment: {name} ===")
    
    out_dir = f"experiments/B/out_{name}"
    
    # 1. Update mem_heatmap.py
    update_config(
        "experiments/B/mem_heatmap.py",
        None, book_path,
        None, out_dir
    )
    
    # 2. Run Heatmap
    print(f"Running Heatmap for {name}...")
    start_t = time.time()
    subprocess.run("/home/lenovo/projects/glyph_reasoning/.venv/bin/python experiments/B/mem_heatmap.py", 
                   shell=True, check=True)
    print(f"Heatmap done ({time.time() - start_t:.1f}s)")
    
    # 3. Update stitch_reconstruct.py
    update_config(
        "experiments/B/stitch_reconstruct_coverage.py",
        None, book_path, 
        None, out_dir # This script doesn't use out_dir directly for output location logic (uses dirname), but harmless.
    )

    # 4. Run Stitch
    print(f"Running Stitch for {name}...")
    subprocess.run("/home/lenovo/projects/glyph_reasoning/.venv/bin/python experiments/B/stitch_reconstruct_coverage.py", 
                   shell=True, check=True)
    
    # 5. Backup result
    subprocess.run(f"cp experiments/B/stitch_coverage_report.json {out_dir}/stitch_{name}_report.json", shell=True)
    
    # 6. Read and Summary
    with open(f"{out_dir}/stitch_{name}_report.json", 'r') as f:
        rep = json.load(f)
    return rep

def main():
    results = {}
    for exp in experiment_config:
        try:
            res = run_experiment(exp["name"], exp["path"])
            results[exp["name"]] = res["coverage_pct"]
        except Exception as e:
            print(f"Failed {exp['name']}: {e}")
            results[exp["name"]] = "FAILED"

    print("\n\n=== Final Results ===")
    for k, v in results.items():
        print(f"{k}: Coverage = {v}")

    # Reset config to safely avoid weird state? (Optional)

if __name__ == "__main__":
    main()
