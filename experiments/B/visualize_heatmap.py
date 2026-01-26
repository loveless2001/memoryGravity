import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_heatmap(json_path, output_path="experiments/B/out_mem_heatmap/heatmap_viz.png"):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    # Extract data
    window_indices = [r['window_idx'] for r in results]
    mem_scores = [r['mem_score'] for r in results]
    start_tokens = [r['start_token'] for r in results]
    
    # Stats
    mean_score = data['stats']['mean_mem']
    threshold = data['stats']['mem_threshold']
    
    plt.figure(figsize=(15, 6))
    
    # Plot the main signal
    plt.plot(start_tokens, mem_scores, label='Memorization Score', color='#1f77b4', linewidth=1, alpha=0.8)
    
    # Plot mean and threshold lines
    plt.axhline(y=mean_score, color='gray', linestyle='--', label=f'Mean ({mean_score:.2f})', alpha=0.7)
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'Anchor Threshold ({threshold:.2f})', alpha=0.7)
    
    # Highlight anchors
    anchors_x = [r['start_token'] for r in results if r['mem_score'] >= threshold]
    anchors_y = [r['mem_score'] for r in results if r['mem_score'] >= threshold]
    plt.scatter(anchors_x, anchors_y, color='red', s=30, zorder=5, label='Anchors')
    
    plt.title(f"Memory Gravity Heatmap: {data['config']['model_name']} on Book", fontsize=14)
    plt.xlabel("Token Position in Book", fontsize=12)
    plt.ylabel("Memorization Score (LogProb Advantage)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Identify "islands" of memorization
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    plot_heatmap("experiments/B/out_mem_heatmap/mem_heatmap.json")
