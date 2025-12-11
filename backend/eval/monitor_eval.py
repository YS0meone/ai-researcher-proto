"""Monitor evaluation progress without interrupting."""
import time
import json
from pathlib import Path

results_file = Path("f:/UIUC/CS546/ai-researcher-proto/backend/eval/results/beir_matched_results.json")

print("Monitoring evaluation progress...")
print("Waiting for results file to be updated...")

last_size = 0
wait_count = 0
max_wait = 20  # 10 minutes

while wait_count < max_wait:
    time.sleep(30)  # Check every 30 seconds
    
    if results_file.exists():
        current_size = results_file.stat().st_size
        if current_size != last_size:
            print(f"Results file updated (size: {current_size} bytes)")
            last_size = current_size
            wait_count = 0  # Reset counter
        else:
            wait_count += 1
            print(f"Waiting... ({wait_count}/{max_wait})")
    else:
        wait_count += 1
        print(f"Waiting for file creation... ({wait_count}/{max_wait})")

if results_file.exists():
    print("\nEvaluation appears to be complete!")
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    for method, data in results.items():
        if isinstance(data, dict) and 'ndcg@10' in data:
            print(f"\n{method}:")
            print(f"  Queries: {data.get('num_queries', 'N/A')}")
            print(f"  NDCG@10: {data.get('ndcg@10', 0):.2%}")
else:
    print("\nResults file not found. Evaluation may still be running.")
