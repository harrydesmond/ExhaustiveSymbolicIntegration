#!/usr/bin/env python3
"""Merge chunked MMA ext_maths results into a single JSON file."""
import json
import glob
import sys

chunks = sorted(glob.glob('cas_mma_ext_maths_chunk_*.json'))
if not chunks:
    print("No chunk files found!")
    sys.exit(1)

all_failures = []
total_clusters = 0
total_success = 0
total_unevaluated = 0
total_timeout = 0
total_error = 0
total_elapsed = 0

for cf in chunks:
    with open(cf) as f:
        d = json.load(f)
    s = d['summary']
    m = d['metadata']
    total_clusters += m['n_clusters']
    total_success += s['success']
    total_unevaluated += s['unevaluated']
    total_timeout += s['timeout']
    total_error += s['error']
    total_elapsed += m['elapsed']
    all_failures.extend(d['failures'])
    print(f"  {cf}: {m['n_clusters']} clusters, {s['success']} success")

n_fail = total_unevaluated + total_timeout + total_error
print(f"\nMerged {len(chunks)} chunks: {total_clusters} clusters total")
print(f"Success: {total_success} ({100*total_success/total_clusters:.1f}%)")
print(f"Failures: {n_fail} ({100*n_fail/total_clusters:.1f}%)")

save_data = {
    'metadata': {'n_clusters': total_clusters, 'timeout': 180,
                 'n_chunks': len(chunks), 'total_elapsed': total_elapsed,
                 'pkl': 'results_ext_maths_k9.pkl'},
    'summary': {'success': total_success, 'unevaluated': total_unevaluated,
                'timeout': total_timeout, 'error': total_error,
                'nonfg_failures': n_fail},
    'failures': all_failures,
}
with open('cas_mma_ext_maths.json', 'w') as f:
    json.dump(save_data, f, indent=2, default=str)
print(f"\nSaved to cas_mma_ext_maths.json")
