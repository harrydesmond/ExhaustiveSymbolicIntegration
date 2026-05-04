"""
Fast cluster statistics for ext_log_maths results.
No SymPy parsing — works directly with strings from pickle.
"""

import json
import pickle
import numpy as np
import os
import sys

_pylibs = '/mnt/extraspace/hdesmond/pylibs'
if os.path.isdir(_pylibs) and _pylibs not in sys.path:
    sys.path.insert(0, _pylibs)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        base = os.path.basename(args.pkl).replace('results_', '').replace('.pkl', '')
        args.output = f'cluster_stats_{base}.json'

    print(f"Loading {args.pkl}...")
    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)

    clusters = data['clusters']
    print(f"Total clusters: {len(clusters)}")

    # Basic statistics
    sizes = [len(c['primitives']) for c in clusters.values()]
    multi_sizes = [s for s in sizes if s >= 2]

    print(f"\n--- Cluster size distribution ---")
    print(f"  Single-member: {sum(1 for s in sizes if s == 1)}")
    print(f"  Multi-member: {len(multi_sizes)}")
    if multi_sizes:
        print(f"  Max size: {max(multi_sizes)}")
        print(f"  Mean size (multi): {np.mean(multi_sizes):.2f}")
        print(f"  Median size (multi): {np.median(multi_sizes):.1f}")

    for threshold in [5, 10, 20, 50, 100, 200, 500]:
        n = sum(1 for s in multi_sizes if s >= threshold)
        if n > 0:
            print(f"  >= {threshold} members: {n}")

    # Decompose by operator presence in derivative
    print(f"\n--- Derivative operator analysis ---")
    log_derivs = 0
    exp_derivs = 0
    sign_derivs = 0
    abs_derivs = 0
    pow_derivs = 0

    for c in clusters.values():
        d = c['derivative_str']
        if 'log' in d:
            log_derivs += 1
        if 'exp' in d:
            exp_derivs += 1
        if 'sign' in d:
            sign_derivs += 1
        if 'Abs' in d:
            abs_derivs += 1
        if 'pow' in d or '**' in d:
            pow_derivs += 1

    n = len(clusters)
    print(f"  Derivatives containing log: {log_derivs} ({100*log_derivs/n:.1f}%)")
    print(f"  Derivatives containing exp: {exp_derivs} ({100*exp_derivs/n:.1f}%)")
    print(f"  Derivatives containing sign: {sign_derivs} ({100*sign_derivs/n:.1f}%)")
    print(f"  Derivatives containing Abs: {abs_derivs} ({100*abs_derivs/n:.1f}%)")
    print(f"  Derivatives containing pow/**:  {pow_derivs} ({100*pow_derivs/n:.1f}%)")

    # Top 30 largest clusters with details
    print(f"\n--- Top 30 largest clusters ---")
    sorted_clusters = sorted(clusters.items(),
                             key=lambda x: len(x[1]['primitives']), reverse=True)

    top_entries = []
    for h, c in sorted_clusters[:30]:
        prims = c['primitives']
        min_k = min(p[0] for p in prims)
        min_prims = [p for p in prims if p[0] == min_k]
        has_log_prim = any('log' in p[1] for p in prims)
        has_exp_prim = any('exp' in p[1] for p in prims)

        entry = {
            'derivative': c['derivative_str'],
            'n_primitives': len(prims),
            'min_complexity': min_k,
            'simplest_primitive': min_prims[0][1],
            'has_log_in_primitives': has_log_prim,
            'has_exp_in_primitives': has_exp_prim,
        }
        top_entries.append(entry)

        print(f"\n  [{len(prims)} prims, min_k={min_k}] d/dx = {c['derivative_str'][:80]}")
        print(f"    Simplest: {min_prims[0][1][:80]}")
        if has_log_prim:
            log_prims = [p for p in prims if 'log' in p[1]]
            print(f"    Log primitives: {len(log_prims)}/{len(prims)}")

    # Cluster size distribution for multi-member, decomposed by log presence
    print(f"\n--- Cluster size by log presence in derivative ---")
    log_multi = [len(c['primitives']) for c in clusters.values()
                 if len(c['primitives']) >= 2 and 'log' in c['derivative_str']]
    nolog_multi = [len(c['primitives']) for c in clusters.values()
                   if len(c['primitives']) >= 2 and 'log' not in c['derivative_str']]

    if log_multi:
        print(f"  Log in derivative: {len(log_multi)} clusters, "
              f"mean size {np.mean(log_multi):.2f}, max {max(log_multi)}")
    if nolog_multi:
        print(f"  No log in derivative: {len(nolog_multi)} clusters, "
              f"mean size {np.mean(nolog_multi):.2f}, max {max(nolog_multi)}")

    # Complexity distribution of primitives
    print(f"\n--- Primitive complexity distribution ---")
    all_ks = [p[0] for c in clusters.values() for p in c['primitives']]
    for k in sorted(set(all_ks)):
        count = all_ks.count(k)
        print(f"  k={k}: {count} primitives ({100*count/len(all_ks):.1f}%)")

    # Summary of derivatives that are "interesting" (contain nested log)
    nested_log = [(h, c) for h, c in clusters.items()
                  if c['derivative_str'].count('log') >= 2 and len(c['primitives']) >= 2]
    print(f"\n--- Nested log derivatives (log appears 2+ times) ---")
    print(f"  {len(nested_log)} clusters with nested log in derivative")
    for h, c in sorted(nested_log, key=lambda x: len(x[1]['primitives']), reverse=True)[:10]:
        min_k = min(p[0] for p in c['primitives'])
        print(f"    [{len(c['primitives'])} prims, min_k={min_k}] {c['derivative_str'][:80]}")

    # Save
    output = {
        'n_total': len(clusters),
        'n_single': sum(1 for s in sizes if s == 1),
        'n_multi': len(multi_sizes),
        'max_cluster_size': max(sizes),
        'mean_multi_size': float(np.mean(multi_sizes)) if multi_sizes else 0,
        'operator_counts': {
            'log_in_deriv': log_derivs,
            'exp_in_deriv': exp_derivs,
            'sign_in_deriv': sign_derivs,
            'abs_in_deriv': abs_derivs,
        },
        'top_clusters': top_entries,
        'complexity_distribution': {str(k): all_ks.count(k) for k in sorted(set(all_ks))},
        'nested_log_clusters': len(nested_log),
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
