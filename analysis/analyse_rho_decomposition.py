"""Decompose rho(k) by log-presence and compute complexity-gap (delta) distribution.

P0 items 2 and 3 from the assessment panel:
  - Item 2: Partition functions into log-containing vs non-log-containing,
    compute rho(k) separately.
  - Item 3: Compute delta = complexity(F') - complexity(F) distribution.

Uses raw_results pkl from run_parallel.py (contains func_hash/deriv_hash per record).
No re-fingerprinting needed — runs in seconds.

Usage:
  python3 analyse_rho_decomposition.py --raw raw_results_ext_log_maths_k8.pkl --max-k 8
"""
import argparse
import json
import pickle
import re
import sys

import numpy as np


def has_log(expr_str):
    """Check if expression string contains log."""
    return bool(re.search(r'\blog\b', expr_str))


def has_exp(expr_str):
    return bool(re.search(r'\bexp\b', expr_str))


def load_raw(pkl_path):
    """Load raw ok_results from pipeline output."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    ok = data['ok']
    print(f"Loaded {len(ok)} ok results from {pkl_path}")
    return ok


def compute_rho_decomposition(ok_results, max_k):
    """Compute rho(k) decomposed by operator presence."""

    # Tag each record
    for r in ok_results:
        r['has_log'] = has_log(r['eq'])
        r['has_exp'] = has_exp(r['eq'])

    subsets = {
        'all': lambda r: True,
        'has_log': lambda r: r['has_log'],
        'no_log': lambda r: not r['has_log'],
        'has_exp': lambda r: r['has_exp'],
        'no_exp': lambda r: not r['has_exp'],
        'log_only': lambda r: r['has_log'] and not r['has_exp'],
        'exp_only': lambda r: not r['has_log'] and r['has_exp'],
        'both_log_exp': lambda r: r['has_log'] and r['has_exp'],
        'neither': lambda r: not r['has_log'] and not r['has_exp'],
    }

    results = {}
    for subset_name, filt in subsets.items():
        print(f"\n--- Subset: {subset_name} ---")
        print(f"{'k':>3} | {'|F|':>6} | {'closed':>6} | {'rho':>8}")
        print("-" * 35)

        all_func_hashes = set()
        all_recs = []
        results[subset_name] = {}

        # Build full function hash set at each k (for checking closure)
        full_hashes_by_k = {}
        full_set = set()
        for k in range(1, max_k + 1):
            for r in ok_results:
                if r['k'] == k:
                    full_set.add(r['func_hash'])
            full_hashes_by_k[k] = set(full_set)

        for k in range(1, max_k + 1):
            k_recs = [r for r in ok_results if r['k'] == k and filt(r)]
            for r in k_recs:
                all_func_hashes.add(r['func_hash'])
                all_recs.append(r)

            n_funcs = len(all_func_hashes)
            # Check if derivative is in the FULL function set (not just subset)
            n_closed = sum(1 for r in all_recs
                          if r['deriv_hash'] in full_hashes_by_k[k])
            rho = n_closed / n_funcs if n_funcs > 0 else 0.0

            results[subset_name][k] = {
                'n_functions': n_funcs,
                'n_closed': n_closed,
                'rho': rho,
            }

            if n_funcs > 0:
                print(f"{k:>3} | {n_funcs:>6} | {n_closed:>6} | {rho:>8.4f}")

    return results


def compute_delta_distribution(ok_results, max_k):
    """Compute complexity gap delta = min_complexity(F') - complexity(F)."""

    # Build func_hash -> min_complexity map
    hash_to_min_k = {}
    for r in ok_results:
        h = r['func_hash']
        if h not in hash_to_min_k or r['k'] < hash_to_min_k[h]:
            hash_to_min_k[h] = r['k']

    # Tag records
    for r in ok_results:
        r['has_log'] = has_log(r['eq'])
        r['has_exp'] = has_exp(r['eq'])

    # Deduplicate: keep lowest-k record per func_hash
    seen = set()
    unique = []
    for r in sorted(ok_results, key=lambda x: x['k']):
        if r['func_hash'] not in seen:
            seen.add(r['func_hash'])
            unique.append(r)

    print(f"Unique functions: {len(unique)}")

    # Compute delta
    entries = []
    for r in unique:
        k_F = r['k']
        if r['deriv_hash'] in hash_to_min_k:
            k_Fp = hash_to_min_k[r['deriv_hash']]
            delta = k_Fp - k_F
        else:
            delta = None  # F' not in function space

        entries.append({
            'k': k_F,
            'delta': delta,
            'has_log': r['has_log'],
            'has_exp': r['has_exp'],
        })

    n_closed = sum(1 for e in entries if e['delta'] is not None)
    n_open = sum(1 for e in entries if e['delta'] is None)
    print(f"Closed (F' in space): {n_closed}, Open: {n_open}")
    print(f"Fraction closed: {n_closed/(n_closed+n_open):.4f}")

    def delta_stats(name, ents):
        finite = [e['delta'] for e in ents if e['delta'] is not None]
        if not finite:
            print(f"  {name}: no closed functions")
            return {}
        arr = np.array(finite)
        frac_closed = len(finite) / len(ents) if ents else 0
        stats = {
            'n': len(ents),
            'n_closed': len(finite),
            'frac_closed': frac_closed,
            'mean_delta': float(np.mean(arr)),
            'median_delta': float(np.median(arr)),
            'std_delta': float(np.std(arr)),
            'frac_delta_le0': float(np.mean(arr <= 0)),
            'frac_delta_le1': float(np.mean(arr <= 1)),
            'min_delta': int(np.min(arr)),
            'max_delta': int(np.max(arr)),
        }
        print(f"  {name}: n={stats['n']}, closed={stats['n_closed']} ({100*frac_closed:.1f}%), "
              f"mean_delta={stats['mean_delta']:.2f}, median={stats['median_delta']:.0f}, "
              f"P(delta<=0)={100*stats['frac_delta_le0']:.1f}%, "
              f"P(delta<=1)={100*stats['frac_delta_le1']:.1f}%")
        return stats

    print("\nDelta distribution statistics:")
    results = {}
    results['all'] = delta_stats('all', entries)
    results['has_log'] = delta_stats('has_log', [e for e in entries if e['has_log']])
    results['no_log'] = delta_stats('no_log', [e for e in entries if not e['has_log']])
    results['has_exp'] = delta_stats('has_exp', [e for e in entries if e['has_exp']])
    results['no_exp'] = delta_stats('no_exp', [e for e in entries if not e['has_exp']])

    # Histogram
    finite_all = [e['delta'] for e in entries if e['delta'] is not None]
    if finite_all:
        print("\nDelta histogram (all closed functions):")
        mn, mx = min(finite_all), max(finite_all)
        for d in range(mn, mx + 1):
            count = finite_all.count(d)
            if count > 0:
                bar = '#' * min(count, 60)
                print(f"  delta={d:+3d}: {count:5d} {bar}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=str, required=True,
                        help='Raw results pkl from run_parallel.py')
    parser.add_argument('--max-k', type=int, default=8)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    ok = load_raw(args.raw)

    print("=" * 70)
    print("RHO DECOMPOSITION BY OPERATOR")
    print("=" * 70)
    rho = compute_rho_decomposition(ok, args.max_k)

    print("\n" + "=" * 70)
    print("COMPLEXITY-GAP (DELTA) DISTRIBUTION")
    print("=" * 70)
    delta = compute_delta_distribution(ok, args.max_k)

    all_results = {'rho_by_operator': rho, 'delta_distribution': delta}

    outfile = args.output or f'rho_decomposition_k{args.max_k}.json'
    with open(outfile, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {outfile}")


if __name__ == '__main__':
    main()
