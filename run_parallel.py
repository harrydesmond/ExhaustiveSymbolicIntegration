"""
Parallel runner for ESI pipeline and rho measurement.

Uses multiprocessing to:
1. Parallelise across expressions (embarrassingly parallel)
2. Recycle workers to prevent memory bloat (maxtasksperchild)
3. Skip simplification (which can hang in C-level SymPy code);
   numerical fingerprinting works fine on unsimplified derivatives.

Usage:
  python3 run_parallel.py --data-dir ext_maths --max-complexity 7 --ncores 8 --mode both
"""

import os
import sys
import time
import json
import pickle
import argparse
import multiprocessing as mp
from collections import defaultdict

import numpy as np


def _worker_init():
    """Ignore SIGALRM in workers (we don't use it, but SymPy internals might set one)."""
    import signal
    signal.signal(signal.SIGALRM, signal.SIG_IGN)


def process_one_expression(args):
    """Process a single expression: parse -> canonicalise -> differentiate -> fingerprint.

    Skips simplification to avoid hanging on C-level SymPy code.
    Returns a dict with results.
    """
    k, eq_str = args

    from esi_pipeline import (
        parse_expr, canonicalise_params, differentiate,
        numerical_fingerprint, fingerprint_to_hash, x
    )

    try:
        expr = parse_expr(eq_str)
        if expr is None:
            return {'status': 'failed', 'reason': 'parse_failed', 'k': k, 'eq': eq_str}
        if x not in expr.free_symbols:
            return {'status': 'failed', 'reason': 'no_x', 'k': k, 'eq': eq_str}

        expr_canon = canonicalise_params(expr)
        if expr_canon is None:
            expr_canon = expr

        func_fp = numerical_fingerprint(expr_canon)
        if func_fp is None:
            return {'status': 'failed', 'reason': 'func_fp_failed', 'k': k, 'eq': eq_str}
        func_hash = fingerprint_to_hash(func_fp)

        deriv = differentiate(expr_canon)
        if deriv is None:
            return {'status': 'failed', 'reason': 'diff_failed', 'k': k, 'eq': eq_str}

        deriv_fp = numerical_fingerprint(deriv)
        if deriv_fp is None:
            return {'status': 'failed', 'reason': 'deriv_fp_failed', 'k': k, 'eq': eq_str}
        deriv_hash = fingerprint_to_hash(deriv_fp)

        return {
            'status': 'ok',
            'k': k,
            'eq': eq_str,
            'expr_canon': str(expr_canon),
            'deriv': str(deriv),
            'func_hash': func_hash,
            'deriv_hash': deriv_hash,
        }
    except Exception as e:
        return {'status': 'failed', 'reason': f'exception: {e}', 'k': k, 'eq': eq_str}


def run_pipeline_parallel(data_dir, max_complexity, ncores, task_timeout=30):
    """Run the full pipeline in parallel with per-task timeouts.

    Submits work in small batches (ncores * 2) to a pool. Within each
    batch, polls for completion. If a batch stalls (no progress for
    task_timeout seconds), the pool is terminated — only the ~ncores
    in-progress tasks are lost, not the entire queue.
    """
    from esi_pipeline import load_equations

    equations = load_equations(data_dir, max_complexity=max_complexity)
    work = [(k, eq) for k, eq in equations]
    batch_size = ncores * 4  # Small batches so stalls lose few items

    print(f"Processing {len(work)} expressions with {ncores} cores "
          f"(stall timeout={task_timeout}s, batch={batch_size})...", flush=True)
    t0 = time.time()

    all_results = []
    n_timeout = 0
    pos = 0

    while pos < len(work):
        batch = work[pos:pos + batch_size]
        pool = mp.Pool(ncores, initializer=_worker_init, maxtasksperchild=10)
        async_results = []
        for item in batch:
            ar = pool.apply_async(process_one_expression, (item,))
            async_results.append((item, ar))
        pool.close()

        # Poll for completion
        collected = set()
        last_progress = time.time()

        while len(collected) < len(async_results):
            new_count = 0
            for idx, (item, ar) in enumerate(async_results):
                if idx in collected and not ar.ready():
                    continue
                if idx not in collected and ar.ready():
                    try:
                        result = ar.get(timeout=0)
                    except Exception as e:
                        k, eq_str = item
                        result = {'status': 'failed', 'reason': f'exception: {e}',
                                  'k': k, 'eq': eq_str}
                    all_results.append(result)
                    collected.add(idx)
                    new_count += 1

            if new_count > 0:
                last_progress = time.time()
            elif time.time() - last_progress > task_timeout:
                # Stalled — mark uncollected as timeout
                for idx, (item, _) in enumerate(async_results):
                    if idx not in collected:
                        k, eq_str = item
                        all_results.append({'status': 'failed', 'reason': 'timeout',
                                            'k': k, 'eq': eq_str})
                        n_timeout += 1
                break
            else:
                time.sleep(0.05)

        pool.terminate()
        pool.join()
        del pool, async_results
        import gc; gc.collect()
        pos += batch_size

        # Progress report
        total_done = len(all_results)
        if total_done % 500 < batch_size or pos >= len(work):
            elapsed = time.time() - t0
            rate = total_done / elapsed if elapsed > 0 else 0
            print(f"  {total_done}/{len(work)} ({rate:.0f} expr/s, "
                  f"{elapsed:.0f}s elapsed, {n_timeout} timeouts)", flush=True)

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({elapsed/len(work)*1000:.1f}ms/expr)")

    ok = [r for r in all_results if r['status'] == 'ok']
    failed = [r for r in all_results if r['status'] == 'failed']

    print(f"  OK: {len(ok)}, Failed: {len(failed)}")
    reasons = defaultdict(int)
    for r in failed:
        reasons[r['reason']] += 1
    for reason, count in sorted(reasons.items()):
        print(f"    {reason}: {count}")

    return ok, failed


def compute_rho_from_results(ok_results, max_k):
    """Compute rho(k) from parallel results."""
    print(f"\n{'k':>3} | {'|F|':>6} | {'processed':>9} | {'|D|':>6} | {'closed':>6} | {'rho':>8}")
    print("-" * 55)

    rho_data = {}
    all_func_hashes = set()
    all_records = []

    for k in range(1, max_k + 1):
        # Add functions at this complexity
        k_records = [r for r in ok_results if r['k'] == k]
        for r in k_records:
            all_func_hashes.add(r['func_hash'])
            all_records.append(r)

        # Count closed (derivative hash in function hash set)
        n_funcs = len(all_func_hashes)
        n_closed = sum(1 for r in all_records if r['deriv_hash'] in all_func_hashes)
        n_deriv_unique = len(set(r['deriv_hash'] for r in all_records))
        rho = n_closed / n_funcs if n_funcs > 0 else 0.0

        rho_data[k] = {
            'n_functions': n_funcs,
            'n_closed': n_closed,
            'n_deriv_unique': n_deriv_unique,
            'n_processed': len(all_records),
            'rho': rho,
        }

        print(f"{k:>3} | {n_funcs:>6} | {len(all_records):>9} | "
              f"{n_deriv_unique:>6} | {n_closed:>6} | {rho:>8.4f}")

    return rho_data


def build_clusters_from_results(ok_results):
    """Build derivative clusters from parallel results."""
    from esi_pipeline import DerivativeCluster, parse_expr

    clusters = defaultdict(DerivativeCluster)
    for r in ok_results:
        expr = parse_expr(r['expr_canon'])
        deriv = parse_expr(r['deriv'])
        clusters[r['deriv_hash']].add(r['k'], r['eq'], expr, deriv)

    return dict(clusters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel ESI pipeline")
    parser.add_argument("--data-dir", type=str, default="ext_maths")
    parser.add_argument("--max-complexity", type=int, default=7)
    parser.add_argument("--ncores", type=int, default=4)
    parser.add_argument("--task-timeout", type=int, default=30,
                        help="Hard timeout per expression in seconds")
    parser.add_argument("--mode", choices=["pipeline", "rho", "both"], default="both")
    parser.add_argument("--output-prefix", type=str, default=None)
    args = parser.parse_args()

    if args.output_prefix is None:
        basis = os.path.basename(args.data_dir.rstrip('/'))
        args.output_prefix = f"{basis}_k{args.max_complexity}"

    # Run pipeline
    ok, failed = run_pipeline_parallel(
        args.data_dir, args.max_complexity, args.ncores, args.task_timeout
    )

    # Compute rho
    if args.mode in ("rho", "both"):
        print(f"\n{'='*55}")
        print(f"RHO(k) MEASUREMENT")
        print(f"{'='*55}")
        rho_data = compute_rho_from_results(ok, args.max_complexity)

        rho_file = f"rho_results_{args.output_prefix}.json"
        with open(rho_file, 'w') as f:
            json.dump(rho_data, f, indent=2, default=str)
        print(f"\nSaved rho results to {rho_file}")

    # Build and save clusters
    if args.mode in ("pipeline", "both"):
        print(f"\n{'='*55}")
        print(f"BUILDING CLUSTERS")
        print(f"{'='*55}")
        clusters = build_clusters_from_results(ok)

        n_multi = sum(1 for c in clusters.values() if c.size() > 1)
        sizes = sorted([c.size() for c in clusters.values()], reverse=True)
        print(f"  Clusters: {len(clusters)}, multi-member: {n_multi}")
        print(f"  Top sizes: {sizes[:15]}")

        # Show interesting clusters
        multi = sorted(
            [(h, c) for h, c in clusters.items() if c.size() > 1],
            key=lambda x: x[1].size(), reverse=True
        )
        for h, c in multi[:10]:
            print(f"\n  Derivative: {c.derivative_str}")
            print(f"  Primitives ({c.size()}):")
            for comp, eq_str, _ in c.primitives[:8]:
                print(f"    [k={comp}] {eq_str}")
            if c.size() > 8:
                print(f"    ... and {c.size()-8} more")

        # Save
        pkl_file = f"results_{args.output_prefix}.pkl"
        data = {
            'clusters': {h: {
                'primitives': [(k, s, str(e)) for k, s, e in c.primitives],
                'derivative_str': c.derivative_str,
            } for h, c in clusters.items()},
            'failures': [(r['k'], r['eq'], r['reason']) for r in failed],
        }
        with open(pkl_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"\nSaved clusters to {pkl_file}")

    # Always save raw ok_results for downstream analysis (rho decomposition, delta)
    raw_file = f"raw_results_{args.output_prefix}.pkl"
    with open(raw_file, 'wb') as f:
        pickle.dump({'ok': ok, 'failed': failed}, f)
    print(f"Saved raw results to {raw_file}")
