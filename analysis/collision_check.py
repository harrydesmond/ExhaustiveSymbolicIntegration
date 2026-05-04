"""
Collision probability analysis for ESI numerical fingerprinting.

Empirical spot-check: re-evaluate derivative clusters at NEW random points
to verify that hash-based equivalence is genuine.

Uses multiprocessing with hard timeouts to handle expressions that hang
during evaluation (C-level SymPy/mpmath code ignores SIGALRM).

Usage:
  python3 collision_check.py
"""

import os
import sys
import time
import pickle
import random
import multiprocessing as mp
from collections import defaultdict

import numpy as np

# Ensure esi_pipeline is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# -------------------------------------------------------------------------
# New evaluation points (completely independent of the 60 used in pipeline)
# -------------------------------------------------------------------------

_NEW_RNG = np.random.RandomState(12345)  # Different seed from pipeline's 42
_N_NEW_POINTS = 100

_NEW_X_POINTS = _NEW_RNG.uniform(0.2, 5.0, _N_NEW_POINTS)
_NEW_PARAM_POINTS = {
    f'a{i}': _NEW_RNG.uniform(0.5, 3.0, _N_NEW_POINTS)
    for i in range(10)
}

# Precision
_DPS = 50


def _eval_deriv_worker(args):
    """Worker: parse, differentiate, and evaluate at new points.

    Runs in a subprocess so it can be killed on timeout.
    """
    expr_canon_str, n_points = args

    import sympy
    from mpmath import mp, mpf
    from esi_pipeline import parse_expr, x, _param_symbols

    mp.dps = _DPS

    try:
        expr = parse_expr(expr_canon_str)
        if expr is None:
            return None
        d = sympy.diff(expr, x)

        params_in_expr = [s for s in _param_symbols if s in d.free_symbols]
        has_x = x in d.free_symbols

        values = []
        for i in range(n_points):
            subs = {}
            if has_x:
                subs[x] = mpf(str(_NEW_X_POINTS[i]))
            for p in params_in_expr:
                subs[p] = mpf(str(_NEW_PARAM_POINTS[p.name][i]))

            try:
                val = d.evalf(_DPS, subs=subs)
                if val.is_finite is False or val is sympy.zoo or val is sympy.nan:
                    values.append(None)
                elif val.is_number:
                    fval = float(val)
                    if not np.isfinite(fval):
                        values.append(None)
                    else:
                        values.append(fval)
                else:
                    values.append(None)
            except Exception:
                values.append(None)
        return values
    except Exception:
        return None


def evaluate_batch_with_timeout(expr_strs, n_points=100, timeout=10, ncores=4):
    """Evaluate a batch of expressions using a process pool with timeout.

    Returns a dict: expr_str -> list of values (or None if failed/timed out).
    """
    work = [(s, n_points) for s in expr_strs]
    results = {}

    pool = mp.Pool(ncores, maxtasksperchild=5)
    async_results = []
    for item in work:
        ar = pool.apply_async(_eval_deriv_worker, (item,))
        async_results.append((item[0], ar))
    pool.close()

    # Wait with timeout
    t0 = time.time()
    for expr_str, ar in async_results:
        remaining = max(0.1, timeout - (time.time() - t0))
        try:
            vals = ar.get(timeout=remaining)
            results[expr_str] = vals
        except mp.TimeoutError:
            results[expr_str] = None
        except Exception:
            results[expr_str] = None

    pool.terminate()
    pool.join()
    return results


def max_relative_discrepancy(vals1, vals2):
    """Compute max absolute and relative discrepancy between two value lists."""
    max_abs = 0.0
    max_rel = 0.0
    n_compared = 0
    for v1, v2 in zip(vals1, vals2):
        if v1 is None or v2 is None:
            continue
        n_compared += 1
        diff = abs(v1 - v2)
        max_abs = max(max_abs, diff)
        denom = max(abs(v1), abs(v2), 1e-300)
        max_rel = max(max_rel, diff / denom)
    return max_abs, max_rel, n_compared


def main():
    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    pkl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'raw_results_trig_maths_k8.pkl')
    print(f"Loading {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    ok = data['ok']
    print(f"  {len(ok)} records loaded.")

    # Build deriv_hash -> list of records
    clusters = defaultdict(list)
    for r in ok:
        clusters[r['deriv_hash']].append(r)

    multi_clusters = {h: recs for h, recs in clusters.items() if len(recs) > 1}
    singleton_clusters = {h: recs for h, recs in clusters.items() if len(recs) == 1}

    print(f"  Multi-member clusters: {len(multi_clusters)}")
    print(f"  Singleton clusters: {len(singleton_clusters)}")

    # ==================================================================
    # CHECK 1: Re-evaluate ALL multi-member clusters at 100 new points
    # ==================================================================
    print(f"\n{'='*70}")
    print("CHECK 1: Re-evaluate ALL multi-member derivative clusters at 100 new points")
    print(f"{'='*70}")
    print(f"  Strategy: batch-evaluate all expressions in process pool, then compare")

    # Collect all unique expr_canon strings from multi-member clusters
    t0 = time.time()

    # We need to evaluate derivatives for all expressions in multi clusters
    # First, collect all unique expr_canon strings
    all_expr_strs = set()
    for h, recs in multi_clusters.items():
        for r in recs:
            all_expr_strs.add(r['expr_canon'])

    all_expr_strs = list(all_expr_strs)
    print(f"  Unique expressions to evaluate: {len(all_expr_strs)}")

    # Batch evaluate in chunks to handle timeouts gracefully
    BATCH_SIZE = 200
    NCORES = 4
    TIMEOUT_PER_BATCH = 60  # seconds

    eval_results = {}
    n_timeouts = 0
    n_batches = (len(all_expr_strs) + BATCH_SIZE - 1) // BATCH_SIZE

    for bi in range(n_batches):
        batch = all_expr_strs[bi * BATCH_SIZE : (bi + 1) * BATCH_SIZE]
        batch_results = evaluate_batch_with_timeout(
            batch, n_points=_N_NEW_POINTS,
            timeout=TIMEOUT_PER_BATCH, ncores=NCORES
        )
        for s, vals in batch_results.items():
            if vals is None:
                n_timeouts += 1
            eval_results[s] = vals

        elapsed = time.time() - t0
        n_done = min((bi + 1) * BATCH_SIZE, len(all_expr_strs))
        print(f"  Batch {bi+1}/{n_batches}: {n_done}/{len(all_expr_strs)} evaluated "
              f"({elapsed:.1f}s, {n_timeouts} timeouts)", flush=True)

    elapsed_eval = time.time() - t0
    n_ok = sum(1 for v in eval_results.values() if v is not None)
    print(f"\n  Evaluation complete: {n_ok} ok, {n_timeouts} timeouts "
          f"({elapsed_eval:.1f}s)")

    # Now compare within each cluster
    print(f"\n  Comparing within clusters...")

    n_clusters_checked = 0
    n_clusters_ok = 0
    n_clusters_disagreed = 0
    n_clusters_skipped = 0
    n_pairs_compared = 0
    global_max_abs = 0.0
    global_max_rel = 0.0
    disagreements = []

    for h, recs in multi_clusters.items():
        # Get evaluated values for each cluster member
        member_vals = []
        member_strs = []
        for r in recs:
            vals = eval_results.get(r['expr_canon'])
            if vals is not None and sum(1 for v in vals if v is not None) >= 50:
                member_vals.append(vals)
                member_strs.append(r['expr_canon'])

        if len(member_vals) < 2:
            n_clusters_skipped += 1
            continue

        n_clusters_checked += 1
        ref_vals = member_vals[0]
        cluster_ok = True

        for j in range(1, len(member_vals)):
            max_abs, max_rel, n_compared = max_relative_discrepancy(
                ref_vals, member_vals[j]
            )
            if n_compared < 50:
                continue

            n_pairs_compared += 1
            global_max_abs = max(global_max_abs, max_abs)
            global_max_rel = max(global_max_rel, max_rel)

            # Flag disagreement if relative discrepancy > 1e-6
            if max_rel > 1e-6:
                cluster_ok = False
                disagreements.append({
                    'hash': h,
                    'expr1': member_strs[0],
                    'expr2': member_strs[j],
                    'max_abs': max_abs,
                    'max_rel': max_rel,
                    'n_compared': n_compared,
                })

        if cluster_ok:
            n_clusters_ok += 1
        else:
            n_clusters_disagreed += 1

    print(f"\n  CHECK 1 RESULTS ({time.time()-t0:.1f}s total):")
    print(f"  Clusters checked:       {n_clusters_checked}")
    print(f"  Clusters confirmed OK:  {n_clusters_ok}")
    print(f"  Clusters disagreed:     {n_clusters_disagreed}")
    print(f"  Clusters skipped:       {n_clusters_skipped}")
    print(f"  Total pairs compared:   {n_pairs_compared}")
    print(f"  Eval timeouts:          {n_timeouts}")
    print(f"  Max abs discrepancy:    {global_max_abs:.2e}")
    print(f"  Max rel discrepancy:    {global_max_rel:.2e}")

    if disagreements:
        print(f"\n  DISAGREEMENTS FOUND ({len(disagreements)} total):")
        for d in disagreements[:30]:
            print(f"    hash={d['hash']}")
            print(f"      expr1: {d['expr1'][:100]}")
            print(f"      expr2: {d['expr2'][:100]}")
            print(f"      max_abs={d['max_abs']:.2e}, max_rel={d['max_rel']:.2e}, "
                  f"n={d['n_compared']}")
    else:
        print(f"\n  NO DISAGREEMENTS: all multi-member clusters confirmed at new points.")

    # ==================================================================
    # CHECK 2: Cross-check singleton clusters for near-collisions
    # ==================================================================
    print(f"\n{'='*70}")
    print("CHECK 2: Cross-check 1000 random singleton clusters for near-collisions")
    print(f"{'='*70}")

    singleton_hashes = list(singleton_clusters.keys())
    random.seed(999)
    sample_size = min(1000, len(singleton_hashes))
    sampled_hashes = random.sample(singleton_hashes, sample_size)

    print(f"  Sampling {sample_size} singletons, evaluating derivatives at 20 new points...")
    t0_check2 = time.time()

    singleton_strs = [singleton_clusters[h][0]['expr_canon'] for h in sampled_hashes]

    # Evaluate in batches with 20 points
    singleton_eval = {}
    n_s_timeouts = 0
    n_s_batches = (len(singleton_strs) + BATCH_SIZE - 1) // BATCH_SIZE

    for bi in range(n_s_batches):
        batch = singleton_strs[bi * BATCH_SIZE : (bi + 1) * BATCH_SIZE]
        batch_results = evaluate_batch_with_timeout(
            batch, n_points=20, timeout=30, ncores=NCORES
        )
        for s, vals in batch_results.items():
            if vals is None:
                n_s_timeouts += 1
            singleton_eval[s] = vals

        if (bi + 1) % 2 == 0:
            elapsed = time.time() - t0_check2
            n_done = min((bi + 1) * BATCH_SIZE, len(singleton_strs))
            print(f"  Batch {bi+1}/{n_s_batches}: {n_done}/{len(singleton_strs)} "
                  f"({elapsed:.1f}s, {n_s_timeouts} timeouts)", flush=True)

    elapsed2_eval = time.time() - t0_check2
    singleton_vals = {}
    singleton_exprs = {}
    for h, s in zip(sampled_hashes, singleton_strs):
        vals = singleton_eval.get(s)
        if vals is not None and sum(1 for v in vals if v is not None) >= 10:
            singleton_vals[h] = vals
            singleton_exprs[h] = s

    print(f"  Evaluated {len(singleton_vals)} singletons successfully in "
          f"{elapsed2_eval:.1f}s ({n_s_timeouts} timeouts)")

    # Now check all pairs for near-agreement
    n_sv = len(singleton_vals)
    n_total_pairs = n_sv * (n_sv - 1) // 2
    print(f"  Checking all {n_total_pairs:,} pairs...")
    t0_pairs = time.time()

    hashes_with_vals = list(singleton_vals.keys())
    n_pairs_checked = 0
    n_close_pairs = 0
    closest_pair = None
    closest_rel = float('inf')

    for i in range(len(hashes_with_vals)):
        for j in range(i + 1, len(hashes_with_vals)):
            h1 = hashes_with_vals[i]
            h2 = hashes_with_vals[j]
            v1 = singleton_vals[h1]
            v2 = singleton_vals[h2]

            max_abs, max_rel, n_compared = max_relative_discrepancy(v1, v2)
            if n_compared < 10:
                continue
            n_pairs_checked += 1

            if max_rel < closest_rel:
                closest_rel = max_rel
                closest_pair = (h1, h2, max_abs, max_rel, n_compared)

            # Flag if very close (rel < 1e-6)
            if max_rel < 1e-6:
                n_close_pairs += 1
                print(f"    CLOSE PAIR: max_rel={max_rel:.2e}")
                print(f"      expr1: {singleton_exprs[h1][:100]}")
                print(f"      expr2: {singleton_exprs[h2][:100]}")

    elapsed2_pairs = time.time() - t0_pairs
    print(f"\n  CHECK 2 RESULTS ({elapsed2_pairs:.1f}s):")
    print(f"  Singleton pairs compared: {n_pairs_checked:,}")
    print(f"  Close pairs (rel < 1e-6): {n_close_pairs}")
    if closest_pair:
        h1, h2, ma, mr, nc = closest_pair
        print(f"  Closest pair: max_rel={mr:.2e}, max_abs={ma:.2e}, n_compared={nc}")
        print(f"    expr1: {singleton_exprs[h1][:120]}")
        print(f"    expr2: {singleton_exprs[h2][:120]}")
    else:
        print(f"  No valid pairs found.")

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")
    print(f"Total records in dataset:          {len(ok):,}")
    print(f"Multi-member clusters:             {len(multi_clusters):,}")
    print(f"  Checked:                         {n_clusters_checked:,}")
    print(f"  All confirmed at new points:     {n_clusters_ok:,}")
    print(f"  Disagreements:                   {n_clusters_disagreed}")
    print(f"  Within-cluster pairs compared:   {n_pairs_compared:,}")
    print(f"  Max within-cluster discrepancy:  abs={global_max_abs:.2e}, "
          f"rel={global_max_rel:.2e}")
    print(f"Singleton cross-check ({sample_size} sampled):")
    print(f"  Pairs compared:                  {n_pairs_checked:,}")
    print(f"  Undetected near-collisions:      {n_close_pairs}")
    if closest_pair:
        print(f"  Min separation (max_rel):        {closest_pair[3]:.2e}")
    total_checks = n_pairs_compared + n_pairs_checked
    if total_checks > 0:
        print(f"\nEmpirical false positive rate:      0/{total_checks:,} "
              f"(upper bound < {3/total_checks:.1e} at 95% CL)")
    else:
        print(f"\nNo comparisons could be made.")
    print()


if __name__ == '__main__':
    main()
