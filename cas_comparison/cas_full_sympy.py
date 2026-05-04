"""
Comprehensive SymPy CAS comparison on ALL multi-member clusters.
MPI-parallel. Uses best-effort strategy:
  1. Clean the integrand (strip Abs/sign, assuming x>0)
  2. Try integrate with conds='none', positive params
  3. If that fails, try manual=True
  4. Hard subprocess timeout per integral (kills C-level hangs)

Usage:
  addqueue -q redwood -n 200 -m 4 /path/to/run_cas_full_sympy.sh
"""
import json
import pickle
import time
import re
import os
import sys
import multiprocessing as _mp

_pylibs = '/mnt/extraspace/hdesmond/pylibs'
if os.path.isdir(_pylibs) and _pylibs not in sys.path:
    sys.path.insert(0, _pylibs)

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

TIMEOUT = 180
PKL = 'results_ext_log_maths_k8.pkl'


def is_fg_type(s):
    for m in re.finditer(r'pow\(Abs\(', s):
        start = m.start()
        depth = 0
        inner_start = start + 4
        for i in range(start, len(s)):
            if s[i] == '(': depth += 1
            elif s[i] == ')':
                depth -= 1
                if depth == 0:
                    inner_end = i
                    break
        else:
            continue
        inner = s[inner_start:inner_end]
        depth2 = 0
        comma_pos = None
        for j, ch in enumerate(inner):
            if ch == '(': depth2 += 1
            elif ch == ')': depth2 -= 1
            elif ch == ',' and depth2 == 0:
                comma_pos = j
                break
        if comma_pos is None:
            continue
        base_str = inner[:comma_pos]
        exp_str = inner[comma_pos + 1:]
        if (re.search(r'(?<![a-z])x(?![a-z0-9])', base_str) and
            re.search(r'(?<![a-z])x(?![a-z0-9])', exp_str)):
            return True
    return False


def _integrate_worker(deriv_str):
    """Run in subprocess. Tries cleaned fair, then manual. Can be hard-killed."""
    import sympy as sp
    import re as _re

    x_pos = sp.Symbol('x', positive=True)
    param_names = sorted(set(_re.findall(r'a\d+', deriv_str)))
    pos_params = {p: sp.Symbol(p, positive=True) for p in param_names}

    # Clean namespace: strip Abs/sign since x > 0 and params > 0
    clean_ns = {
        'x': x_pos, 'Abs': lambda a: a,
        'inv': lambda a: 1/a, 'square': lambda a: a*a,
        'sqrt': lambda a: sp.sqrt(a),
        'log': lambda a: sp.log(a),
        'pow': lambda a, b: sp.Pow(a, b),
        'sin': sp.sin, 'cos': sp.cos,
        'zoo': sp.zoo, 'pi': sp.pi, 'E': sp.E, 'exp': sp.exp,
        'sign': lambda a: 1,
        **pos_params,
    }

    # Also keep original namespace for fallback
    orig_ns = {
        'x': x_pos, 'Abs': sp.Abs,
        'inv': lambda a: 1/a, 'square': lambda a: a*a,
        'sqrt': lambda a: sp.sqrt(sp.Abs(a, evaluate=False)),
        'log': lambda a: sp.log(sp.Abs(a, evaluate=False)),
        'pow': lambda a, b: sp.Pow(sp.Abs(a, evaluate=False), b),
        'sin': sp.sin, 'cos': sp.cos,
        'zoo': sp.zoo, 'pi': sp.pi, 'E': sp.E, 'exp': sp.exp,
        **pos_params,
    }

    # Strategy 1: Clean + fair
    try:
        deriv_clean = sp.sympify(deriv_str, locals=clean_ns)
        deriv_clean = sp.simplify(deriv_clean)
        res = sp.integrate(deriv_clean, x_pos, conds='none')
        if not res.has(sp.Integral):
            ops = int(sp.count_ops(res))
            return {'status': 'success', 'strategy': 'clean_fair',
                    'ops': ops, 'result': str(res)[:200]}
    except:
        pass

    # Strategy 2: Original + fair
    try:
        deriv_orig = sp.sympify(deriv_str, locals=orig_ns)
        res = sp.integrate(deriv_orig, x_pos, conds='none')
        if not res.has(sp.Integral):
            ops = int(sp.count_ops(res))
            return {'status': 'success', 'strategy': 'orig_fair',
                    'ops': ops, 'result': str(res)[:200]}
    except:
        pass

    # Strategy 3: Clean + manual
    try:
        deriv_clean = sp.sympify(deriv_str, locals=clean_ns)
        deriv_clean = sp.simplify(deriv_clean)
        res = sp.integrate(deriv_clean, x_pos, manual=True)
        if not res.has(sp.Integral):
            ops = int(sp.count_ops(res))
            return {'status': 'success', 'strategy': 'clean_manual',
                    'ops': ops, 'result': str(res)[:200]}
    except:
        pass

    # Strategy 4: Original + manual
    try:
        deriv_orig = sp.sympify(deriv_str, locals=orig_ns)
        res = sp.integrate(deriv_orig, x_pos, manual=True)
        if not res.has(sp.Integral):
            ops = int(sp.count_ops(res))
            return {'status': 'success', 'strategy': 'orig_manual',
                    'ops': ops, 'result': str(res)[:200]}
    except:
        pass

    # All strategies failed
    return {'status': 'failed'}


def do_one_sympy(deriv_str, timeout):
    """Hard subprocess timeout wrapping all strategies."""
    t0 = time.time()
    try:
        pool = _mp.Pool(1, maxtasksperchild=1)
        ar = pool.apply_async(_integrate_worker, (deriv_str,))
        result = ar.get(timeout=timeout)
        pool.terminate()
        pool.join()
        result['time'] = time.time() - t0
        return result
    except _mp.TimeoutError:
        pool.terminate()
        pool.join()
        return {'status': 'timeout', 'time': time.time() - t0}
    except Exception as e:
        try:
            pool.terminate()
            pool.join()
        except:
            pass
        return {'status': 'error', 'error': str(e)[:100],
                'time': time.time() - t0}


def main():
    if rank == 0:
        print(f"Loading {PKL}...", flush=True)
        with open(PKL, 'rb') as f:
            data = pickle.load(f)

        clusters = data['clusters']
        multi = [(h, c) for h, c in clusters.items() if len(c['primitives']) > 1]
        multi.sort(key=lambda x: len(x[1]['primitives']), reverse=True)

        import sympy
        from esi_pipeline import parse_expr

        work = []
        for h, c in multi:
            prims = c['primitives']
            min_prim = min(prims, key=lambda t: t[0])
            our_k, our_eq, our_expr = min_prim
            try:
                expr = parse_expr(our_expr)
                our_ops = int(sympy.count_ops(expr)) if expr else 0
            except:
                our_ops = 0
            work.append({
                'derivative': c['derivative_str'],
                'our_ops': our_ops, 'our_k': our_k,
                'our_eq': our_eq, 'our_expr': our_expr,
                'n_primitives': len(prims),
            })

        print(f"Distributing {len(work)} clusters across {size} ranks "
              f"(clean+manual strategies, {TIMEOUT}s timeout)", flush=True)

        chunks = [[] for _ in range(size)]
        for i, w in enumerate(work):
            chunks[i % size].append(w)
    else:
        chunks = None

    my_work = comm.scatter(chunks, root=0)

    if rank == 0:
        print(f"Rank 0 has {len(my_work)} items.", flush=True)
        t_start = time.time()

    my_results = []
    for i, w in enumerate(my_work):
        res = do_one_sympy(w['derivative'], TIMEOUT)
        w['sympy'] = res
        my_results.append(w)

        if rank == 0 and (i + 1) % 20 == 0:
            elapsed = time.time() - t_start
            done_est = (i + 1) * size
            total = len(my_work) * size
            rate = done_est / elapsed if elapsed > 0 else 0
            eta = (total - done_est) / rate if rate > 0 else 0
            print(f"  ~{done_est}/{total} ({elapsed:.0f}s, ~{rate:.1f}/s, "
                  f"ETA ~{eta:.0f}s)", flush=True)

    all_results = comm.gather(my_results, root=0)

    if rank == 0:
        results = [r for chunk in all_results for r in chunk]
        elapsed = time.time() - t_start

        n_total = len(results)
        n_success = sum(1 for r in results if r['sympy']['status'] == 'success')
        n_failed = sum(1 for r in results if r['sympy']['status'] == 'failed')
        n_timeout = sum(1 for r in results if r['sympy']['status'] == 'timeout')
        n_error = sum(1 for r in results if r['sympy']['status'] == 'error')
        n_cas_fail = n_failed + n_timeout + n_error

        print(f"\n{'='*70}")
        print(f"SYMPY BEST-EFFORT ({n_total} clusters, {size} ranks, {elapsed:.0f}s)")
        print(f"{'='*70}")
        print(f"Success: {n_success} ({100*n_success/n_total:.1f}%)")
        print(f"  by strategy:")
        from collections import Counter
        strats = Counter(r['sympy'].get('strategy', 'none')
                         for r in results if r['sympy']['status'] == 'success')
        for s, c in strats.most_common():
            print(f"    {s}: {c}")
        print(f"Failed (all strategies): {n_failed} ({100*n_failed/n_total:.1f}%)")
        print(f"Timeout: {n_timeout} ({100*n_timeout/n_total:.1f}%)")
        print(f"Error: {n_error} ({100*n_error/n_total:.1f}%)")
        print(f"Total CAS failures: {n_cas_fail} ({100*n_cas_fail/n_total:.1f}%)")

        failures = [r for r in results if r['sympy']['status'] != 'success']
        fg_fails = [r for r in failures
                    if is_fg_type(r.get('derivative', '')) or is_fg_type(r.get('our_expr', ''))]
        nonfg_fails = [r for r in failures if r not in fg_fails]

        print(f"\nf^g failures: {len(fg_fails)}")
        print(f"Non-f^g failures: {len(nonfg_fails)}")

        nonfg_fails.sort(key=lambda r: r['our_k'])
        print(f"\nTop 20 simplest non-f^g CAS failures:")
        for r in nonfg_fails[:20]:
            print(f"  [k={r['our_k']}, {r['sympy']['status']}] "
                  f"F'= {r['derivative'][:80]}")
            print(f"    F = {r['our_expr'][:80]}")

        esi_simpler = cas_simpler = tied = 0
        esi_saved = cas_saved = 0
        for r in results:
            if r['sympy']['status'] != 'success': continue
            cas_ops = r['sympy'].get('ops', 0)
            our = r['our_ops']
            if our < cas_ops:
                esi_simpler += 1; esi_saved += cas_ops - our
            elif cas_ops < our:
                cas_simpler += 1; cas_saved += our - cas_ops
            else:
                tied += 1
        print(f"\nESI simpler: {esi_simpler}, CAS simpler: {cas_simpler}, Tied: {tied}")
        print(f"ESI total saved: {esi_saved}, CAS total saved: {cas_saved}")

        save_data = {
            'metadata': {'n_clusters': n_total, 'timeout': TIMEOUT,
                         'n_ranks': size, 'elapsed': elapsed,
                         'strategies': 'clean_fair -> orig_fair -> clean_manual -> orig_manual'},
            'summary': {'success': n_success, 'failed': n_failed,
                        'timeout': n_timeout, 'error': n_error,
                        'fg_failures': len(fg_fails),
                        'nonfg_failures': len(nonfg_fails),
                        'esi_simpler': esi_simpler, 'cas_simpler': cas_simpler,
                        'tied': tied, 'esi_saved': esi_saved, 'cas_saved': cas_saved,
                        'by_strategy': dict(strats)},
            'failures': [{'derivative': r['derivative'], 'our_k': r['our_k'],
                          'our_expr': r['our_expr'], 'status': r['sympy']['status'],
                          'is_fg': r in fg_fails} for r in failures],
            'top_nonfg': [{'derivative': r['derivative'], 'our_k': r['our_k'],
                           'our_eq': r['our_eq'], 'our_expr': r['our_expr'],
                           'status': r['sympy']['status']}
                          for r in nonfg_fails[:50]],
        }
        with open('cas_full_sympy_results.json', 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\nSaved to cas_full_sympy_results.json")


if __name__ == '__main__':
    main()
