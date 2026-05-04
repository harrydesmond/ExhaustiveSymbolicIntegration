"""
Comprehensive Mathematica CAS comparison on ALL multi-member clusters.
MPI-parallel: each rank processes its batch serially, calling math subprocess per integral.

Usage:
  addqueue -q redwood -n 400 -m 4 /path/to/run_cas_full_mma.sh
"""
import json
import pickle
import time
import re
import os
import sys
import subprocess

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


def _find_math_binary():
    for path in ['/usr/local/shared/mathematica/13.3/Executables/math',
                 '/usr/local/bin/math']:
        if os.path.isfile(path):
            return path
    import shutil
    found = shutil.which('math')
    return found or 'math'


MATH_BIN = _find_math_binary()


def sympy_to_mathematica(deriv_str):
    """Convert derivative string to Mathematica input form.
    First cleans Abs/sign (since x>0, params>0), then converts."""
    import sympy

    x_pos = sympy.Symbol('x', positive=True)
    param_names = sorted(set(re.findall(r'a\d+', deriv_str)))
    pos_params = {p: sympy.Symbol(p, positive=True) for p in param_names}

    # Clean namespace: strip Abs/sign
    clean_ns = {
        'x': x_pos, 'Abs': lambda a: a,
        'inv': lambda a: 1/a, 'square': lambda a: a*a,
        'sqrt': lambda a: sympy.sqrt(a),
        'log': lambda a: sympy.log(a),
        'pow': lambda a, b: sympy.Pow(a, b),
        'sin': sympy.sin, 'cos': sympy.cos,
        'zoo': sympy.zoo, 'pi': sympy.pi, 'E': sympy.E, 'exp': sympy.exp,
        'sign': lambda a: 1,
        **pos_params,
    }

    try:
        expr = sympy.sympify(deriv_str, locals=clean_ns)
        expr = sympy.simplify(expr)
        return sympy.printing.mathematica.mathematica_code(expr)
    except:
        # Fallback: try raw conversion without cleaning
        from esi_pipeline import parse_expr
        expr = parse_expr(deriv_str)
        if expr is None:
            return None
        try:
            return sympy.printing.mathematica.mathematica_code(expr)
        except:
            return None


def do_one_mma(deriv_str, timeout):
    """Integrate one derivative with Mathematica subprocess.
    Adds Assumptions -> {x > 0} for best results."""
    mma_expr = sympy_to_mathematica(deriv_str)
    if mma_expr is None:
        return {'status': 'error', 'error': 'conversion'}

    # Build assumptions string
    param_names = sorted(set(re.findall(r'a\d+', deriv_str)))
    assumptions = ' && '.join([f'{p} > 0' for p in param_names] + ['x > 0'])

    cmd = (
        f'result = Integrate[{mma_expr}, x, Assumptions -> {{{assumptions}}}]; '
        f'If[FreeQ[result, Integrate], '
        f'Print["SUCCESS|" <> ToString[LeafCount[result]] <> "|" <> ToString[result, InputForm]], '
        f'Print["FAILED|" <> ToString[result, InputForm]]]'
    )

    t0 = time.time()
    try:
        proc = subprocess.run(
            [MATH_BIN, '-noprompt', '-run', cmd + '; Exit[]'],
            capture_output=True, text=True, timeout=timeout
        )
        elapsed = time.time() - t0
        output = proc.stdout.strip()
        for q in ['"', '\u201c', '\u201d']:
            output = output.strip(q)
        output = output.strip()

        if output.startswith('SUCCESS|'):
            parts = output.split('|', 2)
            leaf_count = int(parts[1])
            result_str = parts[2] if len(parts) > 2 else ''
            return {'status': 'success', 'result': result_str[:200],
                    'ops': leaf_count, 'time': elapsed}
        elif output.startswith('FAILED|'):
            return {'status': 'unevaluated', 'result': output[7:][:200],
                    'time': elapsed}
        else:
            return {'status': 'error',
                    'error': f'unexpected: {output[:100]}', 'time': elapsed}
    except subprocess.TimeoutExpired:
        return {'status': 'timeout', 'time': time.time() - t0}
    except Exception as e:
        return {'status': 'error', 'error': str(e)[:100],
                'time': time.time() - t0}


def main():
    if rank == 0:
        print(f"Mathematica binary: {MATH_BIN}", flush=True)
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

        print(f"Distributing {len(work)} clusters across {size} ranks", flush=True)
        chunks = [[] for _ in range(size)]
        for i, w in enumerate(work):
            chunks[i % size].append(w)
    else:
        chunks = None

    my_work = comm.scatter(chunks, root=0)

    if rank == 0:
        print(f"Rank 0 has {len(my_work)} items. Timeout={TIMEOUT}s.", flush=True)
        t_start = time.time()

    my_results = []
    for i, w in enumerate(my_work):
        res = do_one_mma(w['derivative'], TIMEOUT)
        w['mathematica'] = res
        my_results.append(w)

        if rank == 0 and (i + 1) % 10 == 0:
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
        n_success = sum(1 for r in results if r['mathematica']['status'] == 'success')
        n_uneval = sum(1 for r in results if r['mathematica']['status'] == 'unevaluated')
        n_timeout = sum(1 for r in results if r['mathematica']['status'] == 'timeout')
        n_error = sum(1 for r in results if r['mathematica']['status'] == 'error')
        n_fail = n_uneval + n_timeout + n_error

        print(f"\n{'='*70}")
        print(f"MATHEMATICA COMPREHENSIVE ({n_total} clusters, {size} ranks, {elapsed:.0f}s)")
        print(f"{'='*70}")
        print(f"Success: {n_success} ({100*n_success/n_total:.1f}%)")
        print(f"Unevaluated: {n_uneval} ({100*n_uneval/n_total:.1f}%)")
        print(f"Timeout: {n_timeout} ({100*n_timeout/n_total:.1f}%)")
        print(f"Error: {n_error} ({100*n_error/n_total:.1f}%)")
        print(f"Total failures: {n_fail} ({100*n_fail/n_total:.1f}%)")

        failures = [r for r in results if r['mathematica']['status'] != 'success']
        fg_fails = [r for r in failures
                    if is_fg_type(r.get('derivative', '')) or is_fg_type(r.get('our_expr', ''))]
        nonfg_fails = [r for r in failures if r not in fg_fails]

        print(f"\nf^g failures: {len(fg_fails)}")
        print(f"Non-f^g failures: {len(nonfg_fails)}")

        nonfg_fails.sort(key=lambda r: r['our_k'])
        print(f"\nTop 20 simplest non-f^g failures:")
        for r in nonfg_fails[:20]:
            print(f"  [k={r['our_k']}, {r['mathematica']['status']}] "
                  f"F'= {r['derivative'][:80]}")
            print(f"    F = {r['our_expr'][:80]}")

        esi_simpler = cas_simpler = tied = 0
        for r in results:
            if r['mathematica']['status'] != 'success': continue
            cas_ops = r['mathematica'].get('ops', 0)
            our = r['our_ops']
            if our < cas_ops: esi_simpler += 1
            elif cas_ops < our: cas_simpler += 1
            else: tied += 1
        print(f"\nESI simpler: {esi_simpler}, MMA simpler: {cas_simpler}, Tied: {tied}")

        save_data = {
            'metadata': {'n_clusters': n_total, 'timeout': TIMEOUT,
                         'n_ranks': size, 'elapsed': elapsed},
            'summary': {'success': n_success, 'unevaluated': n_uneval,
                        'timeout': n_timeout, 'error': n_error,
                        'fg_failures': len(fg_fails),
                        'nonfg_failures': len(nonfg_fails),
                        'esi_simpler': esi_simpler, 'cas_simpler': cas_simpler,
                        'tied': tied},
            'failures': [{'derivative': r['derivative'], 'our_k': r['our_k'],
                          'our_expr': r['our_expr'],
                          'status': r['mathematica']['status'],
                          'is_fg': r in fg_fails} for r in failures],
            'top_nonfg': [{'derivative': r['derivative'], 'our_k': r['our_k'],
                           'our_eq': r['our_eq'], 'our_expr': r['our_expr'],
                           'status': r['mathematica']['status']}
                          for r in nonfg_fails[:50]],
        }
        with open('cas_full_mathematica_results.json', 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\nSaved to cas_full_mathematica_results.json")


if __name__ == '__main__':
    main()
