"""
Mathematica CAS comparison for any basis.
MPI-parallel. Same methodology as cas_full_mathematica.py but takes --pkl and --out args.
Keep n_ranks <= 5 due to Mathematica license limits.

Usage:
  mpirun -n 5 python3 cas_mma_allbases.py --pkl results_core_maths_k10.pkl --out cas_mma_core_maths.json
"""
import json
import pickle
import time
import re
import os
import sys
import subprocess
import argparse

_pylibs = '/mnt/extraspace/hdesmond/pylibs'
if os.path.isdir(_pylibs) and _pylibs not in sys.path:
    sys.path.insert(0, _pylibs)

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

TIMEOUT = 180


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
    import sympy
    x_pos = sympy.Symbol('x', positive=True)
    param_names = sorted(set(re.findall(r'a\d+', deriv_str)))
    pos_params = {p: sympy.Symbol(p, positive=True) for p in param_names}
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
        return None


def do_one_mma(deriv_str, timeout):
    mma_expr = sympy_to_mathematica(deriv_str)
    if mma_expr is None:
        return {'status': 'error', 'error': 'conversion'}

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
            return {'status': 'success', 'leaf_count': leaf_count,
                    'result': parts[2][:200] if len(parts) > 2 else '',
                    'time': elapsed}
        elif output.startswith('FAILED|'):
            return {'status': 'unevaluated', 'time': elapsed}
        else:
            return {'status': 'error', 'error': output[:100], 'time': elapsed}
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except:
            pass
        return {'status': 'timeout', 'time': time.time() - t0}
    except Exception as e:
        return {'status': 'error', 'error': str(e)[:100], 'time': time.time() - t0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--start', type=int, default=0,
                        help='Start index into sorted work list (inclusive)')
    parser.add_argument('--end', type=int, default=-1,
                        help='End index into sorted work list (exclusive, -1=all)')
    args = parser.parse_args()

    if rank == 0:
        print(f"Loading {args.pkl}...", flush=True)
        with open(args.pkl, 'rb') as f:
            data = pickle.load(f)

        clusters = data['clusters']
        multi = [(h, c) for h, c in clusters.items() if len(c['primitives']) > 1]
        multi.sort(key=lambda x: len(x[1]['primitives']), reverse=True)

        work = []
        for h, c in multi:
            prims = c['primitives']
            min_prim = min(prims, key=lambda t: t[0])
            our_k, our_eq, our_expr = min_prim
            work.append({
                'derivative': c['derivative_str'],
                'our_k': our_k, 'our_eq': our_eq, 'our_expr': our_expr,
                'n_primitives': len(prims),
            })

        # Slice work list if --start/--end given
        end = args.end if args.end >= 0 else len(work)
        work = work[args.start:end]
        print(f"Distributing {len(work)} clusters (indices {args.start}:{end}) "
              f"across {size} ranks ({TIMEOUT}s timeout)", flush=True)

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
        res = do_one_mma(w['derivative'], TIMEOUT)
        w['mma'] = res
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
        n_success = sum(1 for r in results if r['mma']['status'] == 'success')
        n_unevaluated = sum(1 for r in results if r['mma']['status'] == 'unevaluated')
        n_timeout = sum(1 for r in results if r['mma']['status'] == 'timeout')
        n_error = sum(1 for r in results if r['mma']['status'] == 'error')
        n_fail = n_unevaluated + n_timeout + n_error

        print(f"\n{'='*70}")
        print(f"MATHEMATICA ({n_total} clusters, {size} ranks, {elapsed:.0f}s)")
        print(f"{'='*70}")
        print(f"Success: {n_success} ({100*n_success/n_total:.1f}%)")
        print(f"Unevaluated: {n_unevaluated}, Timeout: {n_timeout}, Error: {n_error}")
        print(f"Total failures: {n_fail} ({100*n_fail/n_total:.1f}%)")

        save_data = {
            'metadata': {'n_clusters': n_total, 'timeout': TIMEOUT,
                         'n_ranks': size, 'elapsed': elapsed,
                         'pkl': args.pkl},
            'summary': {'success': n_success, 'unevaluated': n_unevaluated,
                        'timeout': n_timeout, 'error': n_error,
                        'nonfg_failures': n_fail},
            'failures': [{'derivative': r['derivative'], 'our_k': r['our_k'],
                          'our_expr': r['our_expr'], 'status': r['mma']['status']}
                         for r in results if r['mma']['status'] != 'success'],
        }
        with open(args.out, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\nSaved to {args.out}")


if __name__ == '__main__':
    main()
