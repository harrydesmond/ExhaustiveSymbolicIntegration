#!/usr/bin/env python3
"""MPI-parallel stress test for any basis.

Usage:
  mpirun python3 cas_stress_allbases.py --basis core_maths
  mpirun python3 cas_stress_allbases.py --basis ext_maths --sympy-only
"""
import json, time, re, os, sys, subprocess, multiprocessing as mp, argparse

_pylibs = '/mnt/extraspace/hdesmond/pylibs'
if os.path.isdir(_pylibs) and _pylibs not in sys.path:
    sys.path.insert(0, _pylibs)

from mpi4py import MPI

TIMEOUT = 600
MATH_BIN = None
for p in ['/usr/local/shared/mathematica/13.3/Executables/math', '/usr/local/bin/math']:
    if os.path.isfile(p):
        MATH_BIN = p
        break

SYMPY_STRATEGIES = ['clean_default', 'clean_fair', 'clean_manual', 'clean_heurisch',
                    'clean_meijerg', 'clean_risch', 'orig_default', 'orig_fair', 'orig_manual']


def _sympy_worker(args):
    deriv_str, strategy = args
    import sympy as sp
    x_pos = sp.Symbol('x', positive=True)
    param_names = sorted(set(re.findall(r'a\d+', deriv_str)))
    pos_params = {p: sp.Symbol(p, positive=True) for p in param_names}
    clean_ns = {
        'x': x_pos, 'Abs': lambda a: a, 'inv': lambda a: 1/a,
        'square': lambda a: a*a, 'sqrt': lambda a: sp.sqrt(a),
        'log': lambda a: sp.log(a), 'pow': lambda a, b: sp.Pow(a, b),
        'sin': sp.sin, 'cos': sp.cos, 'zoo': sp.zoo,
        'pi': sp.pi, 'E': sp.E, 'exp': sp.exp,
        'sign': lambda a: 1, **pos_params,
    }
    orig_ns = {
        'x': x_pos, 'Abs': sp.Abs, 'inv': lambda a: 1/a,
        'square': lambda a: a*a,
        'sqrt': lambda a: sp.sqrt(sp.Abs(a, evaluate=False)),
        'log': lambda a: sp.log(sp.Abs(a, evaluate=False)),
        'pow': lambda a, b: sp.Pow(sp.Abs(a, evaluate=False), b),
        'sin': sp.sin, 'cos': sp.cos, 'zoo': sp.zoo,
        'pi': sp.pi, 'E': sp.E, 'exp': sp.exp,
        **pos_params,
    }
    try:
        if strategy.startswith('clean'):
            deriv = sp.simplify(sp.sympify(deriv_str, locals=clean_ns))
        else:
            deriv = sp.sympify(deriv_str, locals=orig_ns)
        if 'manual' in strategy:
            res = sp.integrate(deriv, x_pos, manual=True)
        elif 'heurisch' in strategy:
            from sympy.integrals.heurisch import heurisch
            res = heurisch(deriv, x_pos)
            if res is None:
                return {'strategy': strategy, 'status': 'failed'}
        elif 'meijerg' in strategy:
            res = sp.integrate(deriv, x_pos, meijerg=True)
        elif 'risch' in strategy:
            res = sp.integrate(deriv, x_pos, risch=True)
        elif 'fair' in strategy:
            res = sp.integrate(deriv, x_pos, conds='none')
        else:
            res = sp.integrate(deriv, x_pos)
        if res is None or (hasattr(res, 'has') and res.has(sp.Integral)):
            return {'strategy': strategy, 'status': 'unevaluated'}
        return {'strategy': strategy, 'status': 'success', 'ops': int(sp.count_ops(res))}
    except Exception as e:
        return {'strategy': strategy, 'status': 'error', 'error': str(e)[:80]}


def run_sympy(deriv_str, strategy):
    pool = mp.Pool(1, maxtasksperchild=1)
    try:
        ar = pool.apply_async(_sympy_worker, ((deriv_str, strategy),))
        return ar.get(timeout=TIMEOUT)
    except mp.TimeoutError:
        return {'strategy': strategy, 'status': 'timeout'}
    except:
        return {'strategy': strategy, 'status': 'error'}
    finally:
        pool.terminate(); pool.join()


def run_mma(deriv_str, mma_cmd, name):
    if not MATH_BIN:
        return {'strategy': name, 'status': 'error', 'error': 'no mathematica'}
    full = (mma_cmd + '; If[FreeQ[result, Integrate], '
            'Print["SUCCESS|" <> ToString[LeafCount[result]]], '
            'Print["FAILED"]]')
    try:
        proc = subprocess.run(
            [MATH_BIN, '-noprompt', '-run', full + '; Exit[]'],
            capture_output=True, text=True, timeout=TIMEOUT)
        out = proc.stdout.strip().strip('"').strip()
        if out.startswith('SUCCESS|'):
            return {'strategy': name, 'status': 'success', 'ops': int(out.split('|')[1])}
        elif out.startswith('FAILED'):
            return {'strategy': name, 'status': 'unevaluated'}
        else:
            return {'strategy': name, 'status': 'error', 'error': out[:50]}
    except subprocess.TimeoutExpired:
        return {'strategy': name, 'status': 'timeout'}
    except Exception as e:
        return {'strategy': name, 'status': 'error', 'error': str(e)[:50]}


def test_one(r, sympy_only=False):
    deriv_str = r['derivative']
    results = {'derivative': deriv_str[:500], 'our_k': r['our_k'], 'our_expr': r['our_expr'],
               'sympy': {}, 'mathematica': {}, 'sympy_solved': False, 'mma_solved': False}

    # SymPy strategies
    for s in SYMPY_STRATEGIES:
        res = run_sympy(deriv_str, s)
        results['sympy'][s] = res
        if res.get('status') == 'success':
            results['sympy_solved'] = True
            break

    # MMA strategies
    if not sympy_only and MATH_BIN:
        import sympy
        x_pos = sympy.Symbol('x', positive=True)
        param_names = sorted(set(re.findall(r'a\d+', deriv_str)))
        pos_params = {p: sympy.Symbol(p, positive=True) for p in param_names}
        clean_ns = {
            'x': x_pos, 'Abs': lambda a: a, 'inv': lambda a: 1/a,
            'square': lambda a: a*a, 'sqrt': lambda a: sympy.sqrt(a),
            'log': lambda a: sympy.log(a), 'pow': lambda a, b: sympy.Pow(a, b),
            'sin': sympy.sin, 'cos': sympy.cos, 'zoo': sympy.zoo,
            'pi': sympy.pi, 'E': sympy.E, 'exp': sympy.exp,
            'sign': lambda a: 1, **pos_params,
        }
        try:
            clean = sympy.simplify(sympy.sympify(deriv_str, locals=clean_ns))
            mma_expr = sympy.printing.mathematica.mathematica_code(clean)
        except:
            mma_expr = None

        if mma_expr:
            assumptions = ' && '.join([p + ' > 0' for p in param_names] + ['x > 0'])
            mma_strats = {
                'clean_standard': 'result = Integrate[%s, x]' % mma_expr,
                'clean_assumptions': 'result = Integrate[%s, x, Assumptions -> {%s}]' % (mma_expr, assumptions),
                'clean_simplify': 'result = Integrate[FullSimplify[%s, Assumptions -> {%s}], x, Assumptions -> {%s}]' % (mma_expr, assumptions, assumptions),
            }
            for name, cmd in mma_strats.items():
                res = run_mma(deriv_str, cmd, name)
                results['mathematica'][name] = res
                if res.get('status') == 'success':
                    results['mma_solved'] = True
                    break

    return results


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser()
    parser.add_argument('--basis', required=True)
    parser.add_argument('--sympy-only', action='store_true')
    args = parser.parse_args()

    input_file = 'stress_input_%s.json' % args.basis
    with open(input_file) as f:
        integrals = json.load(f)

    # Distribute across ranks
    my_integrals = [integrals[i] for i in range(rank, len(integrals), size)]

    if rank == 0:
        mode = 'SymPy-only' if args.sympy_only else 'SymPy+MMA'
        print('=== Stress test: %s (%s) ===' % (args.basis, mode), flush=True)
        print('Total: %d integrals, %d ranks, ~%d per rank' % (len(integrals), size, len(my_integrals)), flush=True)

    results = []
    n_sympy_solved = 0
    n_mma_solved = 0
    t0 = time.time()

    for i, r in enumerate(my_integrals):
        res = test_one(r, sympy_only=args.sympy_only)
        results.append(res)
        if res['sympy_solved']:
            n_sympy_solved += 1
        if res['mma_solved']:
            n_mma_solved += 1

        if (i + 1) % 10 == 0 or i == len(my_integrals) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(my_integrals) - i - 1) / rate if rate > 0 else 0
            print('  [rank %d] %d/%d (%.1f/s, ETA %.0fs) sympy_solved=%d mma_solved=%d' % (
                rank, i+1, len(my_integrals), rate, eta, n_sympy_solved, n_mma_solved), flush=True)

    # Save per-rank results
    rank_file = 'stress_%s_rank%04d.json' % (args.basis, rank)
    with open(rank_file, 'w') as f:
        json.dump({'rank': rank, 'n_tested': len(results),
                   'n_sympy_solved': n_sympy_solved, 'n_mma_solved': n_mma_solved,
                   'results': results}, f)

    comm.Barrier()

    if rank == 0:
        # Merge all ranks
        import glob
        all_results = []
        total_sympy_solved = 0
        total_mma_solved = 0
        for f in sorted(glob.glob('stress_%s_rank*.json' % args.basis)):
            d = json.load(open(f))
            all_results.extend(d['results'])
            total_sympy_solved += d['n_sympy_solved']
            total_mma_solved += d['n_mma_solved']
            os.remove(f)

        n_sympy_impossible = sum(1 for r in all_results if not r['sympy_solved'])
        n_mma_impossible = sum(1 for r in all_results if not r['mma_solved'])

        summary = {
            'basis': args.basis,
            'total': len(all_results),
            'sympy_solved': total_sympy_solved,
            'sympy_impossible': n_sympy_impossible,
            'mma_solved': total_mma_solved,
            'mma_impossible': n_mma_impossible,
            'results': all_results,
        }
        outf = 'stress_results_%s.json' % args.basis
        with open(outf, 'w') as f:
            json.dump(summary, f)

        print('\n' + '=' * 60, flush=True)
        print('STRESS TEST: %s' % args.basis, flush=True)
        print('Total tested: %d' % len(all_results), flush=True)
        print('SymPy solved: %d (impossible: %d)' % (total_sympy_solved, n_sympy_impossible), flush=True)
        print('MMA solved: %d (impossible: %d)' % (total_mma_solved, n_mma_impossible), flush=True)
        print('Saved to %s' % outf, flush=True)
        print('=' * 60, flush=True)


if __name__ == '__main__':
    main()
