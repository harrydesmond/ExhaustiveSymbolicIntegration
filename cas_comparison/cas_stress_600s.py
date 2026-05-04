"""
Deep stress test: 15 integrals × multiple strategies × 600s each.
Each invocation tests ONE integral (specified by --index) with ALL strategies.
Parallelise by submitting 15 jobs.

Usage:
  python3 cas_stress_600s.py --index 0 --with-mma
"""
import json
import time
import re
import os
import sys
import subprocess
import multiprocessing as mp
import argparse

_pylibs = '/mnt/extraspace/hdesmond/pylibs'
if os.path.isdir(_pylibs) and _pylibs not in sys.path:
    sys.path.insert(0, _pylibs)

TIMEOUT = 600
MATH_BIN = '/usr/local/shared/mathematica/13.3/Executables/math'


def _sympy_worker(args):
    """Single SymPy strategy in subprocess."""
    deriv_str, strategy = args
    import sympy as sp

    x_pos = sp.Symbol('x', positive=True)
    param_names = sorted(set(re.findall(r'a\d+', deriv_str)))
    pos_params = {p: sp.Symbol(p, positive=True) for p in param_names}

    clean_ns = {
        'x': x_pos, 'Abs': lambda a: a,
        'inv': lambda a: 1/a, 'square': lambda a: a*a,
        'sqrt': lambda a: sp.sqrt(a), 'log': lambda a: sp.log(a),
        'pow': lambda a, b: sp.Pow(a, b),
        'sin': sp.sin, 'cos': sp.cos,
        'zoo': sp.zoo, 'pi': sp.pi, 'E': sp.E, 'exp': sp.exp,
        'sign': lambda a: 1, **pos_params,
    }
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

    try:
        if strategy == 'clean_default':
            deriv = sp.simplify(sp.sympify(deriv_str, locals=clean_ns))
            res = sp.integrate(deriv, x_pos)
        elif strategy == 'clean_fair':
            deriv = sp.simplify(sp.sympify(deriv_str, locals=clean_ns))
            res = sp.integrate(deriv, x_pos, conds='none')
        elif strategy == 'clean_manual':
            deriv = sp.simplify(sp.sympify(deriv_str, locals=clean_ns))
            res = sp.integrate(deriv, x_pos, manual=True)
        elif strategy == 'clean_heurisch':
            from sympy.integrals.heurisch import heurisch
            deriv = sp.simplify(sp.sympify(deriv_str, locals=clean_ns))
            res = heurisch(deriv, x_pos)
            if res is None:
                return {'strategy': strategy, 'status': 'failed'}
        elif strategy == 'clean_meijerg':
            deriv = sp.simplify(sp.sympify(deriv_str, locals=clean_ns))
            res = sp.integrate(deriv, x_pos, meijerg=True)
        elif strategy == 'orig_default':
            deriv = sp.sympify(deriv_str, locals=orig_ns)
            res = sp.integrate(deriv, x_pos)
        elif strategy == 'orig_fair':
            deriv = sp.sympify(deriv_str, locals=orig_ns)
            res = sp.integrate(deriv, x_pos, conds='none')
        elif strategy == 'orig_manual':
            deriv = sp.sympify(deriv_str, locals=orig_ns)
            res = sp.integrate(deriv, x_pos, manual=True)
        elif strategy == 'clean_risch':
            deriv = sp.simplify(sp.sympify(deriv_str, locals=clean_ns))
            res = sp.integrate(deriv, x_pos, risch=True)
        else:
            return {'strategy': strategy, 'status': 'error', 'error': 'unknown'}

        if res is None or (hasattr(res, 'has') and res.has(sp.Integral)):
            return {'strategy': strategy, 'status': 'unevaluated',
                    'result': str(res)[:200] if res else 'None'}
        ops = int(sp.count_ops(res))
        return {'strategy': strategy, 'status': 'success',
                'ops': ops, 'result': str(res)[:200]}
    except Exception as e:
        return {'strategy': strategy, 'status': 'error',
                'error': str(e)[:150]}


def run_sympy_strategy(deriv_str, strategy, timeout):
    """Hard-kill subprocess wrapper."""
    pool = mp.Pool(1, maxtasksperchild=1)
    t0 = time.time()
    try:
        ar = pool.apply_async(_sympy_worker, ((deriv_str, strategy),))
        r = ar.get(timeout=timeout)
        r['time'] = time.time() - t0
        return r
    except mp.TimeoutError:
        return {'strategy': strategy, 'status': 'timeout', 'time': timeout}
    except Exception as e:
        return {'strategy': strategy, 'status': 'error',
                'error': str(e)[:100], 'time': time.time() - t0}
    finally:
        pool.terminate(); pool.join()


def run_mma_strategy(deriv_str, strategy_name, mma_cmd, timeout):
    """Run one Mathematica strategy."""
    t0 = time.time()
    full_cmd = (
        f'{mma_cmd}; '
        f'If[FreeQ[result, Integrate], '
        f'Print["SUCCESS|" <> ToString[LeafCount[result]] <> "|" '
        f'<> ToString[result, InputForm]], '
        f'Print["FAILED|" <> ToString[result, InputForm]]]'
    )
    try:
        proc = subprocess.run(
            [MATH_BIN, '-noprompt', '-run', full_cmd + '; Exit[]'],
            capture_output=True, text=True, timeout=timeout
        )
        elapsed = time.time() - t0
        output = proc.stdout.strip().strip('"').strip()
        if output.startswith('SUCCESS|'):
            parts = output.split('|', 2)
            return {'strategy': strategy_name, 'status': 'success',
                    'ops': int(parts[1]),
                    'result': parts[2][:200] if len(parts) > 2 else '',
                    'time': elapsed}
        elif output.startswith('FAILED|'):
            return {'strategy': strategy_name, 'status': 'unevaluated',
                    'time': elapsed}
        else:
            return {'strategy': strategy_name, 'status': 'error',
                    'error': f'unexpected: {output[:80]}', 'time': elapsed}
    except subprocess.TimeoutExpired:
        return {'strategy': strategy_name, 'status': 'timeout', 'time': timeout}
    except Exception as e:
        return {'strategy': strategy_name, 'status': 'error',
                'error': str(e)[:80], 'time': time.time() - t0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--with-mma', action='store_true')
    parser.add_argument('--input', default='stress_test_15.json')
    args = parser.parse_args()

    with open(args.input) as f:
        integrals = json.load(f)

    r = integrals[args.index]
    deriv_str = r['derivative']
    print(f"Integral {args.index}: k={r['our_k']} F={r['our_expr'][:60]}", flush=True)
    print(f"  F'={deriv_str[:80]}", flush=True)

    # SymPy strategies
    sympy_strategies = [
        'clean_default', 'clean_fair', 'clean_manual', 'clean_heurisch',
        'clean_meijerg', 'clean_risch',
        'orig_default', 'orig_fair', 'orig_manual',
    ]

    results = {'index': args.index, 'our_k': r['our_k'],
               'our_expr': r['our_expr'], 'derivative': deriv_str,
               'sympy': {}, 'mathematica': {}}

    for strat in sympy_strategies:
        print(f"\n  SymPy [{strat}]...", end=' ', flush=True)
        res = run_sympy_strategy(deriv_str, strat, TIMEOUT)
        results['sympy'][strat] = res
        print(f"{res['status']} ({res.get('time',0):.1f}s)", flush=True)
        if res['status'] == 'success':
            print(f"    = {res.get('result','')[:60]}")

    if args.with_mma and os.path.isfile(MATH_BIN):
        import sympy
        # Clean the expression for MMA
        x_pos = sympy.Symbol('x', positive=True)
        param_names = sorted(set(re.findall(r'a\d+', deriv_str)))
        pos_params = {p: sympy.Symbol(p, positive=True) for p in param_names}
        clean_ns = {
            'x': x_pos, 'Abs': lambda a: a,
            'inv': lambda a: 1/a, 'square': lambda a: a*a,
            'sqrt': lambda a: sympy.sqrt(a), 'log': lambda a: sympy.log(a),
            'pow': lambda a, b: sympy.Pow(a, b),
            'sin': sympy.sin, 'cos': sympy.cos,
            'zoo': sympy.zoo, 'pi': sympy.pi, 'E': sympy.E, 'exp': sympy.exp,
            'sign': lambda a: 1, **pos_params,
        }
        try:
            clean_expr = sympy.simplify(sympy.sympify(deriv_str, locals=clean_ns))
            mma_clean = sympy.printing.mathematica.mathematica_code(clean_expr)
        except:
            mma_clean = None

        from esi_pipeline import parse_expr
        raw_expr = parse_expr(deriv_str)
        try:
            mma_raw = sympy.printing.mathematica.mathematica_code(raw_expr) if raw_expr else None
        except:
            mma_raw = None

        assumptions = ' && '.join([f'{p} > 0' for p in param_names] + ['x > 0'])

        mma_strategies = {}
        if mma_clean:
            mma_strategies['clean_standard'] = f'result = Integrate[{mma_clean}, x]'
            mma_strategies['clean_assumptions'] = f'result = Integrate[{mma_clean}, x, Assumptions -> {{{assumptions}}}]'
            mma_strategies['clean_simplify_first'] = f'result = Integrate[FullSimplify[{mma_clean}, Assumptions -> {{{assumptions}}}], x, Assumptions -> {{{assumptions}}}]'
        if mma_raw:
            mma_strategies['raw_standard'] = f'result = Integrate[{mma_raw}, x]'
            mma_strategies['raw_assumptions'] = f'result = Integrate[{mma_raw}, x, Assumptions -> {{{assumptions}}}]'
            mma_strategies['raw_refine'] = f'result = Integrate[Refine[{mma_raw}, {assumptions}], x, Assumptions -> {{{assumptions}}}]'

        for name, cmd in mma_strategies.items():
            print(f"\n  MMA [{name}]...", end=' ', flush=True)
            res = run_mma_strategy(deriv_str, name, cmd, TIMEOUT)
            results['mathematica'][name] = res
            print(f"{res['status']} ({res.get('time',0):.1f}s)", flush=True)
            if res['status'] == 'success':
                print(f"    = {res.get('result','')[:60]}")

    # Summary
    any_sympy = any(v['status'] == 'success' for v in results['sympy'].values())
    any_mma = any(v['status'] == 'success' for v in results['mathematica'].values())
    print(f"\n{'='*60}")
    print(f"VERDICT: SymPy={'SOLVED' if any_sympy else 'FAILED'}, "
          f"MMA={'SOLVED' if any_mma else 'FAILED'}")
    if not any_sympy and not any_mma:
        print(">>> CONFIRMED CAS-BLIND <<<")
    print(f"{'='*60}")

    outfile = f'stress_600s_result_{args.index:02d}.json'
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved to {outfile}")


if __name__ == '__main__':
    main()
