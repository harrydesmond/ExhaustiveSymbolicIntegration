#!/usr/bin/env python3
"""
Fast RUBI vs ESI comparison using float64 evaluation instead of mpmath.

Strategy: evaluate RUBI integrands at ESI's 60 x-points (with RUBI params
mapped to ESI a0-a9), hash the results, and look up in ESI's hash table.

Since ESI uses mpmath at 50 digits rounded to 10 sig figs, while we use
float64, hashes won't match directly. Instead we:
  1. Build a lookup of ESI's derivative_str -> hash for all clusters
  2. For each RUBI integrand, compute its numerical values at the 60 points
  3. Hash using the same rounding scheme and check for matches

Actually, easier: we just rebuild ESI's hashes from the derivative strings
using the same float64 approach, so both sides use the same precision.
"""

import glob
import hashlib
import json
import os
import pickle
import re
import signal
import sys
import time
import warnings

import numpy as np
import sympy
from sympy.parsing.mathematica import parse_mathematica

warnings.filterwarnings('ignore')

# ── Evaluation points (must match esi_pipeline.py) ──────────────────
_RNG = np.random.RandomState(42)
_N_EVAL = 60
_X_POINTS = _RNG.uniform(0.2, 5.0, _N_EVAL)
_PARAM_POINTS = {f'a{i}': _RNG.uniform(0.5, 3.0, _N_EVAL) for i in range(10)}

x = sympy.Symbol('x', positive=True)
_esi_param_syms = {f'a{i}': sympy.Symbol(f'a{i}', real=True) for i in range(10)}


def _rewrite_reciprocal_trig(expr):
    """Rewrite sec/csc/cot/sech/csch/coth in terms of sin/cos/exp
    so numpy lambdify can evaluate them correctly."""
    replacements = {
        sympy.sec: lambda arg: 1/sympy.cos(arg),
        sympy.csc: lambda arg: 1/sympy.sin(arg),
        sympy.cot: lambda arg: sympy.cos(arg)/sympy.sin(arg),
        sympy.tan: lambda arg: sympy.sin(arg)/sympy.cos(arg),
        sympy.coth: lambda arg: sympy.cosh(arg)/sympy.sinh(arg),
        sympy.sech: lambda arg: 1/sympy.cosh(arg),
        sympy.csch: lambda arg: 1/sympy.sinh(arg),
        sympy.tanh: lambda arg: sympy.sinh(arg)/sympy.cosh(arg),
    }
    for func, repl in replacements.items():
        if expr.has(func):
            expr = expr.replace(func, repl)
    return expr


def fast_fingerprint(expr):
    """Evaluate at 60 points using float64. Returns hash string or None."""
    if not hasattr(expr, 'free_symbols'):
        expr = sympy.sympify(expr)

    # Rewrite reciprocal trig/hyperbolic functions for correct numpy evaluation
    try:
        expr = _rewrite_reciprocal_trig(expr)
    except:
        pass

    free = expr.free_symbols
    has_x = x in free

    # Build lambdify args
    syms = []
    vals = []
    if has_x:
        syms.append(x)
        vals.append(_X_POINTS)
    for name, sym in _esi_param_syms.items():
        if sym in free:
            syms.append(sym)
            vals.append(_PARAM_POINTS[name])

    if not syms:
        return None

    try:
        f = sympy.lambdify(syms, expr, modules=['numpy'])
        result = f(*vals)
        if np.isscalar(result):
            result = np.full(_N_EVAL, result)
        result = np.asarray(result, dtype=np.float64)
        if not np.all(np.isfinite(result)):
            n_bad = np.sum(~np.isfinite(result))
            if n_bad > _N_EVAL * 0.3:
                return None
    except:
        return None

    # Hash using same scheme as ESI
    rounded = []
    for v in result:
        if not np.isfinite(v):
            rounded.append("None")
        elif v == 0.0:
            rounded.append("0.0")
        else:
            rounded.append(f"{v:.10e}")
    return hashlib.md5("|".join(rounded).encode()).hexdigest()


def split_mma_tuple(s):
    s = s.strip()
    if not s.startswith('{') or not s.endswith('}'):
        return None
    s = s[1:-1]
    fields = []
    depth = 0
    current = []
    for ch in s:
        if ch in '([{':
            depth += 1
            current.append(ch)
        elif ch in ')]}':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            fields.append(''.join(current))
            current = []
        else:
            current.append(ch)
    if current:
        fields.append(''.join(current))
    return fields


def parse_rubi_files(rubi_dir):
    problems = []
    files = sorted(glob.glob(os.path.join(rubi_dir, '**', '*.m'), recursive=True))
    for fpath in files:
        category = os.path.relpath(fpath, rubi_dir)
        with open(fpath, 'r', errors='replace') as f:
            content = f.read()
        for line in content.split('\n'):
            line = line.strip()
            if not line.startswith('{') or line.startswith('(*'):
                continue
            try:
                if '(*' in line:
                    line = line[:line.index('(*')]
                line = line.strip().rstrip(',')
                fields = split_mma_tuple(line)
                if fields and len(fields) == 4:
                    integrand, var, steps, antideriv = fields
                    problems.append({
                        'integrand_mma': integrand.strip(),
                        'var': var.strip(),
                        'steps': int(steps.strip()) if steps.strip().lstrip('-').isdigit() else -1,
                        'antideriv_mma': antideriv.strip(),
                        'file': category,
                    })
            except:
                pass
    return problems


def main():
    data_dir = '/home/harry/Symbolic_regression/ESI'
    rubi_dir = os.path.join(data_dir, 'rubi_test')

    # Step 1: Build ESI hash table using float64 (same points, same rounding)
    # We need to evaluate each ESI derivative_str at the same points
    print("Building ESI float64 hash table...")
    sys.stdout.flush()

    # Load ESI cluster derivative strings and their primitives
    esi_derivs = {}  # derivative_str -> list of (k, expr)
    for pkl_name in ['results_ext_log_maths_k8.pkl', 'results_trig_maths_k8.pkl']:
        pkl_path = os.path.join(data_dir, pkl_name)
        if not os.path.exists(pkl_path):
            continue
        basis = pkl_name.replace('results_', '').replace('.pkl', '')
        print(f"  Loading {basis}...", end=' ', flush=True)
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        n = 0
        for h, c in data['clusters'].items():
            dstr = c['derivative_str']
            prims = [(k, eq) for k, eq, _ in c['primitives']]
            if dstr not in esi_derivs:
                esi_derivs[dstr] = prims
                n += 1
            else:
                esi_derivs[dstr].extend(prims)
        print(f"{n} derivatives")

    print(f"  Total unique derivatives: {len(esi_derivs)}")

    # Now fingerprint all ESI derivatives using float64
    print("  Fingerprinting ESI derivatives (float64)...", flush=True)
    t0 = time.time()

    # ESI parse namespace
    esi_ns = {
        'x': x, 'Abs': lambda a: a,  # strip Abs for positive domain
        'inv': lambda a: 1/a, 'square': lambda a: a**2,
        'sqrt': lambda a: sympy.sqrt(a),
        'log': lambda a: sympy.log(a),
        'pow': lambda a, b: sympy.Pow(a, b),
        'sin': sympy.sin, 'cos': sympy.cos,
        'exp': sympy.exp,
        'zoo': sympy.zoo, 'pi': sympy.pi, 'E': sympy.E,
        'sign': lambda a: 1,
        **{k: v for k, v in _esi_param_syms.items()},
    }

    esi_hash_db = {}  # float64 hash -> primitives
    n_esi_ok = 0
    n_esi_fail = 0
    for j, (dstr, prims) in enumerate(esi_derivs.items()):
        if (j + 1) % 20000 == 0:
            print(f"    {j+1}/{len(esi_derivs)} ({time.time()-t0:.0f}s)", flush=True)
        try:
            expr = sympy.sympify(dstr, locals=esi_ns)
            if not hasattr(expr, 'free_symbols'):
                expr = sympy.sympify(expr)
        except:
            n_esi_fail += 1
            continue

        h = fast_fingerprint(expr)
        if h is None:
            n_esi_fail += 1
            continue
        n_esi_ok += 1

        if h not in esi_hash_db:
            esi_hash_db[h] = prims
        else:
            esi_hash_db[h].extend(prims)

    print(f"  ESI: {n_esi_ok} fingerprinted, {n_esi_fail} failed, "
          f"{len(esi_hash_db)} unique hashes ({time.time()-t0:.0f}s)")

    # Step 2: Parse RUBI problems
    print("\nParsing RUBI test files...")
    sys.stdout.flush()
    problems = parse_rubi_files(rubi_dir)
    print(f"Total RUBI problems: {len(problems)}")
    x_problems = [p for p in problems if p['var'] == 'x']
    print(f"Problems with var=x: {len(x_problems)}")

    # Parameter mapping
    rubi_to_esi = {
        'a': 'a0', 'b': 'a1', 'c': 'a2', 'd': 'a3',
        'e': 'a4', 'f': 'a5', 'g': 'a6', 'n': 'a7', 'p': 'a8', 'm': 'a9',
    }

    # Step 3: Fingerprint RUBI integrands
    print(f"\nFingerprinting {len(x_problems)} RUBI integrands...")
    sys.stdout.flush()
    t0 = time.time()

    stats = {'parsed': 0, 'parse_fail': 0, 'skip_params': 0,
             'fingerprinted': 0, 'fp_fail': 0, 'matched': 0, 'timeout': 0}
    matches = []

    for i, prob in enumerate(x_problems):
        if (i + 1) % 2000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(x_problems) - i - 1) / rate
            print(f"  {i+1}/{len(x_problems)} ({elapsed:.0f}s, {rate:.1f}/s, "
                  f"ETA {eta:.0f}s, {stats['matched']} matches)", flush=True)

        # Parse
        try:
            expr = parse_mathematica(prob['integrand_mma'])
        except:
            stats['parse_fail'] += 1
            continue
        if expr is None:
            stats['parse_fail'] += 1
            continue
        stats['parsed'] += 1

        # Check x presence
        x_sym = sympy.Symbol('x')
        if x_sym not in expr.free_symbols and x not in expr.free_symbols:
            continue
        expr = expr.subs(x_sym, x)

        # Skip expressions with imaginary unit (ESI is real-valued only)
        if expr.has(sympy.I):
            stats['skip_params'] += 1
            continue

        # Map params
        free = expr.free_symbols - {x}
        unmappable = {s for s in free if s.name not in rubi_to_esi}
        if unmappable:
            stats['skip_params'] += 1
            continue

        subs = {}
        for s in free:
            subs[s] = _esi_param_syms[rubi_to_esi[s.name]]
        try:
            mapped_expr = expr.subs(subs)
        except:
            stats['parse_fail'] += 1
            continue

        # Fingerprint with timeout
        def _timeout(signum, frame):
            raise TimeoutError()
        old = signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(3)
        try:
            h = fast_fingerprint(mapped_expr)
        except:
            h = None
            stats['timeout'] += 1
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)

        if h is None:
            stats['fp_fail'] += 1
            continue
        stats['fingerprinted'] += 1

        if h in esi_hash_db:
            stats['matched'] += 1
            prims = esi_hash_db[h]
            simplest = min(prims, key=lambda t: (t[0], len(t[1])))
            matches.append({
                'rubi_integrand': prob['integrand_mma'],
                'rubi_antideriv': prob['antideriv_mma'],
                'rubi_steps': prob['steps'],
                'rubi_file': prob['file'],
                'esi_primitive': simplest[1],
                'esi_complexity': simplest[0],
                'n_esi_primitives': len(prims),
            })

    elapsed = time.time() - t0

    # Report
    print(f"\n{'='*70}")
    print(f"RUBI vs ESI COMPARISON")
    print(f"{'='*70}")
    print(f"Total RUBI problems: {len(problems)}")
    print(f"  with var=x: {len(x_problems)}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if stats['fingerprinted'] > 0:
        print(f"  match rate: {100*stats['matched']/stats['fingerprinted']:.1f}%")
    print(f"  Time: {elapsed:.0f}s")

    from collections import Counter
    print(f"\nMatches by RUBI category:")
    cats = Counter()
    for m in matches:
        cat = m['rubi_file'].split('/')[0] if '/' in m['rubi_file'] else m['rubi_file']
        cats[cat] += 1
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")

    print(f"\nSample matches:")
    for m in matches[:20]:
        print(f"  RUBI: ∫ {m['rubi_integrand'][:60]} dx  [{m['rubi_steps']} steps]")
        print(f"  ESI:  F(x) = {m['esi_primitive'][:60]} [k={m['esi_complexity']}]")
        print()

    hard_rubi = sorted([m for m in matches if m['rubi_steps'] >= 5],
                       key=lambda m: -m['rubi_steps'])
    print(f"Hard RUBI problems (≥5 steps) matched by ESI: {len(hard_rubi)}")
    for m in hard_rubi[:15]:
        print(f"  [{m['rubi_steps']} steps] ∫ {m['rubi_integrand'][:55]} dx")
        print(f"    ESI: {m['esi_primitive'][:60]} [k={m['esi_complexity']}]")

    # Save
    results = {'summary': stats, 'matches': matches,
               'total_rubi': len(problems), 'var_x': len(x_problems)}
    with open(os.path.join(data_dir, 'rubi_esi_comparison.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to rubi_esi_comparison.json")


if __name__ == '__main__':
    main()
