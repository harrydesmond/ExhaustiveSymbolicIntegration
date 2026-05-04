#!/usr/bin/env python3
"""
Compare RUBI test suite against ESI database.

1. Parse all RUBI test problems (Mathematica syntax -> SymPy)
2. Fingerprint each integrand using ESI's scheme
3. Look up in ESI's derivative hash table
4. Report coverage and identify interesting overlaps
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

import numpy as np
import sympy
from sympy.parsing.mathematica import parse_mathematica
from mpmath import mp, mpf

# ── ESI fingerprinting (must match esi_pipeline.py exactly) ──────────

_RNG = np.random.RandomState(42)
_N_EVAL = 60
_X_POINTS = _RNG.uniform(0.2, 5.0, _N_EVAL)
_PARAM_POINTS = {f'a{i}': _RNG.uniform(0.5, 3.0, _N_EVAL) for i in range(10)}
_DPS = 50

x = sympy.Symbol('x', positive=True)


def fingerprint(expr, param_vals):
    mp.dps = _DPS
    has_x = x in expr.free_symbols
    values = []
    n_failed = 0
    for i in range(_N_EVAL):
        subs = {}
        if has_x:
            subs[x] = mpf(str(_X_POINTS[i]))
        for p, vals in param_vals.items():
            if p in expr.free_symbols:
                subs[p] = mpf(str(vals[i]))
        try:
            val = expr.evalf(_DPS, subs=subs)
            if val.is_finite is False or val is sympy.zoo or val is sympy.nan:
                n_failed += 1
                values.append(None)
            elif val.is_number:
                rounded = float(val)
                if not np.isfinite(rounded):
                    n_failed += 1
                    values.append(None)
                else:
                    values.append(rounded)
            else:
                n_failed += 1
                values.append(None)
        except Exception:
            n_failed += 1
            values.append(None)
    if n_failed > _N_EVAL * 0.3:
        return None
    return tuple(values)


def fp_to_hash(fp):
    if fp is None:
        return None
    rounded = []
    for v in fp:
        if v is None:
            rounded.append("None")
        elif v == 0.0:
            rounded.append("0.0")
        else:
            rounded.append(f"{v:.10e}")
    return hashlib.md5("|".join(rounded).encode()).hexdigest()


# ── Parse RUBI test files ──────────────────────────────────────────

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
                    var = var.strip()
                    try:
                        steps = int(steps.strip())
                    except:
                        steps = -1
                    problems.append({
                        'integrand_mma': integrand.strip(),
                        'var': var,
                        'steps': steps,
                        'antideriv_mma': antideriv.strip(),
                        'file': category,
                    })
            except:
                pass
    return problems


# ── Main ──────────────────────────────────────────────────────────

def main():
    data_dir = '/home/harry/Symbolic_regression/ESI'
    rubi_dir = '/home/harry/Symbolic_regression/ESI/rubi_test'

    # Load ESI hash index
    print("Loading ESI database...")
    sys.stdout.flush()
    pkl_path = os.path.join(data_dir, 'results_ext_log_maths_k8.pkl')
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    db = {}
    for h, c in data['clusters'].items():
        prims = [(k, eq) for k, eq, _ in c['primitives']]
        db[h] = prims
    print(f"  {len(db)} integrands loaded")

    # Also load trig database
    trig_pkl = os.path.join(data_dir, 'results_trig_maths_k8.pkl')
    if os.path.exists(trig_pkl):
        print("Loading trig database...", end=' ', flush=True)
        with open(trig_pkl, 'rb') as f:
            tdata = pickle.load(f)
        n_new = 0
        for h, c in tdata['clusters'].items():
            prims = [(k, eq) for k, eq, _ in c['primitives']]
            if h not in db:
                db[h] = prims
                n_new += 1
            else:
                db[h].extend(prims)
        print(f"{n_new} new integrands (total {len(db)})")

    # Parse RUBI problems
    print("\nParsing RUBI test files...")
    sys.stdout.flush()
    problems = parse_rubi_files(rubi_dir)
    print(f"Total RUBI problems: {len(problems)}")
    x_problems = [p for p in problems if p['var'] == 'x']
    print(f"Problems with var=x: {len(x_problems)}")

    # Map RUBI parameter names -> ESI parameter symbols
    rubi_to_esi = {
        'a': 'a0', 'b': 'a1', 'c': 'a2', 'd': 'a3',
        'e': 'a4', 'f': 'a5', 'g': 'a6', 'n': 'a7', 'p': 'a8', 'm': 'a9',
    }
    esi_syms = {name: sympy.Symbol(name, real=True) for name in _PARAM_POINTS}

    # Process
    print(f"\nFingerprinting {len(x_problems)} integrands...")
    sys.stdout.flush()
    t0 = time.time()

    n_parsed = 0
    n_fingerprinted = 0
    n_matched = 0
    matches = []
    parse_failures = 0
    fp_failures = 0
    n_skipped_params = 0
    n_timeout = 0

    for i, prob in enumerate(x_problems):
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(x_problems) - i - 1) / rate
            print(f"  {i+1}/{len(x_problems)} ({elapsed:.0f}s, {rate:.1f}/s, "
                  f"ETA {eta:.0f}s, {n_matched} matches)", flush=True)

        # Parse Mathematica -> SymPy
        try:
            expr = parse_mathematica(prob['integrand_mma'])
        except:
            parse_failures += 1
            continue
        if expr is None:
            parse_failures += 1
            continue
        n_parsed += 1

        # Check if x is present
        x_sym = sympy.Symbol('x')
        if x_sym not in expr.free_symbols and x not in expr.free_symbols:
            continue

        # Substitute x -> positive x
        expr = expr.subs(x_sym, x)

        # Map RUBI params to ESI params
        free = expr.free_symbols - {x}
        unmappable = {s for s in free if s.name not in rubi_to_esi}
        if unmappable:
            n_skipped_params += 1
            continue

        subs = {}
        for s in free:
            esi_name = rubi_to_esi[s.name]
            subs[s] = esi_syms[esi_name]
        try:
            mapped_expr = expr.subs(subs)
        except:
            parse_failures += 1
            continue

        # Build param evaluation points
        pvals = {}
        for s in mapped_expr.free_symbols:
            if s == x:
                continue
            if s.name in _PARAM_POINTS:
                pvals[s] = _PARAM_POINTS[s.name]

        # Fingerprint with 5s timeout
        def _timeout_handler(signum, frame):
            raise TimeoutError()
        old = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(5)
        try:
            fp = fingerprint(mapped_expr, pvals)
        except:
            fp = None
            n_timeout += 1
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)

        if fp is None:
            fp_failures += 1
            continue
        n_fingerprinted += 1

        h = fp_to_hash(fp)
        if h and h in db:
            n_matched += 1
            prims = db[h]
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

    print(f"\n{'='*70}")
    print(f"RUBI vs ESI COMPARISON")
    print(f"{'='*70}")
    print(f"Total RUBI problems: {len(problems)}")
    print(f"  with var=x: {len(x_problems)}")
    print(f"  successfully parsed: {n_parsed}")
    print(f"  parse failures: {parse_failures}")
    print(f"  skipped (unmappable params): {n_skipped_params}")
    print(f"  fingerprinted: {n_fingerprinted}")
    print(f"  fingerprint failures: {fp_failures}")
    print(f"  timeout during fingerprint: {n_timeout}")
    print(f"  MATCHED in ESI: {n_matched}")
    print(f"  Match rate (of fingerprinted): {100*n_matched/max(n_fingerprinted,1):.1f}%")
    print(f"  Time: {elapsed:.0f}s")

    from collections import Counter
    print(f"\nMatches by RUBI category:")
    cats = Counter()
    for m in matches:
        cat = m['rubi_file'].split('/')[0] if '/' in m['rubi_file'] else m['rubi_file']
        cats[cat] += 1
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")

    print(f"\nSample matches (first 20):")
    for m in matches[:20]:
        print(f"  RUBI: ∫ {m['rubi_integrand'][:60]} dx  [{m['rubi_steps']} steps]")
        print(f"  ESI:  F(x) = {m['esi_primitive'][:60]} [k={m['esi_complexity']}]")
        print()

    hard_rubi = [m for m in matches if m['rubi_steps'] >= 5]
    print(f"Hard RUBI problems (≥5 steps) matched by ESI: {len(hard_rubi)}")
    for m in sorted(hard_rubi, key=lambda m: -m['rubi_steps'])[:15]:
        print(f"  [{m['rubi_steps']} steps] ∫ {m['rubi_integrand'][:50]} dx")
        print(f"    ESI: {m['esi_primitive'][:60]} [k={m['esi_complexity']}]")

    results = {
        'summary': {
            'total_rubi': len(problems),
            'var_x': len(x_problems),
            'parsed': n_parsed,
            'parse_failures': parse_failures,
            'skipped_params': n_skipped_params,
            'fingerprinted': n_fingerprinted,
            'fp_failures': fp_failures,
            'fp_timeouts': n_timeout,
            'matched': n_matched,
            'match_rate': n_matched / max(n_fingerprinted, 1),
        },
        'matches': matches,
    }
    out_path = os.path.join(data_dir, 'rubi_esi_comparison.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
