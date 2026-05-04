#!/usr/bin/env python3
"""
ESI Integrate: look up antiderivatives from the ESI database.

Given an integrand f(x), computes its numerical fingerprint and checks
whether it matches any derivative in the ESI database. If found, returns
the lowest-complexity antiderivative F(x) such that F'(x) = f(x).

Usage:
  python3 esi_integrate.py "exp(x)/sqrt(log(x))"
  python3 esi_integrate.py "x**x*(log(x)+1)"
  python3 esi_integrate.py --interactive
  python3 esi_integrate.py --basis ext_log_maths --max-k 8 "1/x"

The first invocation loads the database (~5s), then lookups are instant.
"""

import argparse
import hashlib
import os
import pickle
import sys
import time

import numpy as np
import sympy
from mpmath import mp, mpf


# ── Evaluation points (must match esi_pipeline.py exactly) ──────────

_RNG = np.random.RandomState(42)
_N_EVAL = 60
_X_POINTS = _RNG.uniform(0.2, 5.0, _N_EVAL)
_PARAM_POINTS = {f'a{i}': _RNG.uniform(0.5, 3.0, _N_EVAL) for i in range(10)}
_DPS = 50

x = sympy.Symbol('x', positive=True)
_params = {f'a{i}': sympy.Symbol(f'a{i}', real=True) for i in range(10)}


# ── Fingerprinting ──────────────────────────────────────────────────

def fingerprint(expr):
    """Compute numerical fingerprint of a SymPy expression."""
    mp.dps = _DPS
    params_in = [s for s in _params.values() if s in expr.free_symbols]
    has_x = x in expr.free_symbols

    values = []
    n_failed = 0
    for i in range(_N_EVAL):
        subs = {}
        if has_x:
            subs[x] = mpf(str(_X_POINTS[i]))
        for p in params_in:
            subs[p] = mpf(str(_PARAM_POINTS[p.name][i]))
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
    """Convert fingerprint to MD5 hash."""
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


# ── Database ────────────────────────────────────────────────────────

class ESIDatabase:
    """Loads and queries the ESI derivative database."""

    def __init__(self):
        self.deriv_to_primitives = {}  # hash -> list of (complexity, eq_str)
        self.n_integrands = 0
        self.bases_loaded = []

    def load(self, pkl_path, verbose=False):
        """Load a results pkl file into the database."""
        t0 = time.time()
        basis = os.path.basename(pkl_path).replace('results_', '').replace('.pkl', '')
        if verbose:
            print(f"Loading {basis}...", end=' ', flush=True)

        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        if 'ok' in data:
            # raw_results format
            for r in data['ok']:
                h = r['deriv_hash']
                entry = (r['k'], r['expr_canon'])
                if h not in self.deriv_to_primitives:
                    self.deriv_to_primitives[h] = []
                self.deriv_to_primitives[h].append(entry)
        elif 'clusters' in data:
            # cluster results format
            for h, c in data['clusters'].items():
                prims = [(k, eq) for k, eq, _ in c['primitives']]
                if h not in self.deriv_to_primitives:
                    self.deriv_to_primitives[h] = prims
                else:
                    self.deriv_to_primitives[h].extend(prims)

        elapsed = time.time() - t0
        n_new = len(self.deriv_to_primitives) - self.n_integrands
        self.n_integrands = len(self.deriv_to_primitives)
        self.bases_loaded.append(basis)
        if verbose:
            print(f"{n_new} integrands ({elapsed:.1f}s)")

    def lookup(self, expr):
        """Look up an integrand. Returns list of (complexity, primitive_str) or None."""
        fp = fingerprint(expr)
        if fp is None:
            return None
        h = fp_to_hash(fp)
        if h is None:
            return None
        prims = self.deriv_to_primitives.get(h)
        if prims:
            # Sort by complexity, then by number of parameters (fewer = simpler),
            # then by string length
            return sorted(set(prims), key=lambda t: (t[0], t[1].count('a'), len(t[1])))
        return None


# ── Parsing ─────────────────────────────────────────────────────────

def parse_input(expr_str):
    """Parse user input into a SymPy expression."""
    ns = {
        'x': x, 'e': sympy.E, 'pi': sympy.pi,
        'sqrt': sympy.sqrt, 'log': sympy.log, 'ln': sympy.log,
        'exp': sympy.exp, 'sin': sympy.sin, 'cos': sympy.cos,
        'tan': sympy.tan, 'abs': sympy.Abs, 'Abs': sympy.Abs,
        **_params,
    }
    try:
        return sympy.sympify(expr_str, locals=ns)
    except Exception as e:
        print(f"Parse error: {e}")
        return None


# ── Main ────────────────────────────────────────────────────────────

def find_databases(data_dir='.'):
    """Find all results pkl files."""
    pkls = []
    for f in sorted(os.listdir(data_dir)):
        if f.startswith('results_') and f.endswith('.pkl'):
            pkls.append(os.path.join(data_dir, f))
    # Also check for raw_results
    for f in sorted(os.listdir(data_dir)):
        if f.startswith('raw_results_') and f.endswith('.pkl') and \
           f.replace('raw_', '') not in [os.path.basename(p) for p in pkls]:
            pkls.append(os.path.join(data_dir, f))
    return pkls


def do_lookup(db, expr_str, verbose=False):
    """Parse, fingerprint, look up, and display results."""
    expr = parse_input(expr_str)
    if expr is None:
        return

    if verbose:
        print(f"  Parsed: {expr}")
    result = db.lookup(expr)

    if result is None:
        if verbose:
            print("  Not found in ESI database.")
            print("  (The integrand may not be in the enumerated function space,")
            print("   or it may not have an antiderivative within the basis.)")
        else:
            print("Not found.")
    else:
        simplest_k, simplest_eq = result[0]
        if verbose:
            print(f"  Found {len(result)} primitive(s):")
            n_show = min(5, len(result))
            for i, (k, eq) in enumerate(result[:n_show]):
                print(f"    [k={k}] F(x) = {eq}")
            if len(result) > n_show:
                print(f"    ... and {len(result) - n_show} more")
            print(f"  Simplest: F(x) = {simplest_eq}  (k={simplest_k})")
        else:
            print(simplest_eq)


def main():
    parser = argparse.ArgumentParser(
        description="ESI Integrate: look up antiderivatives from the ESI database.")
    parser.add_argument('expression', nargs='?', default=None,
                        help='Integrand to look up, e.g. "1/x" or "exp(x)*x"')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Interactive mode: enter integrands one at a time')
    parser.add_argument('--data-dir', '-d', default='.',
                        help='Directory containing results pkl files')
    parser.add_argument('--basis', '-b', default=None,
                        help='Load only this basis (e.g. ext_log_maths)')
    parser.add_argument('--max-k', type=int, default=None,
                        help='Only reported; does not filter (all k in database are searched)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show all primitives, loading info, and diagnostics')
    args = parser.parse_args()

    # Find and load databases
    pkls = find_databases(args.data_dir)
    if args.basis:
        pkls = [p for p in pkls if args.basis in p]
    if not pkls:
        print(f"No database files found in {args.data_dir}")
        print("Run the ESI pipeline first, or specify --data-dir")
        sys.exit(1)

    db = ESIDatabase()
    for pkl in pkls:
        db.load(pkl, verbose=args.verbose)
    if args.verbose:
        print(f"Database ready: {db.n_integrands} integrands from {', '.join(db.bases_loaded)}")
        print()

    if args.expression:
        if args.verbose:
            print(f"∫ {args.expression} dx = ?")
        do_lookup(db, args.expression, verbose=args.verbose)

    elif args.interactive:
        print("Enter integrands (or 'q' to quit):")
        while True:
            try:
                expr_str = input("\n∫ ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if expr_str.lower() in ('q', 'quit', 'exit'):
                break
            if not expr_str:
                continue
            if args.verbose:
                print(f"  {expr_str} dx = ?")
            do_lookup(db, expr_str, verbose=args.verbose)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
