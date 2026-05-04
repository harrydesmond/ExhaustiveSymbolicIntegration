#!/usr/bin/env python3
"""Check which RUBI failures (negative step count) ESI can solve."""
import warnings
warnings.filterwarnings('ignore')

import glob, json, pickle, hashlib, signal, sys
import numpy as np
import sympy
from sympy.parsing.mathematica import parse_mathematica

rng = np.random.RandomState(42)
N = 60
xpts = rng.uniform(0.2, 5.0, N)
param_pts = {f'a{i}': rng.uniform(0.5, 3.0, N) for i in range(10)}
x = sympy.Symbol('x', positive=True)
esi_syms = {f'a{i}': sympy.Symbol(f'a{i}', real=True) for i in range(10)}
rubi_to_esi = {'a':'a0','b':'a1','c':'a2','d':'a3','e':'a4',
               'f':'a5','g':'a6','n':'a7','p':'a8','m':'a9'}

def rewrite_trig(expr):
    for func, repl in [(sympy.sec, lambda a: 1/sympy.cos(a)),
                       (sympy.csc, lambda a: 1/sympy.sin(a)),
                       (sympy.cot, lambda a: sympy.cos(a)/sympy.sin(a)),
                       (sympy.coth, lambda a: sympy.cosh(a)/sympy.sinh(a)),
                       (sympy.sech, lambda a: 1/sympy.cosh(a)),
                       (sympy.csch, lambda a: 1/sympy.sinh(a))]:
        if expr.has(func):
            expr = expr.replace(func, repl)
    return expr

def fast_fp(expr):
    try:
        expr = rewrite_trig(expr)
    except:
        pass
    if not hasattr(expr, 'free_symbols'):
        expr = sympy.sympify(expr)
    free = expr.free_symbols
    syms, vals = [], []
    if x in free:
        syms.append(x); vals.append(xpts)
    for name, sym in esi_syms.items():
        if sym in free:
            syms.append(sym); vals.append(param_pts[name])
    if not syms:
        return None
    try:
        f = sympy.lambdify(syms, expr, modules=['numpy'])
        r = np.asarray(f(*vals), dtype=float)
        if np.isscalar(r):
            r = np.full(N, r)
        if np.sum(~np.isfinite(r)) > N * 0.3:
            return None
    except:
        return None
    rounded = []
    for v in r:
        if not np.isfinite(v):
            rounded.append('None')
        elif v == 0.0:
            rounded.append('0.0')
        else:
            rounded.append(f'{v:.10e}')
    return hashlib.md5('|'.join(rounded).encode()).hexdigest()

def split_fields(line):
    line = line.strip()
    if not line.startswith('{') or not line.endswith('}'):
        return None
    depth = 0; fields = []; current = []
    for ch in line[1:-1]:
        if ch in '([{': depth += 1
        elif ch in ')]}': depth -= 1
        elif ch == ',' and depth == 0:
            fields.append(''.join(current).strip()); current = []; continue
        current.append(ch)
    if current:
        fields.append(''.join(current).strip())
    return fields if len(fields) == 4 else None

# Build ESI float64 hash table
print('Building ESI float64 hash table...', flush=True)
esi_ns = {
    'x': x, 'Abs': lambda a: a,
    'inv': lambda a: 1/a, 'square': lambda a: a**2,
    'sqrt': lambda a: sympy.sqrt(a), 'log': lambda a: sympy.log(a),
    'pow': lambda a, b: sympy.Pow(a, b),
    'sin': sympy.sin, 'cos': sympy.cos, 'exp': sympy.exp,
    'zoo': sympy.zoo, 'pi': sympy.pi, 'E': sympy.E, 'sign': lambda a: 1,
    **esi_syms,
}

db = {}
with open('results_ext_log_maths_k8.pkl', 'rb') as f:
    data = pickle.load(f)
n_ok = 0
for i, (h_orig, c) in enumerate(data['clusters'].items()):
    if (i + 1) % 50000 == 0:
        print(f'  {i+1}/{len(data["clusters"])}...', flush=True)
    dstr = c['derivative_str']
    prims = [(k, eq) for k, eq, _ in c['primitives']]
    try:
        expr = sympy.sympify(dstr, locals=esi_ns)
        if not hasattr(expr, 'free_symbols'):
            expr = sympy.sympify(expr)
        h = fast_fp(expr)
        if h:
            n_ok += 1
            if h not in db:
                db[h] = prims
            else:
                db[h].extend(prims)
    except:
        pass

print(f'ESI: {n_ok} fingerprinted, {len(db)} unique hashes', flush=True)

# Also add trig database
with open('results_trig_maths_k8.pkl', 'rb') as f:
    tdata = pickle.load(f)
n_new = 0
for i, (h_orig, c) in enumerate(tdata['clusters'].items()):
    if (i + 1) % 50000 == 0:
        print(f'  trig {i+1}/{len(tdata["clusters"])}...', flush=True)
    dstr = c['derivative_str']
    prims = [(k, eq) for k, eq, _ in c['primitives']]
    try:
        expr = sympy.sympify(dstr, locals=esi_ns)
        if not hasattr(expr, 'free_symbols'):
            expr = sympy.sympify(expr)
        h = fast_fp(expr)
        if h:
            if h not in db:
                db[h] = prims
                n_new += 1
            else:
                db[h].extend(prims)
    except:
        pass
print(f'Added {n_new} trig hashes (total {len(db)})', flush=True)

# Parse RUBI failures
print('\nParsing RUBI failures...', flush=True)
rubi_dir = '/home/harry/Symbolic_regression/ESI/rubi_test'
files = sorted(glob.glob(rubi_dir + '/**/*.m', recursive=True))

failures = []
for fpath in files:
    cat = fpath.replace(rubi_dir + '/', '')
    with open(fpath, 'r', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('{') or line.startswith('(*'):
                continue
            if '(*' in line:
                line = line[:line.index('(*')]
            line = line.strip().rstrip(',')
            fields = split_fields(line)
            if fields:
                try:
                    s = int(fields[2])
                    if s < 0:
                        failures.append({'integrand': fields[0], 'var': fields[1],
                                        'steps': s, 'antideriv': fields[3], 'file': cat})
                except:
                    pass

print(f'RUBI failures: {len(failures)}', flush=True)

# Check each
parseable = 0
matched = []
for prob in failures:
    if prob['var'] != 'x':
        continue
    try:
        expr = parse_mathematica(prob['integrand'])
        if expr is None or expr.has(sympy.I):
            continue
        x_sym = sympy.Symbol('x')
        expr = expr.subs(x_sym, x)
        free = expr.free_symbols - {x}
        if any(s.name not in rubi_to_esi for s in free):
            continue
        parseable += 1
        subs = {s: esi_syms[rubi_to_esi[s.name]] for s in free}
        mapped = expr.subs(subs)

        old = signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TimeoutError()))
        signal.alarm(5)
        try:
            h = fast_fp(mapped)
        except:
            h = None
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, signal.SIG_DFL)

        if h and h in db:
            best = min(db[h], key=lambda t: (t[0], len(t[1])))
            matched.append({
                'rubi': prob['integrand'][:80],
                'steps': prob['steps'],
                'rubi_ans': prob['antideriv'][:80],
                'esi': best[1],
                'esi_k': best[0],
                'file': prob['file'],
            })
    except:
        pass

print(f'Parseable with mappable params: {parseable}', flush=True)
print(f'\nRUBI failures matched by ESI: {len(matched)}', flush=True)
for m in matched:
    print(f'  [{m["steps"]}] int {m["rubi"][:55]} dx')
    print(f'    RUBI optimal: {m["rubi_ans"][:55]}')
    print(f'    ESI: {m["esi"][:55]} [k={m["esi_k"]}]')
    print()
