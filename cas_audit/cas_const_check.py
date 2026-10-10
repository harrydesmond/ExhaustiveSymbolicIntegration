"""Lowest complexity at which selected constant-bearing functions appear in the large catalogues."""
import os
import pickle
import sys

D = sys.argv[1] if len(sys.argv) > 1 else '..'  # directory holding the lookup pickles
targets = {'6*x', '3*x**2', '7*x**2/2', '7*x', 'exp(3*x)/3', 'exp(3*x)', '7*x**2'}
for fn in [os.path.join(D, f) for f in ('results_core_maths_k10.pkl', 'results_ext_maths_k9.pkl', 'results_trig_maths_k9.pkl')]:
    cl = pickle.load(open(fn, 'rb'))['clusters']
    found = {}
    for c in cl.values():
        for k, eq, expr in c['primitives']:
            if expr in targets and (expr not in found or k < found[expr]):
                found[expr] = k
    print(fn, dict(sorted(found.items())), flush=True)
