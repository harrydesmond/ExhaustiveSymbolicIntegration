"""Fraction of catalogue functions, and of CAS-tested integrands, that contain absolute values (Abs)."""
import os
import pickle
import sys

D = sys.argv[1] if len(sys.argv) > 1 else '..'  # directory holding the lookup pickles
files = {'core_maths': 'results_core_maths_k10.pkl', 'core_log_maths': 'results_core_log_maths_k8.pkl',
         'ext_maths': 'results_ext_maths_k9.pkl', 'ext_log_maths (k<=8)': 'results_ext_log_maths_k8.pkl',
         'trig_maths': 'results_trig_maths_k9.pkl'}
files = {b: os.path.join(D, f) if os.path.exists(os.path.join(D, f)) else f for b, f in files.items()}
for b, fn in files.items():
    cl = pickle.load(open(fn, 'rb'))['clusters']
    allF = [p[2] for c in cl.values() for p in c['primitives']]
    reps = [min(c['primitives'], key=lambda t: t[0])[2] for c in cl.values() if len(c['primitives']) > 1]
    fa = sum('Abs' in F for F in allF) / len(allF)
    ra = sum('Abs' in F for F in reps) / len(reps)
    print(f'{b:22s} functions {len(allF):8d} with Abs {fa:.3f} | tested {len(reps):7d} with Abs {ra:.3f}', flush=True)
