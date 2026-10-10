"""Stratified random sample of integrands eliminated by the original SymPy/Mathematica sweeps.

For each basis, the multi-member equivalence classes that were CAS-tested (lowest-complexity antiderivative per
class, as in cas_full_sympy.py) split into:
  stratum M: SymPy failed, Mathematica solved in the original sweep (Mathematica inputs had Abs/sign stripped)
  stratum S: SymPy solved in the original sweep (mostly on the Abs-stripped 'clean' input)
Items already in the strict audit (audit_items.json) are excluded. Up to N per stratum are drawn with a fixed
seed; running them through cas_audit.py bounds how many CAS-resistant integrals the original sweeps could
have missed.

usage: python cas_audit_sample.py [--results-dir R] [--screening-dir S] [--items audit_items.json]
R holds the lookup pickles results_<basis>_k<k>.pkl; S the screening outputs (as for cas_audit_build.py).
The defaults reproduce the layout on Glamdring used for the paper. Writes audit_forms_sample.json.
"""
import argparse
import json
import os
import pickle
import random

ap = argparse.ArgumentParser()
ap.add_argument('--results-dir', default='..')
ap.add_argument('--screening-dir', default='.')
ap.add_argument('--items', default='audit_items.json')
ap.add_argument('--out', default='audit_forms_sample.json')
ARGS = ap.parse_args()


def RD(name):
    return os.path.join(ARGS.results_dir, name)


def SD(name):
    return os.path.join(ARGS.screening_dir, name)


N = 1000
rng = random.Random(20261009)
cand = {(e['basis'], e['F_str']) for e in json.load(open(ARGS.items))}


def reps(pkl):
    clusters = pickle.load(open(pkl, 'rb'))['clusters']
    out = []
    for c in clusters.values():
        p = c['primitives']
        if len(p) > 1:
            out.append(min(p, key=lambda t: t[0])[2])
    return out


def hard_list(fn):
    return json.load(open(fn))['results']


strata = {}

# core_maths k<=10
R = reps(RD('results_core_maths_k10.pkl'))
H = hard_list(SD('stress_results_core_maths.json'))
hs = {e['our_expr'] for e in H}
strata[('core_maths', 'M')] = [e['our_expr'] for e in H if e['mma_solved']]
strata[('core_maths', 'S')] = [F for F in R if F not in hs]
print('core_maths tested', len(R))

# core_log_maths k<=8 (all SymPy failures are already audited)
R = reps(RD('results_core_log_maths_k8.pkl'))
hs = {e['our_expr'] for e in hard_list(SD('stress_results_core_log_maths.json'))}
strata[('core_log_maths', 'S')] = [F for F in R if F not in hs]
print('core_log_maths tested', len(R))

# ext_maths k<=9
R = reps(RD('results_ext_maths_k9.pkl'))
H = hard_list(SD('stress_results_ext_maths.json'))
hs = {e['our_expr'] for e in H}
strata[('ext_maths', 'M')] = [e['our_expr'] for e in H]
strata[('ext_maths', 'S')] = [F for F in R if F not in hs]
print('ext_maths tested', len(R))

# ext_log_maths k<=8
R = reps(RD('results_ext_log_maths_k8.pkl') if os.path.exists(RD('results_ext_log_maths_k8.pkl')) else 'results_ext_log_maths_k8.pkl')
sf = {e['our_expr'] for e in json.load(open(SD('cas_full_sympy_results.json')))['failures']}
mf = {e['our_expr'] for e in json.load(open(SD('cas_full_mathematica_results_cleaned.json')))['failures']}
strata[('ext_log_maths', 'M')] = sorted(sf - mf)
strata[('ext_log_maths', 'S')] = [F for F in R if F not in sf]
print('ext_log_maths tested', len(R))

# trig_maths k<=9
R = reps(RD('results_trig_maths_k9.pkl'))
H = hard_list(SD('stress_results_trig_maths.json'))
hs = {e['our_expr'] for e in H}
strata[('trig_maths', 'M')] = [e['our_expr'] for e in H]
strata[('trig_maths', 'S')] = [F for F in R if F not in hs]
print('trig_maths tested', len(R))

# ext_log k=9 broad Abs-free sqrt+exp search
allF = [l.strip() for l in open(RD('k9_absfree_sqrt_exp_untested.txt') if os.path.exists(RD('k9_absfree_sqrt_exp_untested.txt')) else SD('k9_absfree_sqrt_exp_untested.txt')) if l.strip()]
sfail, msolved = set(), []
for i in range(6):
    sfail |= {e['F_str'] for e in pickle.load(open(RD(f'k9_broad_sympy_chunk_{i:02d}.pkl') if os.path.exists(RD(f'k9_broad_sympy_chunk_{i:02d}.pkl')) else SD(f'k9_broad_sympy_chunk_{i:02d}.pkl'), 'rb'))['sympy_failures']}
    msolved += [e['F_str'] for e in pickle.load(open(RD(f'k9_broad_mma_chunk_{i:02d}.pkl') if os.path.exists(RD(f'k9_broad_mma_chunk_{i:02d}.pkl')) else SD(f'k9_broad_mma_chunk_{i:02d}.pkl'), 'rb'))['mma_solved']]
strata[('ext_log_k9_broad', 'M')] = msolved
strata[('ext_log_k9_broad', 'S')] = [F for F in allF if F not in sfail]
print('k9 broad candidates', len(allF), 'sympy failures', len(sfail))

forms = []
for (basis, st), pop in strata.items():
    b = 'ext_log_maths' if basis.startswith('ext_log') else basis
    pop = sorted({F for F in pop if (b, F) not in cand})
    pick = rng.sample(pop, min(N, len(pop)))
    print(f'{basis:18s} {st}: population {len(pop):7d}, sampled {len(pick)}')
    for F in pick:
        forms.append({'fid': f'S{len(forms):05d}', 'item_id': None, 'set': f'sample_{basis}_{st}',
                      'label': f'{basis}:{st}:pop={len(pop)}', 'F': F, 'real': False})
json.dump(forms, open(ARGS.out, 'w'))
print('total sampled', len(forms))
