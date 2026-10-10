"""Build the candidate list for the strict CAS re-audit (MCS revision, Oct 2026).

Candidates = integrands that Mathematica failed in the latest full-coverage run of each basis:
  core_maths       SymPy-hard (1,059) that Mathematica's stress test left unsolved (411)
  core_log_maths   all 144 SymPy-hard (never run through the full cascade)
  ext_maths        905 Mathematica failures from the no-Assumptions rerun
  ext_log_maths    502 integrands failed by both SymPy and Mathematica at 180 s (k <= 8)
  trig_maths       685 Mathematica failures from the rerun
  ext_log k9n      33 narrow k=9 Mathematica failures
  ext_log k9b      551 broad k=9 Mathematica failures
Each item records the antiderivative F_str (ESR string with explicit Abs).

Writes audit_items.json.

usage: python cas_audit_build.py [--screening-dir DIR] [--out audit_items.json] [--forms-out audit_forms_jobB.json]
DIR holds the screening outputs (Zenodo: results/ of version 1 plus cas_audit/inputs/screening/).
"""
import argparse
import glob
import json
import os
import pickle

ap = argparse.ArgumentParser()
ap.add_argument('--screening-dir', default='.')
ap.add_argument('--out', default='audit_items.json')
ap.add_argument('--forms-out', default='audit_forms_jobB.json')
ARGS = ap.parse_args()


def P(name):
    return os.path.join(ARGS.screening_dir, name)


items = []


def add(cset, basis, F_str, k=None, deriv=None):
    items.append({'set': cset, 'basis': basis, 'F_str': F_str, 'k': k, 'deriv_orig': deriv})


# core_maths: SymPy-hard that MMA stress left unsolved
r = json.load(open(P('stress_results_core_maths.json')))['results']
for e in r:
    if not e['mma_solved']:
        add('core_maths', 'core_maths', e['our_expr'], e['our_k'], e['derivative'])

# core_log_maths: all SymPy-hard
r = json.load(open(P('stress_results_core_log_maths.json')))['results']
for e in r:
    add('core_log_maths', 'core_log_maths', e['our_expr'], e['our_k'], e['derivative'])

# ext_maths: map no-Assumptions MMA failures (derivative only) back to primitives
d2F = {}
for e in json.load(open(P('stress_results_ext_maths.json')))['results']:
    d2F.setdefault(e['derivative'], (e['our_expr'], e['our_k']))
chunks = sorted(glob.glob(P('cas_mma_ext_maths_chunk_*.json'))) or sorted(glob.glob('audit_inputs/cas_mma_ext_maths_chunk_*.json'))
for fn in chunks:
    for e in json.load(open(fn))['failures']:
        d2F.setdefault(e['derivative'], (e['our_expr'], e['our_k']))
n_miss = 0
for part in (0, 1):
    for e in pickle.load(open(P(f'ext_maths_mma_rerun_part_{part}_result.pkl'), 'rb'))['failed']:
        if e['derivative'] in d2F:
            F, k = d2F[e['derivative']]
            add('ext_maths', 'ext_maths', F, k, e['derivative'])
        else:
            n_miss += 1
print('ext_maths derivatives without primitive:', n_miss)

# ext_log_maths k<=8: SymPy & MMA both-fail at 180 s
for fn in ('stress_test_all_remaining.json', 'stress_remaining_untested.json'):
    for e in json.load(open(P(fn))):
        add('ext_log_maths', 'ext_log_maths', e['our_expr'], e['our_k'], e['derivative'])

# trig_maths: MMA rerun failures
for part in (0, 1):
    for e in pickle.load(open(P(f'trig_maths_mma_rerun_part_{part}_result.pkl'), 'rb'))['failed']:
        add('trig_maths', 'trig_maths', e['our_expr'], e['our_k'], e['derivative'])

# ext_log k=9 narrow and broad
for F in json.load(open(P('k9_mma_failures_for_followup.json'))):
    add('ext_log_k9_narrow', 'ext_log_maths', F, 9)
kb = json.load(open(P('k9_broad_sage_results.json')))
for e in kb['sage_solved'] + kb['sage_failed']:
    add('ext_log_k9_broad', 'ext_log_maths', e['F_str'], 9)

# dedup on (basis, F_str) keeping the first set label
seen, out = set(), []
for it in items:
    key = (it['basis'], it['F_str'])
    if key in seen:
        continue
    seen.add(key)
    it['id'] = len(out)
    out.append(it)

from collections import Counter
print('items:', len(items), 'unique:', len(out))
print(Counter(it['set'] for it in out))
json.dump(out, open(ARGS.out, 'w'), indent=0)
# harness forms (one per item; Abs is stripped by cas_audit.py --strip)
forms = [{'fid': f"B{e['id']:04d}", 'item_id': e['id'], 'set': e['set'], 'label': e['basis'], 'F': e['F_str'],
          'real': False} for e in out]
json.dump(forms, open(ARGS.forms_out, 'w'))
