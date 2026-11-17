"""Maple numbers quoted in the paper, from the deposit (run after cas_audit_report.py).

Maple enters at the full-protocol stage. Its run covered the 70 integrands of that stage before the
absolute-value stripping was corrected; the 61 of the corrected cascade are a subset whose stripped integrands are
unchanged, and only those are counted here.
Usage: python3 cas_audit_maple_report.py <cas_audit_dir> [out.json]   (default <dir>/maple_summary.json)
"""
import collections
import json
import os
import sys

D = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(D, 'maple_summary.json')
jl = lambda *p: [json.loads(l) for l in open(os.path.join(D, *p))]
js = lambda *p: json.load(open(os.path.join(D, *p)))
ENG6 = ['sympy', 'mma', 'maxima', 'rubi', 'giac', 'fricas']

summary = js('cas_audit_summary.json')
ids = sorted({f['fid'] for f in js('inputs', 'audit_forms_stage2.json')}
             & {f['fid'] for f in js('inputs', 'audit_forms_maple_stage2.json')})
assert len(ids) == summary['full_protocol_on_survivors']['input'], len(ids)
maple = jl('outputs', 'maple_stage2.jsonl')
s = collections.defaultdict(set)
for r in jl('outputs', 'stage2_results.jsonl') + jl('outputs', 'stage2_fricas.jsonl') + maple:
    if r['status'] == 'solved' and r['fid'] in ids:
        s[r['fid']].add(r['engine'])
resist6 = sorted(f for f in ids if not (s[f] - {'maple'}))
assert resist6 == [x['fid'] for x in summary['cas_resistant']], resist6
xc = [x for x in js('outputs', 'maple_xcheck.json') if x['fid'] in ids]
nine = {r['fid']: r for r in jl('outputs', 'maple_nine_stress.jsonl')}
out = {
    'system': 'Maple 2026.2 (one-month licence, single machine; full-protocol stage only)',
    'strategies': ['default', 'assume_pos', 'simplify', 'symbolic', 'risch', 'allmethods', 'roundtrip'],
    'full_protocol_integrands': len(ids),
    'solved_by_engine': {e: sum(e in s[f] for f in ids) for e in ENG6 + ['maple']},
    'union_six': sum(bool(s[f] - {'maple'}) for f in ids),
    'union_seven': sum(bool(s[f]) for f in ids),
    'solved_by_only_this_engine': {e: sorted(f for f in ids if s[f] == {e}) for e in ENG6 + ['maple']},
    'six_system_solves_missed_by_maple': sum(bool(s[f] - {'maple'}) and 'maple' not in s[f] for f in ids),
    'resist_six': resist6,
    'maple_solves_of_resist_six': sorted(f for f in resist6 if 'maple' in s[f]),
    'resist_all_seven': sorted(f for f in resist6 if 'maple' not in s[f]),
    'resist_all_seven_also_under_stress_rewrites': sorted(
        f for f in resist6 if 'maple' not in s[f]
        and not any(r['status'] == 'solved' for r in jl('outputs', 'maple_nine_stress.jsonl') if r['fid'] == f)),
    'maple_memlimit_or_timeout_on_unsolved': sorted({r['fid'] for r in maple if r['fid'] in ids
                                                     and (r.get('memlimit') or r['timed_out'])}
                                                    - {f for f in ids if 'maple' in s[f]}),
    'independent_check_of_maple_solves': {
        'distinct_solves': len(xc), 'integrands': len({x['fid'] for x in xc}), 'all_pass': all(x['ok'] for x in xc),
        'min_points': min(x['n_pts'] for x in xc), 'max_relative_difference': max(x['max_rel'] for x in xc)},
}
json.dump(out, open(OUT, 'w'), indent=1)
print(json.dumps({k: out[k] for k in ['solved_by_engine', 'union_seven', 'resist_all_seven']}, indent=1))
