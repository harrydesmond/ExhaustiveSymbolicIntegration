"""Summarise the verified CAS audit into cas_audit_summary.json (the numbers quoted in the paper).

usage: python cas_audit_report.py <cas_audit_dir> [out.json]
<cas_audit_dir> contains inputs/, outputs/, intermediate/ and logs/ as in the Zenodo deposit.

Screening-stage counts (integrands tested, and those SymPy failed) come from the original SymPy and
Mathematica sweeps, which stored only their failures; they are given here as constants with their sources.
"""
import collections
import json
import os
import re
import sys

D = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(D, 'cas_audit_summary.json')


def jl(name):
    return [json.loads(l) for l in open(os.path.join(D, 'outputs', name))]


def js(sub, name):
    return json.load(open(os.path.join(D, sub, name)))


def solved_by(rows, fids=None):
    s = collections.defaultdict(set)
    for r in rows:
        if r['status'] == 'solved' and (fids is None or r['fid'] in fids):
            s[r['fid']].add(r['engine'])
    return s


# ---------------- screening constants (sources: stress_results_*.json, cas_full_sympy_results.json,
# k9 SymPy chunk pickles; see the paper's Table 3)
TESTED = {'core_maths': 13993, 'core_log_maths': 5669, 'ext_maths': 117674, 'ext_log_maths': 49538,
          'trig_maths': 51027, 'ext_log_k9_subset': 138395}
SYMPY_FAIL = {'core_maths': 1059, 'core_log_maths': 144, 'ext_maths': 8183, 'ext_log_maths': 2431,
              'trig_maths': 9795, 'ext_log_k9_subset': 35863}


def basis_of(cset):
    return 'ext_log_k9_subset' if cset.startswith('ext_log_k9') else cset


# ---------------- main cascade (Job B, stage 2, stage 3)
formsB = {f['fid']: f for f in js('inputs', 'audit_forms_jobB.json')}
corelog_fail = {r['fid'] for r in jl('corelog_mma.jsonl') if r['status'] != 'solved'}
cand = {fid for fid, f in formsB.items() if f['set'] != 'core_log_maths' or fid in corelog_fail}
sB = solved_by(jl('jobB_results.jsonl'), cand)
after_gfm = {fid for fid in cand if not sB[fid]}
s2 = solved_by(jl('stage2_results.jsonl') + jl('stage2_fricas.jsonl'), after_gfm)
after_s2 = {fid for fid in after_gfm if not s2[fid]}
real3 = js('inputs', 'audit_forms_stage3_real.json')
s3 = solved_by(jl('stage3_results.jsonl') + jl('stage3_fricas.jsonl'))
artefacts = {f['item_id'] for f in real3 if s3[f['fid']]}
final = sorted(after_s2 - artefacts)

table = collections.OrderedDict()
for b in TESTED:
    row = {'tested': TESTED[b], 'fail_sympy': SYMPY_FAIL[b]}
    ids = [fid for fid in cand if basis_of(formsB[fid]['set']) == b]
    row['fail_sympy_and_mathematica'] = len(ids)
    row['fail_giac_fricas_maxima'] = sum(fid in after_gfm for fid in ids)
    row['fail_all_strategies_tilde_f'] = sum(fid in after_s2 for fid in ids)
    row['abs_artefacts_solved_in_real_form'] = sum(fid in artefacts for fid in ids)
    row['resist_all_six'] = sum(fid in final for fid in ids)
    table[b] = row
tot = {k: sum(r[k] for r in table.values()) for k in next(iter(table.values()))}

per_engine = collections.Counter(e for fid in cand for e in sB[fid])
unique = collections.Counter(next(iter(sB[fid])) for fid in cand if len(sB[fid]) == 1)
s2_engine = collections.Counter(e for fid in after_gfm for e in s2[fid])

# ---------------- Job A (the five integrals reported as CAS-resistant in the submitted version)
formsA = {f['fid']: f for f in js('inputs', 'audit_forms_jobA.json')}
sA = collections.defaultdict(set)
for r in jl('jobA_results.jsonl') + jl('jobA_fricas.jsonl'):
    if r['status'] == 'solved':
        sA[r['fid']].add(f"{r['engine']}:{r['strategy']}")
jobA = [{'fid': fid, 'item': f['item_id'], 'form': f['label'], 'F': f['F'], 'solved_by': sorted(sA[fid])}
        for fid, f in formsA.items()]

# ---------------- RUBI-suite integrals that RUBI fails
sR = solved_by(jl('rubi_results.jsonl') + jl('rubi_fricas.jsonl'))
rubi = [{'F': f['F'], 'solved_by': sorted(sR[f['fid']])} for f in js('inputs', 'audit_forms_rubi.json')]

# ---------------- sample check
formsS = {f['fid']: f for f in js('inputs', 'audit_forms_sample.json')}
stages = [('after_giac', 'sample_after_giac.json'), ('after_maxima', 'sample_after_maxima.json'),
          ('after_fricas', 'sample_after_fricas.json'), ('after_full_non_mathematica', 'sample_after_s2a.json'),
          ('after_full_protocol', 'sample_final_survivors.json')]
stage_ids = {k: {f['fid'] for f in js('intermediate', fn)} for k, fn in stages}
strata = collections.OrderedDict()
for fid, f in formsS.items():
    st = strata.setdefault(f['set'], {'population': int(re.search(r'pop=(\d+)', f['label']).group(1)),
                                      'sampled': 0, **{k: 0 for k, _ in stages}})
    st['sampled'] += 1
    for k, _ in stages:
        st[k] += fid in stage_ids[k]
reverify = open(os.path.join(D, 'logs', 'esi_reverify2-954457.out')).read()
verified_numeric = re.findall(r'^(S\d+) .*VERIFIED', reverify, re.M)
symbolic = ['S09279', 'S09746', 'S10341', 'S09807']  # Mathematica answer symbolically identical to F
realS = js('inputs', 'audit_forms_sample_real.json')
sRS = solved_by(jl('sample_real_results.jsonl') + jl('sample_real_fricas.jsonl'))
sample_artefacts = sorted({f['item_id'] for f in realS if sRS[f['fid']]})
final_s = {f['fid']: f['F'] for f in js('intermediate', 'sample_final_survivors.json')}
degenerate = sorted(fid for fid, F in final_s.items() if 'pow(0' in F)
resolved = set(verified_numeric) | set(symbolic) | set(sample_artefacts) | set(degenerate)
sample = {
    'strata': strata,
    'total_sampled': len(formsS),
    'final_unsolved_by_harness': len(final_s),
    'towers_mathematica_verified_arbitrary_precision': sorted(verified_numeric),
    'towers_mathematica_answer_identical_to_F': symbolic,
    'abs_artefacts_solved_in_real_form': sample_artefacts,
    'degenerate_constant_where_defined': degenerate,
    'unresolved': sorted(set(final_s) - resolved),
    'bound': 'no resistant integrand in any stratum; rule of three gives < 3/n per stratum at 95% confidence '
             '(< 0.3% for n = 1000; the core_maths Mathematica-only stratum was tested in full)',
}

summary = {
    'description': 'Verified CAS audit of ESI integrands (revision for Mathematics in Computer Science, Oct 2026)',
    'systems': {'SymPy': '1.14.0', 'Mathematica': '13.3.1', 'RUBI': '4.17.3.0', 'FriCAS': '1.3.12',
                'Maxima': '5.47.0 (SageMath 10.8)', 'Giac': '2.0.0 (conda-forge)'},
    'verification': 'antiderivative differentiated in the same system and compared with the integrand at 8 '
                    'points in (0.2,5) at fixed parameter values, relative tolerance 1e-6',
    'cascade_table': table, 'cascade_total': tot,
    'giac_fricas_maxima': {'solved_by_engine': dict(per_engine), 'solved_by_only_this_engine': dict(unique),
                           'candidates': len(cand), 'solved': len(cand) - len(after_gfm)},
    'full_protocol_on_survivors': {'input': len(after_gfm), 'solved': len(after_gfm) - len(after_s2),
                                   'solved_by_engine': dict(s2_engine)},
    'abs_artefacts': sorted(artefacts),
    'cas_resistant': [{'fid': fid, 'basis': formsB[fid]['label'], 'set': formsB[fid]['set'],
                       'F_catalogue': formsB[fid]['F']} for fid in final],
    'submitted_version_integrals': jobA,
    'rubi_suite_integrals_rubi_fails': rubi,
    'sample_check': sample,
}
json.dump(summary, open(OUT, 'w'), indent=1)
print(json.dumps({'cascade_total': tot, 'n_final': len(final), 'sample_unresolved': sample['unresolved']}, indent=1))
