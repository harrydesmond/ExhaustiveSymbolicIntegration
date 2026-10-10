"""Re-verify 'unverified' CAS answers for sample-check survivors with mpmath (no exponent overflow).

For each survivor, parse the Mathematica (InputForm) and SymPy answers, differentiate with SymPy and compare
with the integrand at six points in (0.2, 5) at 40 digits. One verified answer suffices.
usage: python cas_audit_reverify.py survivors.json results.jsonl [...]
"""
import json, sys, multiprocessing as mpc
import mpmath as mp, sympy as sp
from sympy.parsing.mathematica import parse_mathematica
import cas_audit as ca

PTS = ['0.21', '0.37', '0.73', '1.13', '1.61', '2.29']


def check_one(args):
    try:
        return _check_one(args)
    except Exception as exc:
        return args[0]['fid'], f'check failed: {type(exc).__name__}', ''


def _check_one(args):
    form, answers = args
    mp.mp.dps = 40
    x = ca.x
    vals = {ca.ASYM[f'a{i}']: sp.Rational(v) for i, v in enumerate(ca.PARAMS.values())}
    F, f = ca.build(form, True)
    fnum = sp.lambdify(x, f.subs(vals), 'mpmath')
    pts = [mp.mpf(p) for p in PTS]
    fvals = []
    for p in pts:
        try:
            fvals.append(fnum(p))
        except Exception:
            fvals.append(None)
    for eng, strat, rs in answers:
        try:
            if eng in ('mma', 'rubi'):
                expr = parse_mathematica(rs.replace('`', ''))
                expr = expr.subs({sp.Symbol(f'a{i}'): ca.ASYM[f'a{i}'] for i in range(5)}).subs(sp.Symbol('x'), x)
            else:
                expr = sp.sympify(rs, locals=ca.NS)
            if expr.has(sp.Integral):
                continue
            d = sp.lambdify(x, sp.diff(expr, x).subs(vals), 'mpmath')
            good = n = 0
            for p, fv in zip(pts, fvals):
                if fv is None or not mp.isfinite(fv):
                    continue
                n += 1
                if abs(d(p) - fv) / (1 + abs(fv)) < mp.mpf('1e-20'):
                    good += 1
            if n >= 3 and good == n:
                return form['fid'], f'VERIFIED by {eng}:{strat} at {n} points', rs[:120]
        except Exception:
            continue
    return form['fid'], f'not verified ({len(answers)} answers tried)', ''


if __name__ == '__main__':
    surv = [f for f in json.load(open(sys.argv[1])) if 'pow(0' not in f['F'] and 'core_log' not in f['set']]
    ids = {f['fid'] for f in surv}
    ans = {k: [] for k in ids}
    for fn in sys.argv[2:]:
        for l in open(fn):
            r = json.loads(l)
            if r['fid'] in ids and r['status'] in ('unverified', 'branch', 'wrong') and r['result']:
                ans[r['fid']].append((r['engine'], r['strategy'], r['result']))
    with mpc.Pool(min(len(surv), int(__import__('os').environ.get('SLURM_CPUS_ON_NODE', 4)))) as pool:
        jobs = {f['fid']: pool.apply_async(check_one, ((f, ans[f['fid']]),)) for f in surv}
        for f in surv:
            try:
                fid, verdict, rs = jobs[f['fid']].get(timeout=900)
            except mpc.TimeoutError:
                fid, verdict, rs = f['fid'], 'TIMEOUT in re-verification', ''
            print(fid, f['set'], f['F'][:50], '->', verdict, '|', rs, flush=True)
