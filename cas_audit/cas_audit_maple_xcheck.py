"""Independent check of Maple's verified solves: central-difference derivative of Maple's antiderivative (evaluated
in Maple at 50 digits) against the SymPy integrand evaluated with mpmath, at the harness points.

Points where the integrand overflows double precision are skipped before Maple is called (towers such as
x^(e^{e^{x^2}}) are astronomically large there and stall high-precision evaluation). Each Maple call runs in its
own memory-capped scope and process group; each point is capped at 20 s and the call at 180 s, and on timeout the
whole process group is killed, so no kernel is left running.
Usage: nice -n 19 python3 cas_audit_maple_xcheck.py forms.json results.jsonl out.json
"""
import cmath
import json
import os
import signal
import subprocess
import sys
import tempfile

import mpmath as mp
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv, args = sys.argv[:1], sys.argv[1:]
import cas_audit_maple as m  # noqa: E402

TOL = 1e-6
mp.mp.dps = 40
CMATH = {'re': lambda z: complex(z).real, 'im': lambda z: complex(z).imag, 'arg': lambda z: cmath.phase(z),
         'csign': lambda z: 0 if z == 0 else z / abs(z), 'Abs': abs, 'sqrt': cmath.sqrt, 'exp': cmath.exp,
         'log': cmath.log, 'sin': cmath.sin, 'cos': cmath.cos, 'atan2': lambda y, x: cmath.phase(complex(x, y))}
# atan2's arguments can arrive as mpc with zero imaginary part
MPM = {'atan2': lambda y, x: mp.atan2(mp.re(y), mp.re(x)), 'csign': lambda z: 0 if z == 0 else z / abs(z)}


def integrand(form):
    F, f = m.ca.build(form, True)
    syms = {s.name: s for s in f.free_symbols}
    xs = syms['x']

    def _d(d):  # SymPy leaves d/dx re(g), im(g), arg(g) unevaluated; at real x they are re(g'), im(g'), im(g'/g)
        (g,) = d.expr.args
        gp = sp.diff(g, xs)
        return {sp.re: sp.re(gp), sp.im: sp.im(gp), sp.arg: sp.im(gp / g)}[type(d.expr)]
    f = f.replace(lambda e: isinstance(e, sp.Derivative) and isinstance(e.expr, (sp.re, sp.im, sp.arg)), _d)
    f = f.subs({syms[k]: sp.Rational(v) for k, v in m.ca.PARAMS.items() if k in syms})
    # the math printer turns sign(z) into copysign(1, z), which rejects complex z
    f = f.replace(sp.sign, lambda z: sp.Function('csign')(z))
    return xs, f


def run_maple(script, timeout=180):
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'x.mpl')
        open(path, 'w').write(script)
        cmd = ['systemd-run', '--user', '--scope', '--quiet', '-p', 'MemoryMax=1400M', '-p', 'MemorySwapMax=0'] \
            + m.MAPLE + ['-q', '-s', path]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             start_new_session=True, cwd=td)
        try:
            return p.communicate(timeout=timeout)[0]
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.communicate()
            return ''


forms = {f['fid']: f for f in json.load(open(args[0]))}
rows = [json.loads(l) for l in open(args[1])]
seen, todo = set(), []
for r in rows:  # one check per distinct (integrand, antiderivative)
    if r['status'] == 'solved' and (r['fid'], r['result']) not in seen:
        seen.add((r['fid'], r['result']))
        todo.append(r)
print(f'{len(todo)} distinct solves to check', flush=True)

out = []
def check(r):
    xs, f = integrand(forms[r['fid']])
    fc = sp.lambdify(xs, f, [CMATH, 'math'])
    fm = sp.lambdify(xs, f, [MPM, 'mpmath'])
    pts = []
    for p in (forms[r["fid"]].get("pts") or m.ca.DEFAULT_PTS):  # in-interval points for real pieces
        try:
            v = complex(fc(float(sp.Rational(p))))
            if cmath.isfinite(v) and abs(v) < 1e300:
                pts.append((p, mp.mpc(fm(mp.mpf(sp.Rational(p).p) / sp.Rational(p).q))))
        except (OverflowError, ValueError, ZeroDivisionError, TypeError):
            pass
    vals = ', '.join(f'{k} = {v}' for k, v in m.ca.PARAMS.items())
    lines = ['interface(screenwidth = infinity):', 'Digits := 50:', f'R := eval({r["result"]}, [{vals}]):',
             'h := 10^(-20):']
    for p, _ in pts:
        lines.append(f'try v := timelimit(20, evalf((eval(R, x = {p} + h) - eval(R, x = {p} - h))/(2*h))): '
                     f'printf("ND:{p} %.30e %.30e\\n", Re(v), Im(v)); catch: printf("ND:{p} nan nan\\n"); end try;')
    txt = run_maple('\n'.join(lines + ['quit;', ''])) if pts else ''
    fv = dict(pts)
    res = []
    for line in txt.splitlines():
        if line.startswith('ND:'):
            p, a, b = line[3:].split()
            try:
                nd = mp.mpc(mp.mpf(a), mp.mpf(b))
            except Exception:
                continue
            if mp.isfinite(nd.real) and mp.isfinite(nd.imag):
                res.append((p, float(abs(nd - fv[p]) / (1 + abs(fv[p])))))
    ok = len(res) >= 3 and all(e < TOL for _, e in res)
    return {'fid': r['fid'], 'strategy': r['strategy'], 'n_finite_f': len(pts), 'n_pts': len(res),
            'max_rel': max((e for _, e in res), default=None), 'ok': ok}


for r in todo:
    try:
        out.append(check(r))
    except Exception as exc:
        out.append({'fid': r['fid'], 'strategy': r['strategy'], 'n_pts': 0, 'max_rel': None, 'ok': False,
                    'note': repr(exc)[:200]})
    o = out[-1]
    print(o['fid'], o['strategy'], f"{o['n_pts']}/{o.get('n_finite_f')} pts", o['max_rel'],
          'OK' if o['ok'] else 'CHECK', o.get('note', ''), flush=True)
    json.dump(out, open(args[2], 'w'), indent=1)
print('finished', flush=True)
