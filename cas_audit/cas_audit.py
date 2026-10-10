"""Strict CAS audit of ESI integrands (MCS revision, Oct 2026).

Every claimed solve is verified: the engine's own derivative of its returned antiderivative must match
the integrand at sample points (relative residual < TOL), and the result must contain no unevaluated
integral, error or "implementation incomplete" message. Each (form, engine, strategy) task runs in its
own process group with a hard timeout; results are appended to a JSONL checkpoint so the job can be
requeued.

Input: JSON list of forms {fid, item_id, set, label, F (SymPy string of the antiderivative),
       real (bool: differentiate with x real, keeping Abs/sign), pts (list of x values)}.
       F may be given as an ESR string; with --strip, Abs/re/im/arg are stripped from F before
       differentiating (the analytic, Abs-free problem).

Usage: python cas_audit.py forms.json out.jsonl --engines giac,fricas,maxima [--strategies fast|full]
       [--timeout 180] [--workers N] [--mma-workers 4] [--strip] [--shard i/n]
"""
import argparse
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock, Semaphore

import sympy as sp

# Commands used to launch each system. Override with environment variables, e.g. to run inside a container:
#   ESI_MAXIMA="singularity exec /path/to/sage_sandbox /home/sage/sage/local/bin/maxima"
# (see env_glamdring.sh for the configuration used for the paper).
def _cmd(var, default):
    return shlex.split(os.environ.get(var, default))


GIAC = _cmd('ESI_GIAC', 'icas')
MAXIMA = _cmd('ESI_MAXIMA', 'maxima')
FRICAS = _cmd('ESI_FRICAS', 'fricas')
WOLFRAM = _cmd('ESI_WOLFRAMSCRIPT', 'wolframscript')

TOL = 1e-6
PARAMS = {'a0': '137/100', 'a1': '221/100', 'a2': '83/100', 'a3': '171/100', 'a4': '263/100'}
DEFAULT_PTS = ['31/100', '47/100', '73/100', '113/100', '161/100', '229/100', '307/100', '441/100']
SPECIAL = re.compile(r'(erf|Erf|Ei\b|ExpIntegral|expintegral|\bli\b|LogIntegral|gamma|Gamma|polylog|'
                     r'PolyLog|dilog|LambertW|lambert_w|ProductLog|Hypergeometric|hypergeometric|'
                     r'\bSi\b|\bCi\b|SinIntegral|CosIntegral|Fresnel|fresnel|Bessel|bessel|Elliptic|'
                     r'elliptic|MeijerG|meijerg)')

x = sp.Symbol('x')
xr = sp.Symbol('x', real=True)
ASYM = {f'a{i}': sp.Symbol(f'a{i}', positive=True) for i in range(5)}
NS = dict(ASYM)
NS.update({'x': x, 'pow': sp.Pow, 'Abs': sp.Abs, 'sqrt': sp.sqrt, 'exp': sp.exp, 'log': sp.log,
           'sin': sp.sin, 'cos': sp.cos, 're': sp.re, 'im': sp.im, 'arg': sp.arg, 'atan2': sp.atan2})


# ----------------------------------------------------------------------------------------------
# Integrand construction and syntax conversion
# ----------------------------------------------------------------------------------------------

def strip_F(F):
    """Analytic, Abs-free version of an ESR antiderivative (valid on the positive real domain)."""
    F = F.replace(sp.atan2, lambda a, b: sp.S.Zero if a == 0 else sp.atan2(a, b))
    F = F.replace(sp.Abs, lambda u: u).replace(sp.re, lambda u: u)
    F = F.replace(sp.im, lambda u: sp.S.Zero).replace(sp.arg, lambda u: sp.S.Zero)
    return F


def build(form, strip):
    F = sp.sympify(form['F'], locals=NS)
    if strip:
        F = strip_F(F)
    if form.get('real'):
        F = F.subs(x, xr)
        f = sp.diff(F, xr).subs(xr, x)
        F = F.subs(xr, x)
    else:
        f = sp.diff(F, x)
    return F, f


def infix(e, flavour):
    s = sp.sstr(e, order='none')
    if any(t in s for t in ('zoo', 'nan', 'oo')):
        raise ValueError('non-finite constant in expression')
    s = s.replace('**', '^')
    if flavour == 'giac':
        s = re.sub(r'\bE\b', 'exp(1)', s)
        s = re.sub(r'\bI\b', 'i', s)
        s = s.replace('Abs(', 'abs(')
    elif flavour == 'maxima':
        s = re.sub(r'\bE\b', '%e', s)
        s = re.sub(r'\bI\b', '%i', s)
        s = re.sub(r'\bpi\b', '%pi', s)
        s = s.replace('Abs(', 'abs(').replace('sign(', 'signum(')
    elif flavour == 'fricas':
        s = re.sub(r'\bE\b', '%e', s)
        s = re.sub(r'\bI\b', '%i', s)
        s = re.sub(r'\bpi\b', '%pi', s)
        s = s.replace('Abs(', 'abs(')
    return s


def mma(e):
    s = sp.mathematica_code(e)
    s = s.replace('sign[', 'Sign[')
    if any(t in s for t in ('zoo', 'nan', 'ComplexInfinity')):
        raise ValueError('non-finite constant in expression')
    return s


# ----------------------------------------------------------------------------------------------
# Engine scripts. Each prints  RESULT:<antiderivative>, optional UNEVAL:1, and lines PT:<x> <|f|> <|D r - f|>
# ----------------------------------------------------------------------------------------------

def giac_script(fs, Fs, strat, pts):
    pv = ','.join(PARAMS.values())
    pk = ','.join(PARAMS.keys())
    body = {'default': f'r:=integrate(f,x);',
            'assume_pos': f'assume(x>0);r:=integrate(f,x);',
            'simplify': f'r:=integrate(simplify(f),x);',
            'roundtrip': f'r:=integrate(diff({Fs},x),x);',
            'risch': f'r:=risch(f,x);'}[strat]
    lines = ['threads:=1;'] + [f'assume({k}>0);' for k in PARAMS] + \
            [f'f:={fs};', body, 'print("RESULT:"+string(r));', 'd:=diff(r,x)-f;']
    for p in pts:
        sub = f'subst(%s,[x,{pk}],[{p},{pv}])'
        lines.append(f'print("PT:{p} "+string(abs(evalf({sub % "f"})))+" "+string(abs(evalf({sub % "d"}))));')
    return '\n'.join(lines) + '\n'


def maxima_script(fs, Fs, strat, pts):
    asm = 'assume(' + ','.join(f'{k}>0' for k in PARAMS) + ')$'
    body = {'default': 'r:integrate(f,x)$',
            'assume_pos': 'assume(x>0)$ r:integrate(f,x)$',
            'radcan': 'r:integrate(radcan(f),x)$',
            'risch': 'r:risch(f,x)$',
            'roundtrip': f'r:integrate(diff({Fs},x),x)$'}[strat]
    vals = ','.join(f'{k}={v}' for k, v in PARAMS.items())
    lines = ['display2d:false$', 'linel:1000000$', asm, f'f:{fs}$', body,
             'print("RESULT:",string(r))$',
             'if not freeof(nounify(integrate),r) then print("UNEVAL:1")$',
             'd:diff(r,x)-f$']
    for p in pts:
        lines.append(f'print("PT:",{p},cabs(float(rectform(ev(f,x={p},{vals})))),'
                     f'cabs(float(rectform(ev(d,x={p},{vals})))))$')
    return '\n'.join(lines) + '\n'


def fricas_script(fs, Fs, strat, pts, outfile):
    integ = {'default': 'integrate(f, x)',
             'normalize': 'integrate(normalize(f), x)',
             'roundtrip': f'integrate(D({Fs}, x), x)'}[strat]
    vals = ', '.join(f'{k} = {v}' for k, v in PARAMS.items())
    ptl = ', '.join(pts)
    return '\n'.join([
        ')set messages type off', ')set messages time off', ')set output algebra off',
        f'f := {fs}',
        f'rr := {integ}',
        'r1 := (rr case Expression(Integer) => rr::Expression(Integer); first(rr::List(Expression(Integer))))',
        f'tf : TextFile := open("{outfile}"::FileName, "output")',
        'writeLine!(tf, concat("RESULT:", unparse(r1::InputForm)))',
        'd := D(r1, x) - f',
        f'for p in [{ptl}] repeat writeLine!(tf, concat(["PT:", unparse(p::InputForm), " ", '
        f'unparse(abs(complexNumeric(eval(f, cons(x = p, [{vals}]))))::InputForm), " ", '
        f'unparse(abs(complexNumeric(eval(d, cons(x = p, [{vals}]))))::InputForm)]))',
        'close!(tf)', ')quit', ''])


def wl_script(fm, Fm, strat, pts, rubi, outfile):
    vals = ', '.join(f'{k} -> {v}' for k, v in PARAMS.items())
    asm = 'x > 0 && ' + ' && '.join(f'{k} > 0' for k in PARAMS)
    if rubi:
        call = {'default': 'Int[f, x]', 'simplify': 'Int[Simplify[f], x]'}[strat]
    else:
        call = {'default': 'Integrate[f, x]',
                'assume_pos': f'Integrate[f, x, Assumptions -> {asm}]',
                'assume_gt1': 'Integrate[f, x, Assumptions -> x > 1]',
                'assume_lt1': 'Integrate[f, x, Assumptions -> 0 < x < 1]',
                'noconds': 'Integrate[f, x, GenerateConditions -> False]',
                'simplify': 'Integrate[Simplify[f], x]',
                'fullsimplify': f'Integrate[FullSimplify[f, {asm}], x]',
                'powerexpand': 'Integrate[PowerExpand[f], x]',
                'together': 'Integrate[Together[f], x]',
                'expand': 'Integrate[Expand[f], x]',
                'roundtrip': f'Integrate[D[{Fm}, x], x]'}[strat]
    ptl = ', '.join(pts)
    return '\n'.join([
        'SetOptions[$Output, PageWidth -> Infinity];',
        'Needs["Rubi`"];' if rubi else '',
        f'f = {fm};',
        f'r = Quiet[{call}];',
        'r = r /. Star -> Times;' if rubi else '',
        f'str = OpenWrite["{outfile}"];',
        'WriteString[str, "RESULT:" <> ToString[r, InputForm] <> "\\n"];',
        'If[!FreeQ[r, Integrate | Int | Unintegrable | CannotIntegrate | $Aborted | $Failed], '
        'WriteString[str, "UNEVAL:1\\n"]];',
        'd = D[r, x] - f;',
        f'Do[WriteString[str, "PT:" <> ToString[p, InputForm] <> " " <> '
        f'ToString[N[Abs[f /. Join[{{x -> p}}, {{{vals}}}]], 20], InputForm] <> " " <> '
        f'ToString[N[Abs[d /. Join[{{x -> p}}, {{{vals}}}]], 20], InputForm] <> "\\n"], {{p, {{{ptl}}}}}];',
        'Close[str];', ''])


SYMPY_RUNNER = r'''
import json, sys, sympy as sp
from sympy.integrals.heurisch import heurisch
from sympy.integrals.risch import risch_integrate
x = sp.Symbol('x')
ns = {f'a{i}': sp.Symbol(f'a{i}', positive=True) for i in range(5)}; ns['x'] = x
f = sp.sympify(sys.argv[1], locals=ns)
strat = sys.argv[2]
if strat == 'default':
    r = sp.integrate(f, x)
elif strat == 'risch':
    r = risch_integrate(f, x)
elif strat == 'heurisch':
    r = heurisch(f, x)
    r = sp.Integral(f, x) if r is None else r
elif strat == 'manual':
    r = sp.integrate(f, x, manual=True)
elif strat == 'meijerg':
    r = sp.integrate(f, x, meijerg=True)
elif strat == 'simplify':
    r = sp.integrate(sp.simplify(f), x)
elif strat == 'positive':
    xp = sp.Symbol('x', positive=True)
    r = sp.integrate(f.subs(x, xp), xp).subs(xp, x)
print('RESULT:' + sp.sstr(r))
if r.has(sp.Integral) or r.has(sp.integrals.risch.NonElementaryIntegral):
    print('UNEVAL:1')
d = sp.diff(r, x) - f
vals = {ns[k]: sp.Rational(v) for k, v in json.loads(sys.argv[4]).items()}
for p in json.loads(sys.argv[3]):
    s = dict(vals); s[x] = sp.Rational(p)
    try:
        af = abs(complex(f.subs(s).evalf(30))); ad = abs(complex(d.subs(s).evalf(30)))
    except Exception:
        af = ad = float('nan')
    print(f'PT:{p} {af} {ad}')
'''

STRATS = {
    'fast': {'giac': ['default'], 'fricas': ['default'], 'maxima': ['default'], 'rubi': ['default'],
             'mma': ['default'], 'sympy': ['default']},
    'full': {'giac': ['default', 'assume_pos', 'simplify', 'roundtrip', 'risch'],
             'fricas': ['default', 'normalize', 'roundtrip'],
             'maxima': ['default', 'assume_pos', 'radcan', 'risch', 'roundtrip'],
             'rubi': ['default', 'simplify'],
             'mma': ['default', 'assume_pos', 'assume_gt1', 'assume_lt1', 'noconds', 'simplify',
                     'fullsimplify', 'powerexpand', 'together', 'expand', 'roundtrip'],
             'sympy': ['default', 'risch', 'heurisch', 'manual', 'meijerg', 'simplify', 'positive']},
}


# ----------------------------------------------------------------------------------------------
# Running and classification
# ----------------------------------------------------------------------------------------------

def run_cmd(cmd, timeout, stdin_file=None, cwd=None, env=None):
    t0 = time.time()
    fin = open(stdin_file) if stdin_file else subprocess.DEVNULL
    p = subprocess.Popen(cmd, stdin=fin, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         start_new_session=True, cwd=cwd, env=env)
    try:
        out, _ = p.communicate(timeout=timeout)
        to = False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, _ = p.communicate()
        to = True
    finally:
        if stdin_file:
            fin.close()
    return out or '', to, time.time() - t0


def _fricas_float(m):
    return int(m.group(1)) * 2.0 ** int(m.group(2))


def num(s):
    s = s.strip().rstrip(',')
    if 'float(' in s:  # FriCAS InputForm: float(mantissa, exponent, 2), possibly inside complex(re, im)
        vals = []
        for a, b in re.findall(r'float\((-?\d+),\s*(-?\d+),\s*2\)', s):
            try:
                vals.append(math.ldexp(int(a), int(b)))
            except OverflowError:
                vals.append(float('inf'))
        if s.startswith('complex(') and len(vals) == 2:
            return math.hypot(vals[0], vals[1])
        return abs(vals[0]) if len(vals) == 1 else float('nan')
    s = re.sub(r'`[0-9.]*', '', s).replace('*^', 'e')
    try:
        return float(s)
    except ValueError:
        try:
            return float(sp.sympify(s))
        except Exception:
            return float('nan')


def classify(text, timed_out):
    res = None
    uneval = False
    pts = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('RESULT:'):
            res = line[len('RESULT:'):].strip().strip('"')
        elif line.startswith('UNEVAL:1'):
            uneval = True
        elif line.startswith('PT:'):
            parts = line[3:].split()
            if len(parts) >= 3:
                pts.append((parts[0], num(parts[1]), num(parts[2])))
    low = text.lower()
    err = any(m in low for m in ('implementation incomplete', 'error detected', 'error:', 'error in',
                                 'traceback', 'is  positive, negative or zero', 'positive or negative'))
    if res is None:
        if timed_out:
            return 'timeout', res, pts
        return ('error' if err else 'noresult'), res, pts
    rlow = res.lower()
    # an error inside integrate leaves the result variable unbound, so it prints its own name
    if rlow in ('r1', 'r', 'rr'):
        return ('timeout' if timed_out else 'error'), res, pts
    if uneval or 'integrate(' in rlow or 'integral(' in rlow or '%iint' in rlow or 'int[' in rlow \
            or 'unintegrable' in rlow or 'cannotintegrate' in rlow:
        return 'unevaluated', res, pts
    good = [(p, af, ad) for p, af, ad in pts if af == af and ad == ad and af < 1e300]
    if len(good) < 3:
        return 'unverified', res, pts
    ok = [ad / (1 + af) < TOL for _, af, ad in good]
    if all(ok):
        return 'solved', res, pts
    if any(ok):
        return 'branch', res, pts
    return 'wrong', res, pts


def run_task(form, engine, strat, timeout, strip, mma_sem):
    F, f = build(form, strip)
    pts = form.get('pts') or DEFAULT_PTS
    with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR', '/tmp')) as td:
        outfile = os.path.join(td, 'out.txt')
        if engine == 'giac':
            path = os.path.join(td, 'job.cas')
            open(path, 'w').write(giac_script(infix(f, 'giac'), infix(F, 'giac'), strat, pts))
            text, to, el = run_cmd(GIAC + [path], timeout, cwd=td,
                                   env=dict(os.environ, OMP_NUM_THREADS='1'))
        elif engine == 'maxima':
            path = os.path.join(td, 'job.mac')
            open(path, 'w').write(maxima_script(infix(f, 'maxima'), infix(F, 'maxima'), strat, pts))
            text, to, el = run_cmd(MAXIMA + ['--very-quiet', '-b', path],
                                   timeout, cwd=td)
        elif engine == 'fricas':
            path = os.path.join(td, 'job.input')
            open(path, 'w').write(fricas_script(infix(f, 'fricas'), infix(F, 'fricas'), strat, pts, outfile))
            text, to, el = run_cmd(FRICAS + ['-nosman', '-nox', '-noclef'],
                                   timeout, stdin_file=path, cwd=td)
            if os.path.exists(outfile):
                text = text + '\n' + open(outfile).read()
        elif engine in ('mma', 'rubi'):
            path = os.path.join(td, 'job.wl')
            open(path, 'w').write(wl_script(mma(f), mma(F), strat, pts, engine == 'rubi', outfile))
            with mma_sem:
                text, to, el = run_cmd(WOLFRAM + ['-file', path], timeout, cwd=td)
            if os.path.exists(outfile):
                text = text + '\n' + open(outfile).read()
        elif engine == 'sympy':
            text, to, el = run_cmd([sys.executable, '-c', SYMPY_RUNNER, sp.sstr(f), strat, json.dumps(pts),
                                    json.dumps(PARAMS)], timeout, cwd=td)
        else:
            raise ValueError(engine)
    status, res, ptsout = classify(text, to)
    return {'status': status, 'result': (res or '')[:4000], 'pts': ptsout, 'elapsed': round(el, 2),
            'timed_out': to, 'special': bool(res and SPECIAL.search(res)),
            'tail': '' if status == 'solved' else text[-1500:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('forms')
    ap.add_argument('out')
    ap.add_argument('--engines', default='giac,fricas,maxima')
    ap.add_argument('--strategies', default='fast')
    ap.add_argument('--timeout', type=float, default=180)
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_ON_NODE', 2)))
    ap.add_argument('--mma-workers', type=int, default=4)
    ap.add_argument('--strip', action='store_true')
    ap.add_argument('--shard', default='0/1')
    args = ap.parse_args()

    forms = json.load(open(args.forms))
    si, sn = map(int, args.shard.split('/'))
    forms = [fm for i, fm in enumerate(forms) if i % sn == si]
    engines = args.engines.split(',')

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                r = json.loads(line)
                done.add((r['fid'], r['engine'], r['strategy']))
            except Exception:
                pass
    tasks = [(fm, e, s) for fm in forms for e in engines for s in STRATS[args.strategies][e]
             if (fm['fid'], e, s) not in done]
    # interleave engines so the slow ones do not cluster
    print(f'{len(forms)} forms, {len(tasks)} tasks to run ({len(done)} already done), '
          f'{args.workers} workers', flush=True)
    lock = Lock()
    mma_sem = Semaphore(args.mma_workers)
    fout = open(args.out, 'a')
    counter = {'n': 0}

    def work(t):
        fm, e, s = t
        try:
            r = run_task(fm, e, s, args.timeout, args.strip, mma_sem)
        except Exception as exc:
            r = {'status': 'harness_error', 'result': '', 'pts': [], 'elapsed': 0, 'timed_out': False,
                 'special': False, 'tail': repr(exc)[:1500]}
        r.update({'fid': fm['fid'], 'item_id': fm.get('item_id'), 'set': fm.get('set'),
                  'label': fm.get('label'), 'engine': e, 'strategy': s})
        with lock:
            fout.write(json.dumps(r) + '\n')
            fout.flush()
            counter['n'] += 1
            if counter['n'] % 50 == 0:
                print(f'  {counter["n"]}/{len(tasks)} done', flush=True)

    # Mathematica/RUBI tasks share the licence seats, so they get their own pool
    wl = [t for t in tasks if t[1] in ('mma', 'rubi')]
    other = [t for t in tasks if t[1] not in ('mma', 'rubi')]
    n_other = max(1, args.workers - (args.mma_workers if wl else 0))
    with ThreadPoolExecutor(max_workers=n_other) as ex1, \
            ThreadPoolExecutor(max_workers=max(1, args.mma_workers)) as ex2:
        futs = [ex1.submit(work, t) for t in other] + [ex2.submit(work, t) for t in wl]
        for fu in futs:
            fu.result()
    fout.close()
    print('finished', flush=True)


if __name__ == '__main__':
    main()
