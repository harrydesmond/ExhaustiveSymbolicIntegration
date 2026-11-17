"""Maple engine for the verified CAS audit (cas_audit.py), added in Nov 2026 for the second MCS revision.

Reuses cas_audit.py unchanged (integrand construction, --strip, verification points, classification):
Maple's own derivative of its antiderivative must match the integrand at the sample points.

Usage: python3 cas_audit_maple.py forms.json out.jsonl [--strategies fast|full] [--timeout 600]
       [--workers 2] [--strip]
Maple command: env ESI_MAPLE (default ~/maple2026/bin/maple). --mem-mb M (default 1400, 0 = off) runs each Maple
call in its own systemd user scope with MemoryMax=M and no swap; a call killed there fails with memlimit=True.
(kernelopts(datalimit) does not bound all of Maple's memory: a kernel reached 2.8 GB under a 1.2 GB datalimit.)
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # cas_audit.py sits next to this file
import cas_audit as ca  # noqa: E402
import sympy as sp  # noqa: E402
from sympy.printing.maple import maple_code  # noqa: E402

MAPLE = ca._cmd('ESI_MAPLE', os.path.expanduser('~/maple2026/bin/maple'))

STRATS = {'fast': ['default'],
          'full': ['default', 'assume_pos', 'simplify', 'symbolic', 'risch', 'allmethods', 'roundtrip'],
          # extra rewrites, used to stress-test integrands that resist the full set
          'stress': ['normal', 'expand', 'combine', 'convert_exp', 'simplify_pos', 'simplify_allmethods']}


MAPLE_NAMES = {'re': 'Re', 'im': 'Im', 'arg': 'argument', 'sign': 'signum', 'atan2': 'arctan'}


def mpl(e):
    # SymPy's maple_code leaves these SymPy names as they are; Maple spells them differently (its sign() is the
    # sign of a leading coefficient, not the real sign function), and arctan(y, x) is Maple's atan2(y, x)
    s = re.sub(r'\b(re|im|arg|sign|atan2)\(', lambda mm: MAPLE_NAMES[mm.group(1)] + '(', maple_code(e))
    if any(t in s for t in ('zoo', 'nan', 'infinity')):
        raise ValueError('non-finite constant in expression')
    return s


def maple_script(fs, Fs, strat, pts):
    vals = ', '.join(f'{k} = {v}' for k, v in ca.PARAMS.items())
    ptl = ', '.join(pts)
    body = {'default': 'r := int(f, x):',
            'assume_pos': 'r := int(f, x) assuming x > 0:',
            'simplify': 'r := int(simplify(f), x):',
            'symbolic': 'r := int(simplify(f, symbolic), x):',
            'risch': 'r := int(f, x, method = risch):',
            'roundtrip': f'r := int(diff({Fs}, x), x):',
            'normal': 'r := int(normal(f), x):',
            'expand': 'r := int(expand(f), x):',
            'combine': 'r := int(combine(f), x):',
            'convert_exp': 'r := int(convert(f, exp), x):',
            'simplify_pos': 'g := simplify(f) assuming x > 0: r := int(g, x) assuming x > 0:',
            # every int method; keep the candidate with the smallest residual (verified again below)
            'allmethods': None, 'simplify_allmethods': None}[strat]
    if body is None:
        body = '\n'.join(([] if strat == 'allmethods' else ['f0 := f: f := simplify(f):']) + [
                'rv := int(f, x, method = _RETURNVERBOSE):',
                'r := Int(f, x): best := infinity:',
                'for c in rv do',
                '  if type(c, `=`) and lhs(c) <> FAILS and not has(rhs(c), {int, Int}) and not hastype(rhs(c), string) then',
                '    printf("CAND:%a\\n", c);',
                '    d0 := diff(rhs(c), x) - f:',
                f'    e0 := select(v -> type(v, numeric) and not type(v, {{undefined, infinity}}), [seq(ev(d0, p)/(1 + ev(f, p)), p in [{ptl}])]);',
                '    e0 := `if`(nops(e0) >= 3, max(e0), infinity);',
                '    if type(e0, numeric) and e0 < best then best := e0: r := rhs(c): end if:',
                '  end if:',
                'end do:',
                'printf("METHODS:%a\\n", map(c -> `if`(type(c, `=`), lhs(c), c), rv));']
            + ([] if strat == 'allmethods' else ['f := f0:']))
    return '\n'.join([
        'interface(screenwidth = infinity, prettyprint = 0, warnlevel = 0):',
        'Digits := 30:',
        # numerical check at one point, capped at 30 s (huge towers can stall evalf)
        f'ev := proc(e, p) try timelimit(30, evalf(abs(eval(e, [x = p, {vals}])))) '
        'catch: Float(undefined) end try end proc:',
        f'f := {fs}:',
        't0 := time[real]():',
        body,
        'printf("INTTIME:%a\\n", time[real]() - t0);',
        'printf("RESULT:%a\\n", r);',
        'if has(r, {int, Int}) then printf("UNEVAL:1\\n"); quit; end if;',
        'd := diff(r, x) - f:',
        f'for p in [{ptl}] do printf("PT:%a %a %a\\n", p, ev(f, p), ev(d, p)); end do;',
        'quit;', ''])


def run_maple(path, timeout, cwd, mem_mb):
    cmd = MAPLE + ['-q', '-s', path]
    if mem_mb:
        cmd = ['systemd-run', '--user', '--scope', '--quiet', '-p', f'MemoryMax={mem_mb}M',
               '-p', 'MemorySwapMax=0'] + cmd
    t0 = time.time()
    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, start_new_session=True, cwd=cwd)
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
    return out or '', to, time.time() - t0, p.returncode


def run_task(form, strat, timeout, strip, mem_mb=0):
    F, f = ca.build(form, strip)
    pts = [str(sp.Rational(p)) for p in (form.get('pts') or ca.DEFAULT_PTS)]
    with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR', '/tmp')) as td:
        path = os.path.join(td, 'job.mpl')
        open(path, 'w').write(maple_script(mpl(f), mpl(F), strat, pts))
        text, to, el, rc = run_maple(path, timeout, td, mem_mb)
    status, res, ptsout = ca.classify(text, to)
    return {'status': status, 'result': (res or '')[:4000], 'pts': ptsout, 'elapsed': round(el, 2),
            'timed_out': to, 'special': bool(res and ca.SPECIAL.search(res)),
            'tail': '' if status == 'solved' else text[-1500:],
            # Maple exits 0 after quit; a non-zero exit without a timeout means the kernel was killed (memory cap)
            'memlimit': bool(mem_mb) and not to and rc != 0, 'rc': rc,
            'int_time': next((float(l[8:]) for l in text.splitlines() if l.startswith('INTTIME:')), None),
            'cands': [l for l in text.splitlines() if l.startswith(('CAND:', 'METHODS:'))][:20]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('forms')
    ap.add_argument('out')
    ap.add_argument('--strategies', default='fast')
    ap.add_argument('--timeout', type=float, default=180)
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_ON_NODE', 2)))
    ap.add_argument('--strip', action='store_true')
    ap.add_argument('--mem-mb', type=int, default=1400)
    args = ap.parse_args()

    forms = json.load(open(args.forms))
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                r = json.loads(line)
                done.add((r['fid'], r['strategy']))
            except Exception:
                pass
    tasks = [(fm, s) for fm in forms for s in STRATS[args.strategies] if (fm['fid'], s) not in done]
    print(f'{len(forms)} forms, {len(tasks)} tasks, {args.workers} workers', flush=True)
    lock = Lock()
    fout = open(args.out, 'a')

    def work(t):
        fm, s = t
        try:
            r = run_task(fm, s, args.timeout, args.strip, args.mem_mb)
        except Exception as exc:
            r = {'status': 'harness_error', 'result': '', 'pts': [], 'elapsed': 0, 'timed_out': False,
                 'special': False, 'tail': repr(exc)[:1500]}
        r.update({'fid': fm['fid'], 'item_id': fm.get('item_id'), 'set': fm.get('set'),
                  'label': fm.get('label'), 'engine': 'maple', 'strategy': s})
        with lock:
            fout.write(json.dumps(r) + '\n')
            fout.flush()
            print(f"  {r['fid']} {s}: {r['status']}{' [MEMLIMIT]' if r.get('memlimit') else ''} ({r['elapsed']} s)", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, tasks))
    fout.close()
    print('finished', flush=True)


if __name__ == '__main__':
    main()
