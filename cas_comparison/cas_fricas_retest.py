#!/usr/bin/env python3
"""Retest FriCAS failures by differentiating the known primitive in FriCAS,
then trying to integrate the result. This avoids SymPy Abs/sign artifacts."""

import json
import subprocess
import sys
import os
import time
import re

SINGULARITY = "/usr/local/shared/singularity/bin/singularity"
FRICAS_SIF = os.environ.get("FRICAS_SIF", "/mnt/extraspace/hdesmond/fricas_latest.sif")


def sympy_primitive_to_fricas(expr_str):
    """Convert a SymPy primitive expression to FriCAS syntax.
    Primitives are clean (no sign/arg/re/im) but may have Abs."""
    s = expr_str
    # Remove Abs() -> just argument (x>0, params>0)
    while 'Abs(' in s:
        idx = s.find('Abs(')
        start = idx + 4
        depth = 1
        i = start
        while i < len(s) and depth > 0:
            if s[i] == '(':
                depth += 1
            elif s[i] == ')':
                depth -= 1
            i += 1
        inner = s[start:i-1]
        s = s[:idx] + '(' + inner + ')' + s[i:]

    s = s.replace('**', '^')
    return s


def test_via_differentiation(primitive_fricas, idx, timeout=180):
    """Differentiate the primitive in FriCAS, then try to integrate the result."""
    fricas_commands = (
        f")set quit unprotected\n"
        f")set messages type off\n"
        f")set messages autoload off\n"
        f"prim := {primitive_fricas}\n"
        f"integrand := D(prim, x)\n"
        f"result := integrate(integrand, x)\n"
        f"result\n"
        f")quit\n"
    )

    t0 = time.time()
    try:
        proc = subprocess.run(
            [SINGULARITY, "exec", FRICAS_SIF, "fricas", "-nosman", "-nox", "-noclef"],
            input=fricas_commands,
            capture_output=True, text=True, timeout=timeout
        )
        elapsed = time.time() - t0
        output = proc.stdout

        solved = True
        if 'implementation incomplete' in output:
            solved = False
        if 'integral' in output and '%' not in output.split('integral')[0][-5:]:
            # Check it's not just part of the answer
            if 'integrate' in output.lower():
                solved = False
        if 'Cannot find' in output:
            solved = False
        if 'Error detected' in output:
            # Could still have a result after the error
            lines = output.strip().split('\n')
            has_result = False
            for line in lines:
                if line.strip().startswith('(') and ')' in line and any(c.isalpha() for c in line):
                    has_result = True
            if not has_result:
                solved = False

        return solved, output, elapsed

    except subprocess.TimeoutExpired:
        return False, "TIMEOUT", time.time() - t0
    except Exception as e:
        return False, f"ERROR: {e}", time.time() - t0


def main():
    # Load original results to find failures
    with open('cas_fricas_results.json') as f:
        orig = json.load(f)

    # Load the integrals
    with open('cas_blind_full_list.json') as f:
        integrals = json.load(f)

    # Find failures
    failures = []
    for r in orig['results']:
        output = r['fricas_output']
        is_error = (
            'Cannot find a definition' in output or
            'atan2' in r.get('integrand_fricas', '') or
            'implementation incomplete' in output or
            ('Error detected' in output)
        )
        if is_error:
            failures.append(r['index'])

    print(f"Retesting {len(failures)} FriCAS failures via differentiation approach")
    print(f"Indices: {failures}")

    results = []
    n_solved = 0

    for idx in failures:
        entry = integrals[idx]
        primitive = entry['our_expr']
        k = entry['our_k']

        prim_fricas = sympy_primitive_to_fricas(primitive)
        print(f"\n[{idx}] k={k}: {primitive[:60]}...")
        print(f"  FriCAS primitive: {prim_fricas[:60]}...")
        sys.stdout.flush()

        solved, output, elapsed = test_via_differentiation(prim_fricas, idx, timeout=180)

        result = {
            'index': idx,
            'our_expr': primitive,
            'our_k': k,
            'primitive_fricas': prim_fricas,
            'fricas_solved': solved,
            'fricas_output': output[:1000] if output else '',
            'elapsed': round(elapsed, 1)
        }
        results.append(result)

        if solved:
            n_solved += 1
            print(f"  ** SOLVED ** ({elapsed:.1f}s)")
            # Show last few lines
            for line in output.strip().split('\n')[-3:]:
                if line.strip():
                    print(f"    {line.strip()[:80]}")
        else:
            n_failed_type = "timeout" if output == "TIMEOUT" else "failed"
            print(f"  {n_failed_type} ({elapsed:.1f}s)")
            for line in output.strip().split('\n'):
                if 'incomplete' in line or 'Error' in line or 'Cannot' in line:
                    print(f"    {line.strip()[:80]}")
                    break

        sys.stdout.flush()

    print(f"\n{'='*60}")
    print(f"RETEST RESULTS ({len(failures)} integrals)")
    print(f"{'='*60}")
    print(f"Solved: {n_solved}/{len(failures)}")
    print(f"Still failing: {len(failures) - n_solved}")

    # Save
    with open('cas_fricas_retest.json', 'w') as f:
        json.dump({
            'total': len(failures),
            'solved': n_solved,
            'failed': len(failures) - n_solved,
            'results': results
        }, f, indent=2)


if __name__ == '__main__':
    main()
