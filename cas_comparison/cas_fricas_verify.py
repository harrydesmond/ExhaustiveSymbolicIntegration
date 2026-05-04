#!/usr/bin/env python3
"""
Verify FriCAS failures with atan2 fix and longer timeout (600s).

Takes the 17 FriCAS failures from cas_fricas_retest.json and retests them:
1. Fix atan2(y,x) -> atan(y/x) in FriCAS syntax
2. Use the "differentiation approach": take our known primitive, clean it,
   differentiate in FriCAS, then try to integrate the result
3. Also try direct integration of the cleaned integrand
4. 600s timeout (was 180s in the original test)

Uses the Singularity container at /mnt/extraspace/hdesmond/fricas_latest.sif

Usage:
  python3 cas_fricas_verify.py
  python3 cas_fricas_verify.py --timeout 600
"""

import json
import subprocess
import sys
import os
import time
import re
import argparse

SINGULARITY = "/usr/local/shared/singularity/bin/singularity"
FRICAS_SIF = os.environ.get("FRICAS_SIF", "/mnt/extraspace/hdesmond/fricas_latest.sif")

# Indices of the 17 FriCAS failures from cas_fricas_retest.json
FAILURE_INDICES = [12, 20, 22, 27, 32, 33, 66, 81, 86, 100, 118, 124, 125, 157, 158, 185, 186]


def _replace_func(s, funcname, replacement):
    """Replace first occurrence of funcname(...) with replacement or inner content."""
    search_str = funcname + '('
    idx = 0
    while idx < len(s):
        pos = s.find(search_str, idx)
        if pos == -1:
            return s
        # Check it's not part of a longer identifier
        if pos > 0 and (s[pos-1].isalnum() or s[pos-1] == '_'):
            idx = pos + 1
            continue
        # Found it
        start = pos + len(search_str)
        depth = 1
        i = start
        while i < len(s) and depth > 0:
            if s[i] == '(':
                depth += 1
            elif s[i] == ')':
                depth -= 1
            i += 1
        inner = s[start:i-1]
        if replacement is not None:
            return s[:pos] + replacement + s[i:]
        else:
            return s[:pos] + '(' + inner + ')' + s[i:]
    return s


def sympy_to_fricas(expr_str):
    """Convert a SymPy expression string to FriCAS InputForm.

    Cleaning steps (valid since x > 0, all params > 0):
    - Remove Abs() -> just the argument
    - Remove sign() -> 1
    - Remove re() -> just the argument
    - Remove im() -> 0
    - Remove arg() -> 0
    - atan2(y, x) -> atan(y/x)
    - ** -> ^
    """
    s = expr_str

    # Fix atan2(y, x) -> atan(y/x) BEFORE other replacements
    for _ in range(20):
        new = _fix_atan2(s)
        if new == s:
            break
        s = new

    # Remove sign() -> 1
    for _ in range(20):
        new = _replace_func(s, 'sign', '1')
        if new == s:
            break
        s = new

    # Remove Abs() -> (...)
    for _ in range(20):
        new = _replace_func(s, 'Abs', None)
        if new == s:
            break
        s = new

    # Remove re() -> (...)
    for _ in range(10):
        new = _replace_func(s, 're', None)
        if new == s:
            break
        s = new

    # Remove im() -> 0
    for _ in range(10):
        new = _replace_func(s, 'im', '0')
        if new == s:
            break
        s = new

    # Remove arg() -> 0
    for _ in range(10):
        new = _replace_func(s, 'arg', '0')
        if new == s:
            break
        s = new

    # Python ** -> FriCAS ^
    s = s.replace('**', '^')

    # Clean up multiplicative/additive identities
    s = re.sub(r'\b1\*', '', s)
    s = re.sub(r'\*1\b', '', s)
    s = re.sub(r'\+ 0\b', '', s)
    s = re.sub(r'- 0\b', '', s)

    return s


def _fix_atan2(s):
    """Replace first atan2(y, x) with atan((y)/(x)) in a string."""
    search = 'atan2('
    idx = 0
    while idx < len(s):
        pos = s.find(search, idx)
        if pos == -1:
            return s
        # Check not part of longer identifier
        if pos > 0 and (s[pos-1].isalnum() or s[pos-1] == '_'):
            idx = pos + 1
            continue
        # Find the matching closing paren and split on the comma
        start = pos + len(search)
        depth = 1
        i = start
        comma_pos = None
        while i < len(s) and depth > 0:
            if s[i] == '(':
                depth += 1
            elif s[i] == ')':
                depth -= 1
            elif s[i] == ',' and depth == 1:
                comma_pos = i
            i += 1
        if comma_pos is None:
            return s  # malformed, skip
        arg_y = s[start:comma_pos].strip()
        arg_x = s[comma_pos+1:i-1].strip()
        return s[:pos] + f'atan(({arg_y})/({arg_x}))' + s[i:]
    return s


def sympy_primitive_to_fricas(expr_str):
    """Convert a SymPy primitive expression to FriCAS syntax.
    Primitives are clean (no sign/arg/re/im) but may have Abs and atan2."""
    s = expr_str

    # Fix atan2 -> atan
    for _ in range(20):
        new = _fix_atan2(s)
        if new == s:
            break
        s = new

    # Remove Abs() -> just argument (x>0, params>0)
    while 'Abs(' in s:
        idx = s.find('Abs(')
        # Check not part of longer word
        if idx > 0 and (s[idx-1].isalnum() or s[idx-1] == '_'):
            # Skip this one - shouldn't happen for Abs but be safe
            break
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


def run_fricas(fricas_commands, timeout=600):
    """Run FriCAS commands via Singularity, return (solved, output, elapsed)."""
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
        if 'integral' in output.lower() and 'integrate' in output.lower():
            # Check if the word "integral" appears in a failure context
            # (not just as part of variable names)
            solved = False
        if 'Cannot find' in output:
            solved = False
        if 'Error detected' in output:
            # Could still have a result after the error
            lines = output.strip().split('\n')
            has_result_after_error = False
            error_seen = False
            for line in lines:
                if 'Error detected' in line:
                    error_seen = True
                if error_seen and line.strip().startswith('(') and ')' in line:
                    has_result_after_error = True
            if not has_result_after_error:
                solved = False
        if len(output.strip()) < 20:
            solved = False
        if output == "TIMEOUT":
            solved = False

        return solved, output, elapsed

    except subprocess.TimeoutExpired:
        return False, "TIMEOUT", time.time() - t0
    except Exception as e:
        return False, f"ERROR: {e}", time.time() - t0


def test_differentiation_approach(primitive_fricas, timeout=600):
    """Differentiate the known primitive in FriCAS, then try to integrate."""
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
    return run_fricas(fricas_commands, timeout)


def test_direct_integration(integrand_fricas, timeout=600):
    """Directly integrate the cleaned integrand in FriCAS."""
    fricas_commands = (
        f")set quit unprotected\n"
        f")set messages type off\n"
        f")set messages autoload off\n"
        f"result := integrate({integrand_fricas}, x)\n"
        f"result\n"
        f")quit\n"
    )
    return run_fricas(fricas_commands, timeout)


def main():
    parser = argparse.ArgumentParser(
        description='Verify FriCAS failures with atan2 fix and 600s timeout')
    parser.add_argument('--timeout', type=int, default=600,
                        help='Timeout per approach (default: 600s)')
    parser.add_argument('--indices', type=str, default=None,
                        help='Comma-separated indices to test (default: all 17 failures)')
    args = parser.parse_args()

    # Check container exists
    if not os.path.exists(FRICAS_SIF):
        print(f"ERROR: FriCAS container not found at {FRICAS_SIF}")
        print("Pull it with: singularity pull docker://nilqed/fricas:latest")
        sys.exit(1)

    # Sanity check FriCAS
    print("Sanity check: running FriCAS...")
    try:
        proc = subprocess.run(
            [SINGULARITY, "exec", FRICAS_SIF, "fricas", "-nosman", "-nox", "-noclef"],
            input=")quit\n", capture_output=True, text=True, timeout=30
        )
        if proc.returncode != 0:
            print(f"WARNING: FriCAS returned exit code {proc.returncode}")
        else:
            print("FriCAS OK")
    except Exception as e:
        print(f"ERROR: Cannot run FriCAS: {e}")
        sys.exit(1)

    # Load the CAS-blind integrals
    with open('cas_blind_full_list.json') as f:
        integrals = json.load(f)

    # Load the retest results to find failure indices
    if args.indices:
        indices_to_test = [int(x) for x in args.indices.split(',')]
    else:
        # Load from cas_fricas_retest.json
        try:
            with open('cas_fricas_retest.json') as f:
                retest = json.load(f)
            indices_to_test = [r['index'] for r in retest['results']
                               if not r['fricas_solved']]
        except FileNotFoundError:
            print("cas_fricas_retest.json not found, using hardcoded failure indices")
            indices_to_test = FAILURE_INDICES

    print(f"\nVerifying {len(indices_to_test)} FriCAS failures with atan2 fix")
    print(f"Timeout: {args.timeout}s per approach (differentiation + direct)")
    print(f"Indices: {indices_to_test}")
    print(f"Container: {FRICAS_SIF}")
    sys.stdout.flush()

    results = []
    n_solved = 0
    n_still_failing = 0

    for idx in indices_to_test:
        if idx >= len(integrals):
            print(f"\n[{idx}] SKIP: index out of range (only {len(integrals)} integrals)")
            continue

        entry = integrals[idx]
        primitive = entry['our_expr']
        derivative = entry['derivative']
        k = entry['our_k']

        print(f"\n[{idx}] k={k}: {primitive[:70]}...")
        sys.stdout.flush()

        result = {
            'index': idx,
            'our_expr': primitive,
            'our_k': k,
            'diff_approach': None,
            'direct_approach': None,
            'solved': False,
            'method': None,
        }

        # Approach 1: Differentiation approach (differentiate primitive, then integrate)
        prim_fricas = sympy_primitive_to_fricas(primitive)
        print(f"  Diff approach: prim = {prim_fricas[:60]}...")
        sys.stdout.flush()

        solved_diff, output_diff, elapsed_diff = test_differentiation_approach(
            prim_fricas, timeout=args.timeout)

        result['diff_approach'] = {
            'primitive_fricas': prim_fricas,
            'solved': solved_diff,
            'output': output_diff[:1000] if output_diff else '',
            'elapsed': round(elapsed_diff, 1),
        }

        if solved_diff:
            result['solved'] = True
            result['method'] = 'differentiation'
            n_solved += 1
            print(f"  ** SOLVED via differentiation ** ({elapsed_diff:.1f}s)")
            for line in output_diff.strip().split('\n')[-3:]:
                if line.strip():
                    print(f"    {line.strip()[:80]}")
        else:
            # Approach 2: Direct integration of cleaned integrand
            integrand_fricas = sympy_to_fricas(derivative)
            print(f"  Direct approach: integrand = {integrand_fricas[:60]}...")
            sys.stdout.flush()

            solved_direct, output_direct, elapsed_direct = test_direct_integration(
                integrand_fricas, timeout=args.timeout)

            result['direct_approach'] = {
                'integrand_fricas': integrand_fricas[:300],
                'solved': solved_direct,
                'output': output_direct[:1000] if output_direct else '',
                'elapsed': round(elapsed_direct, 1),
            }

            if solved_direct:
                result['solved'] = True
                result['method'] = 'direct'
                n_solved += 1
                print(f"  ** SOLVED via direct integration ** ({elapsed_direct:.1f}s)")
                for line in output_direct.strip().split('\n')[-3:]:
                    if line.strip():
                        print(f"    {line.strip()[:80]}")
            else:
                n_still_failing += 1
                diff_status = "timeout" if output_diff == "TIMEOUT" else "failed"
                direct_status = "timeout" if output_direct == "TIMEOUT" else "failed"
                print(f"  STILL FAILING: diff={diff_status} ({elapsed_diff:.1f}s), "
                      f"direct={direct_status} ({elapsed_direct:.1f}s)")

        results.append(result)
        sys.stdout.flush()

        # Save periodically
        if len(results) % 3 == 0 or idx == indices_to_test[-1]:
            summary = {
                'total': len(indices_to_test),
                'tested': len(results),
                'solved': n_solved,
                'still_failing': n_still_failing,
                'timeout_setting': args.timeout,
                'results': results,
            }
            with open('cas_fricas_verify_results.json', 'w') as f:
                json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"FRICAS VERIFICATION RESULTS ({len(indices_to_test)} integrals, {args.timeout}s timeout)")
    print(f"{'='*60}")
    print(f"Solved:         {n_solved}/{len(indices_to_test)}")
    print(f"Still failing:  {n_still_failing}/{len(indices_to_test)}")

    if n_solved > 0:
        solved_list = [r for r in results if r['solved']]
        print(f"\nSolved integrals:")
        for r in solved_list:
            print(f"  [{r['index']}] k={r['our_k']} via {r['method']}: {r['our_expr'][:60]}")

    if n_still_failing > 0:
        failing_list = [r for r in results if not r['solved']]
        print(f"\nGenuinely unsolvable by FriCAS:")
        for r in failing_list:
            print(f"  [{r['index']}] k={r['our_k']}: {r['our_expr'][:60]}")

    # Final save
    summary = {
        'total': len(indices_to_test),
        'tested': len(results),
        'solved': n_solved,
        'still_failing': n_still_failing,
        'timeout_setting': args.timeout,
        'results': results,
    }
    with open('cas_fricas_verify_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to cas_fricas_verify_results.json")


if __name__ == '__main__':
    main()
