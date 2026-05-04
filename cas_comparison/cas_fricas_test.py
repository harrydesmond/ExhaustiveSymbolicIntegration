#!/usr/bin/env python3
"""Test CAS-blind integrals with FriCAS via Singularity container.

For each integral in cas_blind_full_list.json, we:
1. Clean the SymPy integrand (strip Abs/sign, assume x>0, params>0)
2. Convert to FriCAS InputForm syntax
3. Run FriCAS with a timeout
4. Check whether FriCAS returns a closed-form result

Usage:
    python3 cas_fricas_test.py [--timeout 180] [--start 0] [--end 187]
"""

import json
import subprocess
import sys
import os
import re
import time
import argparse
import tempfile

# Path to FriCAS via Singularity
SINGULARITY = "/usr/local/shared/singularity/bin/singularity"
FRICAS_SIF = os.environ.get("FRICAS_SIF", "/mnt/extraspace/hdesmond/fricas_latest.sif")


def sympy_to_fricas(expr_str):
    """Convert a SymPy expression string to FriCAS InputForm.

    Cleaning steps (valid since x > 0 and all params > 0):
    - Remove Abs() wrappers → just the argument
    - Remove sign() → 1
    - Remove re() → just the argument
    - Remove im() → 0
    - Remove arg() → 0
    - ** → ^
    """
    s = expr_str

    # Iteratively remove sign(...) -> 1
    for _ in range(20):
        new = _replace_func(s, 'sign', '1')
        if new == s:
            break
        s = new

    # Iteratively remove Abs(...) -> (...)
    for _ in range(20):
        new = _replace_func(s, 'Abs', None)
        if new == s:
            break
        s = new

    # Remove re(...) -> (...)
    for _ in range(10):
        new = _replace_func(s, 're', None)
        if new == s:
            break
        s = new

    # Remove im(...) -> 0
    for _ in range(10):
        new = _replace_func(s, 'im', '0')
        if new == s:
            break
        s = new

    # Remove arg(...) -> 0
    for _ in range(10):
        new = _replace_func(s, 'arg', '0')
        if new == s:
            break
        s = new

    # Python ** -> FriCAS ^
    s = s.replace('**', '^')

    # Clean up: *1 -> *, +0 -> , etc.
    s = re.sub(r'\b1\*', '', s)
    s = re.sub(r'\*1\b', '', s)
    s = re.sub(r'\+ 0\b', '', s)
    s = re.sub(r'- 0\b', '', s)

    return s


def _replace_func(s, funcname, replacement):
    """Replace first occurrence of funcname(...) with replacement or inner content."""
    # Need to handle cases where funcname might be part of a longer word
    # e.g. 'sign' shouldn't match 'assign'
    # Use word boundary check
    pattern_start = None
    search_str = funcname + '('
    idx = 0
    while idx < len(s):
        pos = s.find(search_str, idx)
        if pos == -1:
            return s  # not found
        # Check it's not part of a longer identifier
        if pos > 0 and (s[pos-1].isalnum() or s[pos-1] == '_'):
            idx = pos + 1
            continue
        pattern_start = pos
        break

    if pattern_start is None:
        return s

    start = pattern_start + len(search_str)  # after the opening (
    depth = 1
    i = start
    while i < len(s) and depth > 0:
        if s[i] == '(':
            depth += 1
        elif s[i] == ')':
            depth -= 1
        i += 1
    # i now points just past the closing )

    inner = s[start:i-1]

    if replacement is not None:
        return s[:pattern_start] + replacement + s[i:]
    else:
        return s[:pattern_start] + '(' + inner + ')' + s[i:]


def test_single_integral(integrand_fricas, idx, timeout=180):
    """Run FriCAS on a single integral, return (solved, output, elapsed)."""

    # FriCAS commands: set up, integrate, print marker, quit
    fricas_commands = (
        f")set quit unprotected\n"
        f")set messages type off\n"
        f")set messages autoload off\n"
        f"result := integrate({integrand_fricas}, x)\n"
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

        # Determine if FriCAS succeeded
        # FriCAS prints "integral" when it can't solve
        # It also uses "potentialIntegrate" for partial results
        solved = True
        out_lower = output.lower()
        if 'integral' in output and ('\\int' in output or 'integrate' in out_lower):
            solved = False
        if 'potentialintegrate' in out_lower:
            solved = False
        if 'error' in out_lower and 'cannot' in out_lower:
            solved = False
        # If output is very short, likely an error
        if len(output.strip()) < 20:
            solved = False

        return solved, output, elapsed

    except subprocess.TimeoutExpired:
        return False, "TIMEOUT", time.time() - t0
    except Exception as e:
        return False, f"ERROR: {e}", time.time() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeout', type=int, default=180,
                        help='Timeout per integral (seconds)')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--output', type=str, default='cas_fricas_results.json')
    args = parser.parse_args()

    # Check container exists
    if not os.path.exists(FRICAS_SIF):
        print(f"ERROR: FriCAS container not found at {FRICAS_SIF}")
        print("Pull it with: singularity pull docker://nilqed/fricas:latest")
        sys.exit(1)

    # Quick sanity check: can we run FriCAS?
    print("Sanity check: running FriCAS...")
    try:
        proc = subprocess.run(
            [SINGULARITY, "exec", FRICAS_SIF, "fricas", "-nosman", "-nox", "-noclef"],
            input=")quit\n", capture_output=True, text=True, timeout=30
        )
        if proc.returncode != 0:
            print(f"WARNING: FriCAS returned exit code {proc.returncode}")
            print(proc.stderr[:200])
        else:
            print("FriCAS OK")
    except Exception as e:
        print(f"ERROR: Cannot run FriCAS: {e}")
        sys.exit(1)

    # Load CAS-blind integrals
    with open('cas_blind_full_list.json') as f:
        integrals = json.load(f)

    end = args.end if args.end is not None else len(integrals)
    subset = integrals[args.start:end]

    print(f"\nTesting {len(subset)} CAS-blind integrals with FriCAS "
          f"(timeout={args.timeout}s)")
    print(f"Container: {FRICAS_SIF}")
    print(f"Range: [{args.start}, {end})")

    results = []
    n_solved = 0
    n_failed = 0
    n_timeout = 0

    for i, entry in enumerate(subset):
        idx = args.start + i
        raw = entry['derivative']
        primitive = entry['our_expr']
        k = entry['our_k']

        # Clean and convert to FriCAS syntax
        fricas_expr = sympy_to_fricas(raw)

        print(f"\n[{idx+1}/{end}] k={k}: {primitive[:60]}...")
        sys.stdout.flush()

        solved, output, elapsed = test_single_integral(
            fricas_expr, idx, args.timeout
        )

        result = {
            'index': idx,
            'our_expr': primitive,
            'our_k': k,
            'integrand_raw': raw[:300],
            'integrand_fricas': fricas_expr[:300],
            'fricas_solved': solved,
            'fricas_output': output[:1000] if output else '',
            'elapsed': round(elapsed, 1)
        }
        results.append(result)

        if solved:
            n_solved += 1
            print(f"  ** SOLVED ** ({elapsed:.1f}s)")
            # Print the FriCAS output for solved ones
            for line in output.strip().split('\n')[-5:]:
                if line.strip():
                    print(f"    {line.strip()}")
        elif output == "TIMEOUT":
            n_timeout += 1
            print(f"  timeout ({elapsed:.1f}s)")
        else:
            n_failed += 1
            print(f"  failed ({elapsed:.1f}s)")

        sys.stdout.flush()

        # Save periodically
        if (i + 1) % 5 == 0 or i == len(subset) - 1:
            summary = {
                'total': len(subset),
                'solved': n_solved,
                'failed': n_failed,
                'timeout': n_timeout,
                'timeout_setting': args.timeout,
                'results': results
            }
            with open(args.output, 'w') as f:
                json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"FRICAS RESULTS ({len(subset)} integrals, {args.timeout}s timeout)")
    print(f"{'='*60}")
    print(f"Solved:  {n_solved} ({100*n_solved/len(subset):.1f}%)")
    print(f"Failed:  {n_failed}")
    print(f"Timeout: {n_timeout}")
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
