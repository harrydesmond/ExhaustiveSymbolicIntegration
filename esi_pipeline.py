"""
Exhaustive Symbolic Integration (ESI) — Core Pipeline

P0 action items from the brainstorm panel:
  1. Build core pipeline: load → differentiate → canonicalise → numerical hash → cluster
  2. Canonicalise parameter names (a0, a1, ... in order of first appearance)

Implements numerical fingerprinting as the primary equivalence method,
with symbolic verification for representative pairs within each cluster.
"""

import os
import re
import warnings
import pickle
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import sympy
from mpmath import mp, mpf, fabs

# ---------------------------------------------------------------------------
# Symbol setup (mirrors ESR conventions in esr.fitting.sympy_symbols)
# ---------------------------------------------------------------------------

# x is always positive (ESR convention)
x = sympy.Symbol('x', positive=True)

# Parameters — we use up to a0..a9 (more than ESR's max at complexity 10)
_MAX_PARAMS = 10
_param_symbols = [sympy.Symbol(f'a{i}', real=True) for i in range(_MAX_PARAMS)]
a0, a1, a2, a3, a4, a5, a6, a7, a8, a9 = _param_symbols

# ESR operator definitions (from esr.fitting.sympy_symbols)
_a, _b = sympy.symbols('_a _b', real=True)
inv = sympy.Lambda(_a, 1 / _a)
square = sympy.Lambda(_a, _a * _a)
cube = sympy.Lambda(_a, _a * _a * _a)
sqrt = sympy.Lambda(_a, sympy.sqrt(sympy.Abs(_a, evaluate=False)))
log = sympy.Lambda(_a, sympy.log(sympy.Abs(_a, evaluate=False)))
pow = sympy.Lambda((_a, _b), sympy.Pow(sympy.Abs(_a, evaluate=False), _b))
sin_func = sympy.Lambda(_a, sympy.sin(_a))
cos_func = sympy.Lambda(_a, sympy.cos(_a))

# Namespace for eval/sympify of ESR equation strings
_PARSE_NAMESPACE = {
    "inv": inv, "square": square, "cube": cube,
    "pow": pow, "sqrt": sqrt, "log": log,
    "sin": sin_func, "cos": cos_func,
    "Abs": sympy.Abs,
    "x": x, "zoo": sympy.zoo,
    **{f'a{i}': _param_symbols[i] for i in range(_MAX_PARAMS)},
}


# ---------------------------------------------------------------------------
# 1. Loading expressions
# ---------------------------------------------------------------------------

def load_equations(data_dir, max_complexity=None):
    """Load unique equations from the ESR output directory.

    Deduplicates across complexity levels: if the same expression appears
    at multiple complexities, only the lowest-complexity version is kept.

    Args:
        data_dir: path to e.g. 'core_maths/'
        max_complexity: if set, only load up to this complexity

    Returns:
        list of (complexity, equation_string) tuples
    """
    data_dir = Path(data_dir)
    # Use dict keyed by equation string to deduplicate, keeping lowest complexity
    seen = {}

    for comp_dir in sorted(data_dir.glob("compl_*")):
        k = int(comp_dir.name.split("_")[1])
        if max_complexity is not None and k > max_complexity:
            continue

        unique_file = comp_dir / f"unique_equations_{k}.txt"
        if not unique_file.exists():
            # Fall back to all_equations if unique not available
            unique_file = comp_dir / f"all_equations_{k}.txt"
        if not unique_file.exists():
            continue

        with open(unique_file) as f:
            for line in f:
                eq_str = line.strip()
                if eq_str and (eq_str not in seen or seen[eq_str] > k):
                    seen[eq_str] = k

    equations = [(k, eq_str) for eq_str, k in sorted(seen.items(), key=lambda t: (t[1], t[0]))]
    print(f"Loaded {len(equations)} unique equations (complexity 1-{max_complexity or '?'})")
    return equations


def parse_expr(eq_str):
    """Parse an ESR equation string into a SymPy expression.

    Returns None if parsing fails.
    """
    try:
        # Use sympify with the ESR namespace
        expr = sympy.sympify(eq_str, locals=_PARSE_NAMESPACE)
        return expr
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 2. Parameter canonicalisation
# ---------------------------------------------------------------------------

def canonicalise_params(expr):
    """Rename parameters to a0, a1, ... in order of first appearance.

    This ensures that e.g. a1*x and a0*x are recognised as the same
    template after canonicalisation.

    Uses SymPy's tree traversal order (preorder) to define "first appearance".
    """
    if expr is None:
        return None

    # Find which parameter symbols appear in this expression
    present = [s for s in _param_symbols if s in expr.free_symbols]
    if not present:
        return expr

    # Sort by index to get a stable ordering, then remap
    # We need "order of first appearance" in the expression string
    # Use the string representation which is deterministic
    expr_str = str(expr)
    present_sorted = sorted(present, key=lambda s: _first_occurrence(expr_str, s.name))

    # Build substitution: map present params to a0, a1, a2, ...
    subs = {}
    # Use temporary symbols to avoid collision during substitution
    _temps = [sympy.Symbol(f'_tmp_{i}', real=True) for i in range(len(present_sorted))]
    for i, old in enumerate(present_sorted):
        subs[old] = _temps[i]
    expr = expr.subs(subs)
    # Now map temps to canonical names
    subs2 = {_temps[i]: _param_symbols[i] for i in range(len(_temps))}
    expr = expr.subs(subs2)
    return expr


def _first_occurrence(s, name):
    """Index of first occurrence of parameter name in string, with word boundary."""
    match = re.search(r'\b' + re.escape(name) + r'\b', s)
    return match.start() if match else len(s)


# ---------------------------------------------------------------------------
# 3. Differentiation
# ---------------------------------------------------------------------------

def differentiate(expr):
    """Differentiate expr w.r.t. x. Returns None on failure."""
    if expr is None:
        return None
    try:
        deriv = sympy.diff(expr, x)
        return deriv
    except Exception:
        return None


def try_simplify(expr, timeout_seconds=5):
    """Attempt simplification with a timeout.

    Returns the simplified expression, or the original if simplification
    fails or times out.
    """
    if expr is None:
        return None
    try:
        import signal

        def _handler(signum, frame):
            raise TimeoutError

        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(timeout_seconds)
        try:
            result = sympy.simplify(expr)
        except (TimeoutError, Exception):
            result = expr
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        return result
    except Exception:
        return expr


# ---------------------------------------------------------------------------
# 4. Numerical fingerprinting
# ---------------------------------------------------------------------------

# Fixed random evaluation points (deterministic seed for reproducibility)
_RNG = np.random.RandomState(42)
_N_EVAL_POINTS = 60

# Safe domains: x > 0 (ESR convention), parameters in moderate range
_X_POINTS = _RNG.uniform(0.2, 5.0, _N_EVAL_POINTS)
_PARAM_POINTS = {
    f'a{i}': _RNG.uniform(0.5, 3.0, _N_EVAL_POINTS)
    for i in range(_MAX_PARAMS)
}

# High precision for mpmath
_FINGERPRINT_DPS = 50  # 50 decimal digits


def numerical_fingerprint(expr, n_points=_N_EVAL_POINTS):
    """Compute a numerical fingerprint for an expression.

    Evaluates at fixed random points using mpmath high-precision arithmetic.
    Returns a tuple of rounded mpf values, or None if evaluation fails
    at too many points.

    The fingerprint is a tuple of values that can be hashed for clustering.
    """
    if expr is None:
        return None

    mp.dps = _FINGERPRINT_DPS
    params_in_expr = [s for s in _param_symbols if s in expr.free_symbols]
    has_x = x in expr.free_symbols

    values = []
    n_failed = 0

    for i in range(n_points):
        subs = {}
        if has_x:
            subs[x] = mpf(str(_X_POINTS[i]))
        for p in params_in_expr:
            subs[p] = mpf(str(_PARAM_POINTS[p.name][i]))

        try:
            val = expr.evalf(_FINGERPRINT_DPS, subs=subs)
            # Check for non-finite results
            if val.is_finite is False or val is sympy.zoo or val is sympy.nan:
                n_failed += 1
                values.append(None)
            elif val.is_number:
                # Round to fewer digits to allow for minor numerical differences
                rounded = float(val)
                if not np.isfinite(rounded):
                    n_failed += 1
                    values.append(None)
                else:
                    values.append(rounded)
            else:
                # Expression didn't fully evaluate (still symbolic)
                n_failed += 1
                values.append(None)
        except Exception:
            n_failed += 1
            values.append(None)

    # If too many evaluations fail, this expression is problematic
    if n_failed > n_points * 0.3:
        return None

    return tuple(values)


def fingerprint_to_hash(fp):
    """Convert a numerical fingerprint tuple to a hashable cluster key.

    Uses a tolerance-based binning: values are rounded to 10 significant
    figures before hashing. This groups expressions that agree numerically
    but may differ at the level of floating-point noise.
    """
    if fp is None:
        return None

    # Round each value to 10 significant figures
    rounded = []
    for v in fp:
        if v is None:
            rounded.append("None")
        elif v == 0.0:
            rounded.append("0.0")
        else:
            rounded.append(f"{v:.10e}")
    key = "|".join(rounded)
    return hashlib.md5(key.encode()).hexdigest()


# ---------------------------------------------------------------------------
# 5. Clustering into equivalence classes
# ---------------------------------------------------------------------------

class DerivativeCluster:
    """An equivalence class of primitives sharing the same derivative."""
    def __init__(self):
        self.primitives = []  # list of (complexity, expr_str, expr_sympy)
        self.derivative = None  # representative derivative (sympy expr)
        self.derivative_str = None

    def add(self, complexity, expr_str, expr_sympy, deriv_sympy):
        self.primitives.append((complexity, expr_str, expr_sympy))
        if self.derivative is None:
            self.derivative = deriv_sympy
            self.derivative_str = str(deriv_sympy)

    def size(self):
        return len(self.primitives)

    def min_complexity(self):
        return min(c for c, _, _ in self.primitives)

    def __repr__(self):
        return (f"DerivativeCluster(size={self.size()}, "
                f"deriv='{self.derivative_str}', "
                f"min_complexity={self.min_complexity()})")


def build_derivative_clusters(equations, simplify_timeout=5, verbose=True):
    """Main pipeline: load → differentiate → canonicalise → hash → cluster.

    Args:
        equations: list of (complexity, eq_string) from load_equations
        simplify_timeout: seconds to allow for sympy.simplify per expression
        verbose: print progress

    Returns:
        clusters: dict mapping hash → DerivativeCluster
        failures: list of (complexity, eq_str, reason)
    """
    clusters = defaultdict(DerivativeCluster)
    failures = []

    total = len(equations)
    for idx, (k, eq_str) in enumerate(equations):
        if verbose and (idx + 1) % 100 == 0:
            print(f"  Processing {idx+1}/{total}...")

        # Parse
        expr = parse_expr(eq_str)
        if expr is None:
            failures.append((k, eq_str, "parse_failed"))
            continue

        # Skip pure constants (no x dependence) — their derivative is 0
        if x not in expr.free_symbols:
            failures.append((k, eq_str, "no_x_dependence"))
            continue

        # Canonicalise parameter names in the PRIMITIVE first.
        # This ensures derivative param names are consistent with the
        # primitive, so cluster members have compatible param names.
        expr_canon = canonicalise_params(expr)
        if expr_canon is None:
            expr_canon = expr

        # Differentiate the canonicalised primitive
        deriv = differentiate(expr_canon)
        if deriv is None:
            failures.append((k, eq_str, "diff_failed"))
            continue

        # Simplify derivative (with timeout)
        deriv = try_simplify(deriv, timeout_seconds=simplify_timeout)

        # Numerical fingerprint of the derivative
        fp = numerical_fingerprint(deriv)
        if fp is None:
            failures.append((k, eq_str, "fingerprint_failed"))
            continue

        fp_hash = fingerprint_to_hash(fp)
        if fp_hash is None:
            failures.append((k, eq_str, "hash_failed"))
            continue

        clusters[fp_hash].add(k, eq_str, expr_canon, deriv)

    if verbose:
        n_clusters = len(clusters)
        n_processed = total - len(failures)
        sizes = [c.size() for c in clusters.values()]
        n_multi = sum(1 for s in sizes if s > 1)
        print(f"\nResults:")
        print(f"  Processed: {n_processed}/{total} expressions")
        print(f"  Failed: {len(failures)}")
        print(f"  Unique derivatives (clusters): {n_clusters}")
        print(f"  Clusters with >1 primitive: {n_multi}")
        if sizes:
            print(f"  Largest cluster: {max(sizes)}")
            print(f"  Mean cluster size: {np.mean(sizes):.2f}")

    return dict(clusters), failures


# ---------------------------------------------------------------------------
# 6. Symbolic verification of cluster members
# ---------------------------------------------------------------------------

def verify_cluster_symbolically(cluster, n_checks=3):
    """Verify that members of a cluster genuinely share the same derivative.

    Takes up to n_checks pairs and checks agreement numerically (primary)
    and symbolically (secondary). Numerical agreement is the ground truth,
    following the panel's recommendation.

    Returns:
        verified: bool (True if all checked pairs agree numerically)
        details: list of (pair_indices, result_str)
    """
    if cluster.size() < 2:
        return True, []

    details = []
    prims = cluster.primitives
    n = min(n_checks, cluster.size() - 1)
    all_ok = True

    for i in range(n):
        _, _, F1 = prims[0]
        _, _, F2 = prims[i + 1]
        try:
            d1 = sympy.diff(F1, x)
            d2 = sympy.diff(F2, x)

            # Primary check: numerical agreement
            fp1 = numerical_fingerprint(d1)
            fp2 = numerical_fingerprint(d2)
            if fp1 is not None and fp2 is not None:
                diffs = [abs(a - b) for a, b in zip(fp1, fp2)
                         if a is not None and b is not None]
                max_diff = max(diffs) if diffs else 0.0
                numerically_ok = max_diff < 1e-10
            else:
                max_diff = float('nan')
                numerically_ok = False

            # Secondary: try symbolic simplification
            try:
                diff_expr = sympy.simplify(d1 - d2)
                sym_zero = (diff_expr == 0)
            except Exception:
                diff_expr = "symbolic_failed"
                sym_zero = False

            ok = numerically_ok or sym_zero
            if not ok:
                all_ok = False
            details.append(((0, i + 1),
                            f"numerical_max_diff={max_diff:.2e}, "
                            f"symbolic_zero={sym_zero}, ok={ok}"))
        except Exception as e:
            all_ok = False
            details.append(((0, i + 1), f"Verification failed: {e}"))

    return all_ok, details


# ---------------------------------------------------------------------------
# 7. Analysis utilities
# ---------------------------------------------------------------------------

def compute_rho(clusters, equations, max_k):
    """Compute ρ(k): fraction of functions at complexity ≤ k whose derivative
    is also in F_≤k.

    Here ρ(k) = |{unique derivatives with complexity ≤ k}| / |{unique functions at complexity ≤ k}|

    More precisely: for each cluster, if the derivative can be expressed as
    a function in F_≤k, it counts. We approximate this by checking if the
    derivative's numerical fingerprint matches any function in the original
    set at complexity ≤ k.
    """
    # This requires comparing derivatives against the original function set.
    # We build fingerprints of the original functions and check for matches.
    pass  # Implemented in the analysis script


def complexity_gap_distribution(clusters):
    """Compute Δk = complexity(F') - complexity(F) for all expressions.

    Since F' may not be in the original enumerated set, we estimate its
    complexity using SymPy's count_ops as a proxy.

    Returns list of (complexity_F, estimated_complexity_F_prime, delta).
    """
    gaps = []
    for cluster in clusters.values():
        deriv = cluster.derivative
        if deriv is None:
            continue
        # Estimate derivative complexity via SymPy's count_ops
        try:
            deriv_ops = sympy.count_ops(deriv)
        except Exception:
            continue
        for comp_F, _, _ in cluster.primitives:
            gaps.append((comp_F, int(deriv_ops), int(deriv_ops) - comp_F))
    return gaps


def equivalence_class_sizes(clusters):
    """Return sorted list of cluster sizes."""
    return sorted([c.size() for c in clusters.values()], reverse=True)


# ---------------------------------------------------------------------------
# 8. Saving / loading results
# ---------------------------------------------------------------------------

def save_results(clusters, failures, filepath):
    """Save clusters and failures to a pickle file."""
    # Convert sympy expressions to strings for pickling robustness
    data = {
        'clusters': {},
        'failures': failures,
    }
    for h, c in clusters.items():
        data['clusters'][h] = {
            'primitives': [(k, s, str(e)) for k, s, e in c.primitives],
            'derivative_str': c.derivative_str,
        }
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved results to {filepath}")


def load_results(filepath):
    """Load clusters from a pickle file."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    clusters = {}
    for h, cdata in data['clusters'].items():
        c = DerivativeCluster()
        for k, s, e_str in cdata['primitives']:
            expr = parse_expr(e_str)
            c.add(k, s, expr, parse_expr(cdata['derivative_str']))
        clusters[h] = c

    return clusters, data['failures']


# ---------------------------------------------------------------------------
# Main: run the pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="ESI: Exhaustive Symbolic Integration pipeline")
    parser.add_argument("--data-dir", type=str,
                        default="core_maths",
                        help="Path to function set directory")
    parser.add_argument("--max-complexity", type=int, default=5,
                        help="Maximum complexity to process")
    parser.add_argument("--simplify-timeout", type=int, default=5,
                        help="Timeout (s) for SymPy simplification per expression")
    parser.add_argument("--output", type=str, default=None,
                        help="Output pickle file (default: results_k{max_complexity}.pkl)")
    parser.add_argument("--verify", action="store_true",
                        help="Run symbolic verification on multi-member clusters")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"results_k{args.max_complexity}.pkl"

    print(f"ESI Pipeline")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Max complexity: {args.max_complexity}")
    print(f"  Simplify timeout: {args.simplify_timeout}s")
    print()

    # Load
    equations = load_equations(args.data_dir, max_complexity=args.max_complexity)

    # Run pipeline
    t0 = time.time()
    clusters, failures = build_derivative_clusters(
        equations,
        simplify_timeout=args.simplify_timeout,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\nPipeline completed in {elapsed:.1f}s")

    # Summary statistics
    print(f"\n--- Summary ---")
    sizes = equivalence_class_sizes(clusters)
    print(f"Cluster size distribution (top 20): {sizes[:20]}")

    gaps = complexity_gap_distribution(clusters)
    if gaps:
        deltas = [d for _, _, d in gaps]
        print(f"Complexity gap Δk: mean={np.mean(deltas):.2f}, "
              f"median={np.median(deltas):.1f}, "
              f"min={min(deltas)}, max={max(deltas)}")

    # Optional symbolic verification
    if args.verify:
        print(f"\n--- Symbolic verification ---")
        multi_clusters = {h: c for h, c in clusters.items() if c.size() > 1}
        n_verified = 0
        n_failed = 0
        for h, c in multi_clusters.items():
            ok, details = verify_cluster_symbolically(c)
            if ok:
                n_verified += 1
            else:
                n_failed += 1
                print(f"  MISMATCH in cluster: {c}")
                for pair, msg in details:
                    print(f"    {pair}: {msg}")
        print(f"  Verified: {n_verified}/{len(multi_clusters)}, "
              f"Mismatches: {n_failed}")

    # Save
    save_results(clusters, failures, args.output)

    # Show some interesting clusters
    print(f"\n--- Interesting clusters (size > 1) ---")
    multi = sorted(
        [(h, c) for h, c in clusters.items() if c.size() > 1],
        key=lambda x: x[1].size(),
        reverse=True,
    )
    for h, c in multi[:10]:
        print(f"\n  Derivative: {c.derivative_str}")
        print(f"  Primitives ({c.size()}):")
        for comp, eq_str, _ in c.primitives:
            print(f"    [k={comp}] {eq_str}")
