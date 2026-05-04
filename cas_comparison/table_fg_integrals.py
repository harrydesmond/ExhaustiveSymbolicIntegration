"""
Build verification table of 17 f(x)^g(x) integrals that are CAS-blind
but trivially integrable by ESI (their antiderivative IS f^g by construction).

Loads results_ext_log_maths_k8.pkl, identifies pure f^g primitives at k<=6,
tests SymPy integration with 30s timeout, and writes LaTeX table.

Usage: python3 table_fg_integrals.py
"""
import pickle
import re
import time
import json
import multiprocessing as mp
import sympy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from esi_pipeline import parse_expr, x, _PARSE_NAMESPACE


# ── Helpers ──

def is_pure_fg(eq_str):
    """Check if eq_str is exactly pow(Abs(f(x)), g(x)) where both f,g depend on x."""
    stripped = eq_str.strip()
    if not stripped.startswith('pow(Abs('):
        return False
    depth = 0
    end = -1
    for i, ch in enumerate(stripped):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                end = i
                break
    if end != len(stripped) - 1:
        return False
    inner = stripped[4:-1]
    depth2 = 0
    comma_pos = None
    for j, ch in enumerate(inner):
        if ch == '(':
            depth2 += 1
        elif ch == ')':
            depth2 -= 1
        elif ch == ',' and depth2 == 0:
            comma_pos = j
            break
    if comma_pos is None:
        return False
    base_str = inner[:comma_pos].strip()
    exp_str = inner[comma_pos+1:].strip()
    base_has_x = bool(re.search(r'(?<![a-z])x(?![a-z])', base_str))
    exp_has_x = bool(re.search(r'(?<![a-z])x(?![a-z])', exp_str))
    return base_has_x and exp_has_x


def _try_integrate(deriv_str):
    """Worker: try SymPy integrate, return result dict."""
    import sympy as sp
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from esi_pipeline import parse_expr, x
    try:
        deriv = parse_expr(deriv_str)
        if deriv is None:
            return {'status': 'error', 'error': 'parse'}
        result = sp.integrate(deriv, x)
        if result.has(sp.Integral):
            return {'status': 'failed', 'result': str(result)[:300]}
        return {'status': 'success', 'result': str(result)[:300]}
    except Exception as e:
        return {'status': 'error', 'error': str(e)[:300]}


def try_sympy_integrate(deriv_str, timeout=30):
    """Run SymPy integration in a subprocess with timeout."""
    pool = mp.Pool(1, maxtasksperchild=1)
    try:
        ar = pool.apply_async(_try_integrate, (deriv_str,))
        return ar.get(timeout=timeout)
    except mp.TimeoutError:
        return {'status': 'timeout'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)[:300]}
    finally:
        pool.terminate()
        pool.join()


def expr_to_latex(expr_str):
    """Parse expression string and convert to LaTeX."""
    try:
        expr = parse_expr(expr_str)
        if expr is None:
            return expr_str.replace('_', r'\_')
        return sympy.latex(expr)
    except Exception:
        return expr_str.replace('_', r'\_')


# ── The 17 curated CAS-blind f^g integrals ──

SELECTED_17 = [
    # Base: |log(x)| with different exponents
    'pow(Abs(log(x)),(1/x))',         # g=1/x
    'pow(Abs(log(x)),(x**2))',        # g=x^2
    'pow(Abs(log(x)),(-1/x))',        # g=-1/x
    'pow(Abs(log(x)),(-log(x)))',     # g=-log(x)
    'pow(Abs(log(x)),(1/log(x)))',    # g=1/log(x)
    'pow(Abs(log(x)),exp(1/x))',      # g=exp(1/x)
    'pow(Abs(log(x)),exp(sqrt(x)))',  # g=exp(sqrt(x))
    'pow(Abs(log(x)),(pow(x,x)))',    # g=x^x

    # Base: |log(|log(x)|)| with different exponents
    'pow(Abs(log(Abs(log(x)))),x)',       # g=x
    'pow(Abs(log(Abs(log(x)))),(-x))',    # g=-x
    'pow(Abs(log(Abs(log(x)))),(1/x))',   # g=1/x
    'pow(Abs(log(Abs(log(x)))),log(x))',  # g=log(x)

    # Base: |log(|log(|log(x)|)|)|
    'pow(Abs(log(Abs(log(Abs(log(x)))))),x)',

    # Mixed/algebraic bases
    'pow(Abs(x + log(x)),x)',     # base = x + log(x)
    'pow(Abs(x - exp(x)),x)',     # base = x - exp(x)
    'pow(Abs(x - log(x)),x)',     # base = x - log(x)
    'pow(Abs(x**2 - x),x)',      # base = x^2 - x
]


# ── Main ──

def main():
    pkl_file = "results_ext_log_maths_k8.pkl"
    print(f"Loading {pkl_file}...")
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    clusters = data['clusters']
    print(f"Total clusters: {len(clusters)}")

    # Build lookup: eq_str -> cluster data
    eq_to_info = {}
    for h, cdata in clusters.items():
        for k, eq_str, expr_str in cdata['primitives']:
            if eq_str in SELECTED_17:
                eq_to_info[eq_str] = {
                    'k': k,
                    'derivative_str': cdata['derivative_str'],
                    'cluster_size': len(cdata['primitives']),
                }

    # Load or build SymPy cache
    cache_file = "fg_sympy_cache.json"
    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            cache = json.load(f)

    # Test SymPy integration for each selected integral
    print(f"\nVerifying {len(SELECTED_17)} f^g integrals with SymPy (30s timeout)...")
    table_data = []

    for i, eq_str in enumerate(SELECTED_17):
        info = eq_to_info.get(eq_str)
        if info is None:
            print(f"  WARNING: {eq_str} not found in data!")
            continue

        if eq_str in cache:
            result = cache[eq_str]
            tag = "(cached)"
        else:
            print(f"  [{i+1}/{len(SELECTED_17)}] Testing {eq_str[:60]}...", end=' ', flush=True)
            result = try_sympy_integrate(info['derivative_str'], timeout=30)
            cache[eq_str] = result
            tag = ""
            print(f"-> {result['status']} {tag}")

        table_data.append({
            'eq_str': eq_str,
            'k': info['k'],
            'derivative_str': info['derivative_str'],
            'sympy_status': result['status'],
            'cluster_size': info['cluster_size'],
        })

    # Save cache
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2)

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"VERIFICATION TABLE: {len(table_data)} f(x)^g(x) CAS-blind integrals")
    print(f"{'='*70}")

    all_blind = all(d['sympy_status'] != 'success' for d in table_data)
    print(f"All verified as CAS-blind: {all_blind}")

    ks = [d['k'] for d in table_data]
    print(f"Complexity range: k={min(ks)} to k={max(ks)}")

    statuses = {}
    for d in table_data:
        s = d['sympy_status']
        statuses[s] = statuses.get(s, 0) + 1
    print(f"SymPy failure modes: {statuses}")
    print()

    for i, d in enumerate(table_data):
        print(f"{i+1:2d}. [k={d['k']}] F(x) = {d['eq_str']}")
        print(f"    F'(x) = {d['derivative_str'][:150]}")
        print(f"    SymPy: {d['sympy_status']}  |  cluster size: {d['cluster_size']}")
        print()

    # ── LaTeX table ──
    # Hand-crafted LaTeX for clean formatting (auto-generated expressions are verbose)
    latex_F = [
        r'\left|\log x\right|^{1/x}',
        r'\left|\log x\right|^{x^{2}}',
        r'\left|\log x\right|^{-1/x}',
        r'\left|\log x\right|^{-\log x}',
        r'\left|\log x\right|^{1/\log x}',
        r'\left|\log x\right|^{e^{1/x}}',
        r'\left|\log x\right|^{e^{\sqrt{x}}}',
        r'\left|\log x\right|^{x^{x}}',
        r'\left|\log\!\left|\log x\right|\right|^{x}',
        r'\left|\log\!\left|\log x\right|\right|^{-x}',
        r'\left|\log\!\left|\log x\right|\right|^{1/x}',
        r'\left|\log\!\left|\log x\right|\right|^{\log x}',
        r'\left|\log\!\left|\log\!\left|\log x\right|\right|\right|^{x}',
        r'\left|x + \log x\right|^{x}',
        r'\left|x - e^{x}\right|^{x}',
        r'\left|x - \log x\right|^{x}',
        r'\left|x^{2} - x\right|^{x}',
    ]

    lines = []
    lines.append(r"\begin{table*}")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Representative $f(x)^{g(x)}$-type integrals that are CAS-blind (SymPy fails to")
    lines.append(r"  integrate the derivative) but trivially solvable by ESI, since $\int F'(x)\,dx = F(x)+C$")
    lines.append(r"  by construction. Each $F'(x)$ was tested with \texttt{sympy.integrate} (30\,s timeout).}")
    lines.append(r"  \label{tab:fg_integrals}")
    lines.append(r"  \small")
    lines.append(r"  \begin{tabular}{clcc}")
    lines.append(r"    \toprule")
    lines.append(r"    \# & $F(x)$ & $k$ & SymPy \\")
    lines.append(r"    \midrule")

    for i, d in enumerate(table_data):
        num = i + 1
        F_tex = latex_F[i]
        k = d['k']
        status = d['sympy_status']
        lines.append(f"    {num} & ${F_tex}$ & {k} & \\texttt{{{status}}} \\\\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table*}")

    tex_content = '\n'.join(lines)
    with open("table_fg_integrals.tex", 'w') as f:
        f.write(tex_content)
    print(f"Written to table_fg_integrals.tex")


if __name__ == '__main__':
    main()
