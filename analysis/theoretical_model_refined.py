"""
Refined theoretical model: WHY rho(k) peaks then declines.

The core insight from the data:
  rho(k) = n_closed(k) / |F_k|
  rho increases iff n_closed grows faster than |F_k|

For ext_log_maths:
  |F_k| grows at ~6.1x per k
  n_closed grows at 7.6x (k=4->5), 7.0x (k=5->6), 5.2x (k=6->7), 5.5x (k=7->8)

So the mechanism is:
  - At k=4-6: n_closed grows FASTER (~7x) than |F_k| (~6x) → rho rises
  - At k=7-8: n_closed grows SLOWER (~5.3x) than |F_k| (~6.1x) → rho falls

WHY does n_closed growth slow down? Because at low k, derivatives of new
functions tend to land on ALREADY-KNOWN simple targets (1, x, 1/x, exp(x),
log(x)), multiplying the closure count efficiently. But at high k, derivatives
become increasingly unique — they generate novel forms, many of which are NOT
in the enumerated space.

This can be modelled as:
  n_closed(k) ≈ |F_k| × P(delta<=0) × m(k)

where m(k) is the matching probability, which DECLINES because the derivative
space S_k grows as beta^k with beta > alpha (the function space growth rate).

From the data:
  alpha/beta ≈ 0.87 (ext_log_maths), 0.94 (ext_maths), 0.81 (core_maths)

So the asymptotic decline rate of rho(k) is (alpha/beta)^k ≈ 0.87^k for ext_log.
"""

import json
import numpy as np

def main():
    # Load all data
    with open('rho_results_k8.json') as f:
        core_raw = json.load(f)
    with open('rho_results_ext_maths_k8.json') as f:
        ext = json.load(f)
    with open('rho_results_ext_log_maths_k8.json') as f:
        elog = json.load(f)
    with open('rho_decomposition_ext_log_maths_k8.json') as f:
        decomp = json.load(f)

    print("=" * 70)
    print("THEORETICAL FRAMEWORK FOR rho(k) PEAK-AND-DECLINE")
    print("=" * 70)

    # ---- Part 1: The growth-rate crossover mechanism ----
    print("\n1. THE GROWTH-RATE CROSSOVER MECHANISM")
    print("-" * 50)
    print("""
rho(k) = n_closed(k) / |F_k|

d/dk [log rho(k)] = d/dk [log n_closed] - d/dk [log |F_k|]
                  = (growth rate of closures) - (growth rate of functions)

rho(k) increases when closures grow faster than the function space,
and decreases when the function space outpaces closures.
""")

    # Compute per-basis
    bases = {
        'ext_log_maths': {k: {'n': v['n_functions'], 'c': v['n_closed'], 'r': v['rho']}
                          for k, v in elog.items() if v['n_functions'] > 0},
        'ext_maths': {k: {'n': v['n_functions'], 'c': v['n_closed'], 'r': v['rho']}
                      for k, v in ext.items() if v['n_functions'] > 0},
        'core_maths': {k: {'n': core_raw['per_complexity'][k]['n_functions'],
                           'c': core_raw['per_complexity'][k]['n_closed'],
                           'r': core_raw['per_complexity'][k]['rho_at_k']}
                       for k in core_raw['per_complexity']
                       if core_raw['per_complexity'][k]['n_functions'] > 0},
    }

    for name, data in bases.items():
        ks = sorted(data.keys(), key=int)
        print(f"\n  {name}:")
        print(f"  {'k':>3} {'|F_k|':>9} {'n_closed':>9} {'rho':>8} "
              f"{'F_growth':>9} {'C_growth':>9} {'C/F':>6} {'trend':>6}")
        prev_n = prev_c = None
        for k in ks:
            d = data[k]
            fg = cg = ratio = trend = ""
            if prev_n and prev_n > 0 and prev_c and prev_c > 0:
                fg_val = d['n'] / prev_n
                cg_val = d['c'] / prev_c
                fg = f"{fg_val:.2f}x"
                cg = f"{cg_val:.2f}x"
                ratio = f"{cg_val/fg_val:.2f}"
                trend = "UP" if cg_val > fg_val else "DOWN"
            print(f"  {k:>3} {d['n']:9d} {d['c']:9d} {d['r']:8.4f} "
                  f"{fg:>9} {cg:>9} {ratio:>6} {trend:>6}")
            prev_n = d['n']
            prev_c = d['c']

    # ---- Part 2: Why does closure growth slow down? ----
    print("\n\n2. WHY CLOSURE GROWTH SLOWS: THE MATCHING PROBABILITY")
    print("-" * 50)

    dd = decomp['delta_distribution']['all']
    p_le0 = dd['frac_delta_le0']

    print(f"""
For a function F at complexity k, its derivative F' has complexity k' = k + delta
where delta has mean = {dd['mean_delta']:.2f}, and P(delta <= 0) = {p_le0:.3f}.

n_closed(k) = |F_k| × P(delta<=0) × m(k)

where m(k) = probability that a complexity-reducing derivative matches
an existing function in F_{{<=k}}.
""")

    # Compute m(k) for ext_log_maths
    data = bases['ext_log_maths']
    ks = sorted(data.keys(), key=int)
    print(f"  Matching probability m(k) for ext_log_maths:")
    for k in ks:
        d = data[k]
        if d['r'] > 0:
            m = d['r'] / p_le0
            print(f"    k={k}: m = {m:.4f} = {m:.1%}")

    print(f"""
m(k) peaks at k=6 ({bases['ext_log_maths']['6']['r']/p_le0:.1%}) then declines.

Physical interpretation: at moderate k, many derivatives land on "popular"
targets — simple functions like 1/x, exp(x), log(x) that exist at low k.
These act as "attractors" in derivative space. At high k, derivatives
become structurally exotic (nested compositions, mixed operators) and
increasingly miss the enumerated function space.
""")

    # ---- Part 3: The exponential decline model ----
    print("\n3. ASYMPTOTIC MODEL: EXPONENTIAL DECLINE")
    print("-" * 50)

    print(f"""
If the function space grows as |F_k| ~ alpha^k and the effective derivative
form space grows as S(k) ~ beta^k, then:

  m(k) ~ alpha^k / beta^k = (alpha/beta)^k

  rho(k) ~ P(delta<=0) × (alpha/beta)^k  [for large k]
""")

    results = {}
    for name, data in bases.items():
        ks_int = sorted(int(k) for k in data.keys())
        rhos = [data[str(k)]['r'] for k in ks_int]
        ns = [data[str(k)]['n'] for k in ks_int]

        # Growth rate
        growths = [ns[i]/ns[i-1] for i in range(1, len(ns)) if ns[i-1] > 0]
        alpha = np.exp(np.mean(np.log(growths))) if growths else 1.0

        # Find peak and fit decline
        peak_idx = np.argmax(rhos)
        peak_k = ks_int[peak_idx]

        decline_ks = np.array(ks_int[peak_idx:], dtype=float)
        decline_rhos = np.array(rhos[peak_idx:])

        if len(decline_ks) >= 2 and all(r > 0 for r in decline_rhos):
            coeffs = np.polyfit(decline_ks, np.log(decline_rhos), 1)
            ratio = np.exp(coeffs[0])
            beta = alpha / ratio

            results[name] = {
                'alpha': alpha, 'beta': beta, 'ratio': ratio,
                'peak_k': peak_k, 'peak_rho': rhos[peak_idx],
            }

            print(f"  {name}:")
            print(f"    alpha (function growth) = {alpha:.2f}")
            print(f"    alpha/beta = {ratio:.4f}  (decline rate per k)")
            print(f"    beta (derivative-space growth) = {beta:.2f}")
            print(f"    beta/alpha - 1 = {beta/alpha - 1:.1%} faster")
            print(f"    Peak: k={peak_k}, rho={rhos[peak_idx]:.4f}")
            print()

    # ---- Part 4: Why log delays the peak ----
    print("\n4. WHY LOG DELAYS THE PEAK")
    print("-" * 50)

    # Use decomposition data
    print("  Decomposition by operator presence (ext_log_maths):")
    rho_by_op = decomp['rho_by_operator']

    subsets = ['log_only', 'exp_only', 'both_log_exp', 'neither']
    print(f"\n  {'Subset':>15} {'k=4':>7} {'k=5':>7} {'k=6':>7} {'k=7':>7} {'k=8':>7} {'peak_k':>7}")
    for subset in subsets:
        vals = []
        for k in range(4, 9):
            r = rho_by_op[subset][str(k)]['rho']
            vals.append(r)
        peak_k = 4 + np.argmax(vals)
        row = " ".join(f"{v:7.4f}" for v in vals)
        print(f"  {subset:>15} {row} {peak_k:>7}")

    print(f"""
Key observation: log_only peaks at k=6, while exp_only and neither peak at
k=4-5 or decline monotonically.

Why? Differentiating log-containing functions often produces 1/x terms that
combine with the log structure to stay within the function space. The
chain rule d/dx[log(f)] = f'/f generates rational functions that are more
likely to match existing forms. In contrast, exp generates exp*polynomial
products that quickly become unmatched.

Quantitatively: log_only has m(k=6) = {rho_by_op['log_only']['6']['rho']/p_le0:.3f}, while
exp_only has m(k=6) = {rho_by_op['exp_only']['6']['rho']/p_le0:.3f} — a {rho_by_op['log_only']['6']['rho']/rho_by_op['exp_only']['6']['rho']:.1f}x difference.
""")

    # ---- Part 5: Summary / paper-ready statements ----
    print("\n5. PAPER-READY SUMMARY")
    print("-" * 50)

    el = results.get('ext_log_maths', {})
    em = results.get('ext_maths', {})
    cm = results.get('core_maths', {})

    print(f"""
(a) rho(k) is determined by the competition between two growth rates:
    - |F_k| ~ alpha^k (function space enumeration)
    - S_k ~ beta^k (effective derivative form space)
    with beta > alpha in all three bases tested.

(b) The peak occurs at the crossover point where the closure growth rate
    (driven by matching to existing functions) transitions from exceeding
    to falling below the function space growth rate.

(c) Across operator bases:
    - core_maths:     alpha={cm.get('alpha',0):.1f}, beta={cm.get('beta',0):.1f}, peak at k={cm.get('peak_k','?')}
    - ext_maths:      alpha={em.get('alpha',0):.1f}, beta={em.get('beta',0):.1f}, peak at k={em.get('peak_k','?')}
    - ext_log_maths:  alpha={el.get('alpha',0):.1f}, beta={el.get('beta',0):.1f}, peak at k={el.get('peak_k','?')}

(d) Richer operator bases (more operators) shift the peak to higher k by
    providing more matching targets at moderate complexity. Log is
    particularly effective because d/dx[log(f)] = f'/f produces rational
    forms that remain within the function space.

(e) Asymptotically, rho(k) ~ (alpha/beta)^k, declining exponentially.
    For ext_log_maths, the decline rate is ~{(1-el.get('ratio',1))*100:.0f}% per unit k.
    Extrapolating: rho(k=10) ~ {el.get('peak_rho',0.1) * el.get('ratio',0.87)**(10 - el.get('peak_k',6)):.3f},
    rho(k=12) ~ {el.get('peak_rho',0.1) * el.get('ratio',0.87)**(12 - el.get('peak_k',6)):.3f}.

(f) The matching probability m(k) = rho(k)/P(delta<=0) captures the
    efficiency with which complexity-reducing derivatives find targets
    in the enumerated space. At k=8, only ~{bases['ext_log_maths']['8']['r']/p_le0:.1%} of
    complexity-reducing derivatives match, despite 89% having delta<=0.
""")


if __name__ == '__main__':
    main()
