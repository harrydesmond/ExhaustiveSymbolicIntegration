"""
Theoretical model for rho(k) peak-and-decline.

The key question: why does rho(k) = |{F in F_k : F' in F_<=k}| / |F_k|
peak at intermediate k then decline?

Model: differentiation maps F_k -> F_{k+delta} where delta has a distribution
D(delta) with mean ~ -1.1. For F' to be "closed" (in the space), two conditions:
  1. k + delta >= k_min (derivative complexity >= minimum)
  2. F' must match an existing function at complexity k' = k + delta

Condition 2 depends on a "matching probability" m(k') which we model as:
  m(k') = |F_{k'}| / S(k')
where S(k') is the effective size of the derivative form space at complexity k'.

If |F_k| ~ alpha^k and S(k') grows faster (more diverse derivative forms than
enumerated functions), then m(k') decreases with k', causing rho to decline.

The peak arises because:
- At low k: few functions, few targets for derivatives -> low rho
- At medium k: many targets, derivatives still land on common functions -> peak
- At high k: derivative forms become exotic, outpacing enumerated space -> decline
"""

import json
import numpy as np


def load_all_rho_data():
    """Load rho(k) data for all three operator bases."""

    # core_maths
    with open('rho_results_k8.json') as f:
        core = json.load(f)
    core_rho = {}
    for k_str, v in core['per_complexity'].items():
        k = int(k_str)
        if v['n_functions'] > 0:
            core_rho[k] = {
                'n_functions': v['n_functions'],
                'n_closed': v['n_closed'],
                'rho': v['rho_at_k'],
            }

    # ext_maths
    with open('rho_results_ext_maths_k8.json') as f:
        ext = json.load(f)
    ext_rho = {}
    for k_str, v in ext.items():
        k = int(k_str)
        if v['n_functions'] > 0:
            ext_rho[k] = {
                'n_functions': v['n_functions'],
                'n_closed': v['n_closed'],
                'rho': v['rho'],
            }

    # ext_log_maths
    with open('rho_results_ext_log_maths_k8.json') as f:
        elog = json.load(f)
    elog_rho = {}
    for k_str, v in elog.items():
        k = int(k_str)
        if v['n_functions'] > 0:
            elog_rho[k] = {
                'n_functions': v['n_functions'],
                'n_closed': v['n_closed'],
                'rho': v['rho'],
            }

    return core_rho, ext_rho, elog_rho


def compute_growth_rates(rho_data):
    """Compute the growth rate alpha = |F_k| / |F_{k-1}| for each k."""
    ks = sorted(rho_data.keys())
    growth = {}
    for i in range(1, len(ks)):
        k = ks[i]
        k_prev = ks[i-1]
        if k_prev in rho_data and rho_data[k_prev]['n_functions'] > 0:
            growth[k] = rho_data[k]['n_functions'] / rho_data[k_prev]['n_functions']
    return growth


def load_delta_distribution():
    """Load the delta distribution from decomposition data."""
    with open('rho_decomposition_ext_log_maths_k8.json') as f:
        data = json.load(f)
    return data['delta_distribution']


def simple_model(rho_data, delta_dist, label=""):
    """
    Simple probabilistic model for rho(k).

    For each F at complexity k:
      rho(k) = sum_{delta} P(delta) * P(F' matches | complexity k+delta)

    We estimate P(F' matches | k') empirically from the data:
      If k' is in range and F_{k'} exists, we assume the matching
      probability scales with |F_{k'}| / |F_{k'}|^gamma for some gamma > 1
      (i.e., the derivative space grows faster than the function space).

    But we can also just compute the "effective matching probability" from data:
      m_eff(k) = rho(k) / P(delta <= 0)
    """
    ks = sorted(k for k in rho_data.keys() if rho_data[k]['n_functions'] > 0)

    print(f"\n{'='*70}")
    print(f"THEORETICAL MODEL: {label}")
    print(f"{'='*70}")

    # 1. Growth rates
    growth = compute_growth_rates(rho_data)
    print(f"\nFunction space growth rates |F_k|/|F_{{k-1}}|:")
    for k, g in sorted(growth.items()):
        print(f"  k={k}: {g:.2f}x ({rho_data[k]['n_functions']} functions)")

    # Mean growth rate (geometric mean)
    if growth:
        geo_mean = np.exp(np.mean(np.log(list(growth.values()))))
        print(f"  Geometric mean growth rate: {geo_mean:.2f}x")

    # 2. Closure fraction decomposition
    print(f"\nClosure analysis:")
    print(f"  {'k':>3} {'|F_k|':>8} {'n_closed':>8} {'rho(k)':>8} "
          f"{'n_closed/n_closed_prev':>22}")
    prev_closed = None
    for k in ks:
        d = rho_data[k]
        ratio = ""
        if prev_closed and prev_closed > 0:
            ratio = f"{d['n_closed']/prev_closed:.2f}x"
        print(f"  {k:3d} {d['n_functions']:8d} {d['n_closed']:8d} "
              f"{d['rho']:8.4f} {ratio:>22}")
        prev_closed = d['n_closed']

    # 3. Key insight: compare growth of |F_k| vs growth of n_closed
    print(f"\nGrowth comparison:")
    print(f"  {'k':>3} {'|F_k| growth':>14} {'n_closed growth':>16} {'rho trend':>10}")
    prev_n = prev_c = None
    for k in ks:
        d = rho_data[k]
        n_g = c_g = trend = ""
        if prev_n and prev_n > 0:
            n_g = f"{d['n_functions']/prev_n:.2f}x"
        if prev_c and prev_c > 0:
            c_g = f"{d['n_closed']/prev_c:.2f}x"
            if prev_n and prev_n > 0:
                fn_growth = d['n_functions']/prev_n
                cl_growth = d['n_closed']/prev_c
                if cl_growth > fn_growth:
                    trend = "UP"
                elif cl_growth < fn_growth:
                    trend = "DOWN"
                else:
                    trend = "FLAT"
        print(f"  {k:3d} {n_g:>14} {c_g:>16} {trend:>10}")
        prev_n = d['n_functions']
        prev_c = d['n_closed']

    # 4. Upper bound: rho_max = P(delta <= 0)
    if delta_dist:
        p_delta_le0 = delta_dist['all']['frac_delta_le0']
        print(f"\nUpper bound from delta distribution:")
        print(f"  P(delta <= 0) = {p_delta_le0:.3f}")
        print(f"  This means at most {p_delta_le0:.1%} of derivatives could be closed")
        print(f"  Actual rho(k=8) = {rho_data[max(ks)]['rho']:.4f}")
        print(f"  Matching efficiency = rho / P(delta<=0) = "
              f"{rho_data[max(ks)]['rho'] / p_delta_le0:.4f}")
        print(f"  i.e., only {rho_data[max(ks)]['rho'] / p_delta_le0:.1%} of "
              f"complexity-reducing derivatives match existing functions")

    # 5. Predictive model: rho(k) = P(delta<=0) * m(k)
    # where m(k) is the matching probability
    # m(k) should depend on the "density" of the function space
    # at lower complexities relative to the derivative form space
    print(f"\nMatching probability m(k) = rho(k) / P(delta<=0):")
    for k in ks:
        d = rho_data[k]
        if delta_dist and d['rho'] > 0:
            m = d['rho'] / p_delta_le0
            print(f"  k={k}: m = {m:.4f}")

    return growth


def exponential_model_prediction(rho_data, delta_dist):
    """
    Analytical model: if |F_k| ~ alpha^k and the number of distinct
    derivative forms at complexity k' is S(k') ~ beta^{k'} with beta > alpha,
    then:

    m(k') ~ alpha^{k'} / beta^{k'} = (alpha/beta)^{k'}

    And rho(k) ~ sum_{delta<=0} P(delta) * (alpha/beta)^{k+delta}
              = (alpha/beta)^k * sum_{delta<=0} P(delta) * (alpha/beta)^delta

    For alpha/beta < 1, this is exponentially declining.

    But at LOW k, the model breaks down because the function space is small
    and specific (not random). The peak arises from the transition between
    the "structured" low-k regime and the "statistical" high-k regime.

    We can estimate alpha/beta from the decline rate of rho(k) at high k.
    """
    ks = sorted(k for k in rho_data.keys() if rho_data[k]['rho'] > 0)

    print(f"\n{'='*70}")
    print("EXPONENTIAL DECLINE MODEL")
    print(f"{'='*70}")

    # Fit log(rho) vs k for the declining portion
    # Use the last 3-4 points where rho is declining
    if len(ks) >= 3:
        # Find peak
        rhos = [rho_data[k]['rho'] for k in ks]
        peak_idx = np.argmax(rhos)
        peak_k = ks[peak_idx]

        if peak_idx < len(ks) - 1:
            # Fit decline: log(rho) = log(A) + k * log(alpha/beta)
            decline_ks = np.array(ks[peak_idx:], dtype=float)
            decline_rhos = np.array(rhos[peak_idx:])

            # Linear fit in log space
            if len(decline_ks) >= 2 and all(r > 0 for r in decline_rhos):
                log_rhos = np.log(decline_rhos)
                coeffs = np.polyfit(decline_ks, log_rhos, 1)
                decay_rate = coeffs[0]  # log(alpha/beta) per unit k
                ratio = np.exp(decay_rate)  # alpha/beta

                print(f"  Peak at k={peak_k}, rho={rhos[peak_idx]:.4f}")
                print(f"  Decline phase: k={ks[peak_idx]} to k={ks[-1]}")
                print(f"  Fitted decay: rho ~ {np.exp(coeffs[1]):.4f} * {ratio:.4f}^k")
                print(f"  alpha/beta ratio: {ratio:.4f}")
                print(f"  Per-unit-k decline: {(1-ratio)*100:.1f}%")

                # Extrapolate
                print(f"\n  Extrapolated rho:")
                for k_ext in range(ks[-1]+1, ks[-1]+4):
                    rho_pred = np.exp(coeffs[1] + coeffs[0] * k_ext)
                    print(f"    k={k_ext}: rho ~ {rho_pred:.4f}")

    # Growth rate of |F_k|
    growth = compute_growth_rates(rho_data)
    if growth:
        alpha = np.exp(np.mean(np.log(list(growth.values()))))
        print(f"\n  Function space growth rate alpha = {alpha:.2f}")
        if 'ratio' in dir() or ratio:
            beta = alpha / ratio
            print(f"  Implied derivative-space growth rate beta = alpha/ratio = {beta:.2f}")
            print(f"  Derivative forms grow {beta/alpha:.1f}x faster than function space")


def cross_basis_comparison(core, ext, elog):
    """Compare rho(k) behaviour across operator bases."""
    print(f"\n{'='*70}")
    print("CROSS-BASIS COMPARISON")
    print(f"{'='*70}")

    print(f"\n  {'k':>3} {'core_maths':>12} {'ext_maths':>12} {'ext_log':>12}")
    for k in range(3, 9):
        c = core.get(k, {}).get('rho', None)
        e = ext.get(k, {}).get('rho', None)
        l = elog.get(k, {}).get('rho', None)
        c_s = f"{c:.4f}" if c is not None else "-"
        e_s = f"{e:.4f}" if e is not None else "-"
        l_s = f"{l:.4f}" if l is not None else "-"
        print(f"  {k:3d} {c_s:>12} {e_s:>12} {l_s:>12}")

    # Peak locations
    for name, data in [('core_maths', core), ('ext_maths', ext), ('ext_log_maths', elog)]:
        valid = {k: v['rho'] for k, v in data.items() if v['rho'] > 0}
        if valid:
            peak_k = max(valid, key=valid.get)
            print(f"\n  {name}: peaks at k={peak_k} (rho={valid[peak_k]:.4f})")

    # Size comparison at k=8
    print(f"\n  At k=8:")
    for name, data in [('core_maths', core), ('ext_maths', ext), ('ext_log_maths', elog)]:
        if 8 in data:
            d = data[8]
            print(f"    {name}: {d['n_functions']} functions, "
                  f"{d['n_closed']} closed, rho={d['rho']:.4f}")


def rho_from_delta_and_targets(rho_data, delta_dist):
    """
    Direct computation: for each F at complexity k, its derivative lands at
    complexity k' = k + delta. We compute:

    rho_predicted(k) = sum_{delta} P(delta) * [k+delta is valid] * m(k+delta)

    where m(k') is estimated from observed data as n_closed_at_k' / n_functions_at_k'.

    This is a self-consistency check rather than a prediction.
    """
    print(f"\n{'='*70}")
    print("SELF-CONSISTENCY CHECK: rho from delta distribution")
    print(f"{'='*70}")

    if not delta_dist:
        print("  No delta distribution data available")
        return

    # The delta distribution gives us aggregate stats, not per-k
    dd = delta_dist['all']
    print(f"\n  Delta distribution (aggregate):")
    print(f"    mean = {dd['mean_delta']:.2f}")
    print(f"    std = {dd['std_delta']:.2f}")
    print(f"    P(delta<=0) = {dd['frac_delta_le0']:.3f}")
    print(f"    P(delta<=1) = {dd['frac_delta_le1']:.3f}")
    print(f"    range: [{dd['min_delta']}, {dd['max_delta']}]")

    # With mean delta ~ -1.1 and std ~ 1.5:
    # Most derivatives land 0-2 levels below the primitive
    # This means rho(k) depends on the density of functions at k-1, k-2
    # relative to the diversity of derivative forms

    # The fraction of derivatives with delta=d follows approximately:
    # P(delta=-4) + ... + P(delta=0) + ... + P(delta=4) = 1
    # We know P(delta<=0) = 0.89, P(delta<=1) = 0.96
    # So P(delta=1 to 4) ~ 0.04, P(delta > 4) ~ 0
    # Mean = -1.13, so the distribution is peaked around -1

    # Key structural insight:
    n_closed_growth = []
    n_func_growth = []
    ks = sorted(k for k in rho_data.keys() if rho_data[k]['n_functions'] > 0)

    print(f"\n  The peak-and-decline mechanism:")
    print(f"  rho(k) increases when n_closed grows faster than |F_k|")
    print(f"  rho(k) decreases when |F_k| grows faster than n_closed")
    print(f"\n  At moderate k, many derivatives land on 'popular' low-complexity")
    print(f"  targets (x, x^2, exp(x), log(x), etc.) — high multiplicity clustering.")
    print(f"  At high k, derivatives become increasingly unique — most map to")
    print(f"  novel forms not in the enumerated space.")

    # Compute cumulative |F_<=k| to show target space growth
    print(f"\n  Cumulative target space |F_<=k|:")
    cumul = 0
    for k in ks:
        cumul += rho_data[k]['n_functions']
        d = rho_data[k]
        print(f"    k={k}: |F_<=k| = {cumul:>8d}, |F_k| = {d['n_functions']:>8d}, "
              f"ratio = {cumul/d['n_functions']:.2f}")

    print(f"\n  At high k, |F_k| dominates |F_<=k| (ratio -> 1),")
    print(f"  so the cumulative target space grows at the same rate as the")
    print(f"  function space, making it increasingly unlikely for derivatives")
    print(f"  to 'find' a match.")


if __name__ == '__main__':
    core, ext, elog = load_all_rho_data()
    delta = load_delta_distribution()

    # Cross-basis overview
    cross_basis_comparison(core, ext, elog)

    # Detailed model for each basis
    for label, data in [('core_maths', core), ('ext_maths', ext),
                        ('ext_log_maths', elog)]:
        simple_model(data, delta if label == 'ext_log_maths' else None, label)

    # Exponential decline model for each
    for label, data in [('core_maths', core), ('ext_maths', ext),
                        ('ext_log_maths', elog)]:
        print(f"\n--- {label} ---")
        exponential_model_prediction(data, delta)

    # Self-consistency analysis
    rho_from_delta_and_targets(elog, delta)
