"""Compute theoretical model parameters for trig_maths basis."""
import json
import numpy as np

with open('rho_results_trig_maths_k8.json') as f:
    trig = json.load(f)

# Also load other bases for comparison
with open('rho_results_k8.json') as f:
    core_raw = json.load(f)
with open('rho_results_ext_maths_k8.json') as f:
    ext = json.load(f)
with open('rho_results_ext_log_maths_k8.json') as f:
    elog = json.load(f)

bases = {
    'core_maths': {k: {'n': v['n_functions'], 'c': v['n_closed'], 'r': v.get('rho', v.get('rho_at_k', 0))}
                   for k, v in core_raw['per_complexity'].items()
                   if v['n_functions'] > 0},
    'ext_maths': {k: {'n': v['n_functions'], 'c': v['n_closed'], 'r': v['rho']}
                  for k, v in ext.items() if v['n_functions'] > 0},
    'ext_log_maths': {k: {'n': v['n_functions'], 'c': v['n_closed'], 'r': v['rho']}
                      for k, v in elog.items() if v['n_functions'] > 0},
    'trig_maths': {k: {'n': v['n_functions'], 'c': v['n_closed'], 'r': v['rho']}
                   for k, v in trig.items() if v['n_functions'] > 0},
}

print("=" * 70)
print("GROWTH RATES AND THEORETICAL MODEL PARAMETERS")
print("=" * 70)

results = {}
for name, data in bases.items():
    ks = sorted(data.keys(), key=int)
    ns = [data[k]['n'] for k in ks]
    rhos = [data[k]['r'] for k in ks]
    ks_int = [int(k) for k in ks]

    # Function space growth rate alpha
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
    else:
        ratio = 1.0
        beta = alpha

    results[name] = {
        'alpha': float(alpha), 'beta': float(beta),
        'ratio': float(ratio), 'peak_k': peak_k,
        'peak_rho': float(rhos[peak_idx]),
        'decline_pct': float((1 - ratio) * 100),
    }

    print(f"\n  {name}:")
    print(f"    alpha (function growth)     = {alpha:.2f}")
    print(f"    beta (derivative-space)     = {beta:.2f}")
    print(f"    alpha/beta                  = {ratio:.4f}")
    print(f"    decline rate                = {(1-ratio)*100:.1f}%/k")
    print(f"    peak: k={peak_k}, rho={rhos[peak_idx]:.4f}")
    print(f"    rho(k) values: {[f'{r:.4f}' for r in rhos]}")

    # Print growth table
    print(f"\n    {'k':>3} {'|F_k|':>9} {'n_closed':>9} {'rho':>8} "
          f"{'F_growth':>9} {'C_growth':>9}")
    prev_n = prev_c = None
    for k in ks:
        d = data[k]
        fg = cg = ""
        if prev_n and prev_n > 0 and prev_c and prev_c > 0:
            fg = f"{d['n']/prev_n:.2f}x"
            cg = f"{d['c']/prev_c:.2f}x"
        print(f"    {k:>3} {d['n']:9d} {d['c']:9d} {d['r']:8.4f} {fg:>9} {cg:>9}")
        prev_n = d['n']
        prev_c = d['c']

print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(f"{'Basis':>15} {'alpha':>6} {'beta':>6} {'alpha/beta':>10} "
      f"{'decline':>8} {'peak_k':>6} {'peak_rho':>9}")
for name, r in results.items():
    print(f"{name:>15} {r['alpha']:6.2f} {r['beta']:6.2f} {r['ratio']:10.4f} "
          f"{r['decline_pct']:7.1f}% {r['peak_k']:>6} {r['peak_rho']:9.4f}")

# Save
with open('theoretical_model_all_bases.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to theoretical_model_all_bases.json")
