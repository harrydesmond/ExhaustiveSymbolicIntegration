"""
Generate cluster-size distribution figure (log-log histogram) and Table 4 data
for the ESI paper.

Loads ext_log_maths and trig_maths cluster data, plots cluster size distributions,
and prints the top 10 largest equivalence classes for ext_log_maths.

Usage: python3 fig_cluster_sizes.py
"""
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'legend.fontsize': 10,
    'figure.figsize': (7, 5),
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
})

# ── Load cluster data ──────────────────────────────────────────

print("Loading ext_log_maths_k8 clusters...")
with open('results_ext_log_maths_k8.pkl', 'rb') as f:
    elog_data = pickle.load(f)
elog_clusters = elog_data['clusters']

print("Loading trig_maths_k8 clusters...")
with open('results_trig_maths_k8.pkl', 'rb') as f:
    trig_data = pickle.load(f)
trig_clusters = trig_data['clusters']

# ── Compute cluster sizes ─────────────────────────────────────

elog_sizes = np.array([len(c['primitives']) for c in elog_clusters.values()])
trig_sizes = np.array([len(c['primitives']) for c in trig_clusters.values()])

print(f"ext_log_maths: {len(elog_sizes)} clusters, max size {elog_sizes.max()}")
print(f"trig_maths:    {len(trig_sizes)} clusters, max size {trig_sizes.max()}")

# ── Figure: log-log histogram of cluster sizes ────────────────

fig, ax = plt.subplots(figsize=(7, 5))

# Use logarithmically-spaced bins
max_size = max(elog_sizes.max(), trig_sizes.max())
bins = np.logspace(0, np.log10(max_size + 1), 40)

# ext_log_maths histogram
counts_elog, edges_elog = np.histogram(elog_sizes, bins=bins)
centres_elog = np.sqrt(edges_elog[:-1] * edges_elog[1:])  # geometric mean of bin edges
mask_elog = counts_elog > 0
ax.step(edges_elog[:-1][mask_elog], counts_elog[mask_elog],
        where='post', color='C2', linewidth=1.8, label='ext_log_maths')
ax.fill_between(edges_elog[:-1][mask_elog], counts_elog[mask_elog],
                step='post', color='C2', alpha=0.15)

# trig_maths histogram
counts_trig, edges_trig = np.histogram(trig_sizes, bins=bins)
mask_trig = counts_trig > 0
ax.step(edges_trig[:-1][mask_trig], counts_trig[mask_trig],
        where='post', color='C3', linewidth=1.8, label='trig_maths')
ax.fill_between(edges_trig[:-1][mask_trig], counts_trig[mask_trig],
                step='post', color='C3', alpha=0.15)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Cluster size (number of primitives)')
ax.set_ylabel('Number of clusters')
ax.set_xlim(1, 1000)
ax.legend(loc='upper right')

# Add power-law fit lines
from collections import Counter
for name, sizes, color in [('ext_log_maths', elog_sizes, 'C2'),
                            ('trig_maths', trig_sizes, 'C3')]:
    size_counts = Counter(sizes)
    s_vals = np.array(sorted(size_counts.keys()))
    c_vals = np.array([size_counts[s] for s in s_vals])
    mask = (s_vals >= 2) & (s_vals <= 500)
    if mask.sum() >= 3:
        log_s = np.log10(s_vals[mask].astype(float))
        log_c = np.log10(c_vals[mask].astype(float))
        slope, intercept = np.polyfit(log_s, log_c, 1)
        fit_s = np.logspace(np.log10(2), np.log10(800), 50)
        fit_c = 10**intercept * fit_s**slope
        ax.plot(fit_s, fit_c, '--', color=color, alpha=0.6, linewidth=1.2,
                label=f'$\\gamma = {slope:.2f}$')

ax.legend(loc='upper right')

fig.savefig('fig_cluster_sizes.pdf')
fig.savefig('fig_cluster_sizes.png')
print("\nSaved fig_cluster_sizes.pdf and fig_cluster_sizes.png")
plt.close()


# ── Table 4: Top 10 largest equivalence classes (ext_log_maths) ──

print("\n" + "=" * 90)
print("TABLE 4: Top 10 largest equivalence classes (ext_log_maths, k <= 8)")
print("=" * 90)

sorted_clusters = sorted(elog_clusters.items(),
                         key=lambda x: len(x[1]['primitives']), reverse=True)

# Print formatted table
print(f"\n{'Rank':<5} {'|F|':<6} {'k_min':<6} {'Derivative f(x)':<45} {'Simplest primitive F(x)'}")
print("-" * 90)

table_rows = []
for rank, (h, c) in enumerate(sorted_clusters[:10], 1):
    prims = c['primitives']
    n_prims = len(prims)
    min_k = min(p[0] for p in prims)
    # Find simplest primitive
    min_prims = [p for p in prims if p[0] == min_k]
    simplest = min_prims[0][1]

    deriv = c['derivative_str']
    # Truncate long expressions for display
    deriv_disp = deriv[:42] + '...' if len(deriv) > 45 else deriv
    simplest_disp = simplest[:42] + '...' if len(simplest) > 45 else simplest

    print(f"{rank:<5} {n_prims:<6} {min_k:<6} {deriv_disp:<45} {simplest_disp}")

    table_rows.append({
        'rank': rank,
        'n_primitives': n_prims,
        'min_complexity': min_k,
        'derivative': deriv,
        'simplest_primitive': simplest,
    })

# Print LaTeX-formatted table
print("\n\n% LaTeX table for Table 4")
print("\\begin{table}")
print("  \\centering")
print("  \\caption{Ten largest equivalence classes in \\texttt{ext\\_log\\_maths} at $k \\leq 8$.}")
print("  \\label{tab:largest_clusters}")
print("  \\begin{tabular}{r r r l l}")
print("    \\toprule")
print("    Rank & $|\\mathcal{F}|$ & $k_{\\min}$ & Derivative $f(x)$ & Simplest primitive $F(x)$ \\\\")
print("    \\midrule")

for row in table_rows:
    # LaTeX-safe derivative and primitive strings
    deriv_tex = row['derivative'].replace('_', '\\_').replace('**', '^')
    prim_tex = row['simplest_primitive'].replace('_', '\\_').replace('**', '^')
    # Convert to math mode
    deriv_tex = f"${deriv_tex}$"
    prim_tex = f"${prim_tex}$"
    print(f"    {row['rank']} & {row['n_primitives']} & {row['min_complexity']} "
          f"& {deriv_tex} & {prim_tex} \\\\")

print("    \\bottomrule")
print("  \\end{tabular}")
print("\\end{table}")


# ── Additional summary statistics ──────────────────────────────

print("\n\n--- Additional summary statistics ---")

# Cluster size distribution summary
for name, sizes in [('ext_log_maths', elog_sizes), ('trig_maths', trig_sizes)]:
    print(f"\n{name}:")
    print(f"  Total clusters: {len(sizes)}")
    print(f"  Singleton clusters: {np.sum(sizes == 1)}")
    multi = sizes[sizes >= 2]
    print(f"  Multi-member clusters: {len(multi)}")
    if len(multi) > 0:
        print(f"  Mean multi-member size: {multi.mean():.2f}")
        print(f"  Median multi-member size: {np.median(multi):.1f}")
        print(f"  Max cluster size: {multi.max()}")
        for thresh in [10, 50, 100, 500, 1000]:
            n = np.sum(multi >= thresh)
            if n > 0:
                print(f"  Clusters with >= {thresh} members: {n}")

    # Power law check: log-log slope
    size_counts = Counter(sizes)
    s_vals = np.array(sorted(size_counts.keys()))
    c_vals = np.array([size_counts[s] for s in s_vals])
    mask = s_vals >= 2
    if mask.sum() >= 3:
        log_s = np.log10(s_vals[mask].astype(float))
        log_c = np.log10(c_vals[mask].astype(float))
        slope, intercept = np.polyfit(log_s, log_c, 1)
        print(f"  Log-log slope (power law exponent): {slope:.2f}")
