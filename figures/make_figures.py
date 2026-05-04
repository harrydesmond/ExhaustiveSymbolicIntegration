"""Generate all paper figures for ESI.

Figures:
1. rho(k) vs k for all five bases (main result)
2. rho(k) decomposition by operator (ext_log_maths and trig_maths)
3. Delta distribution comparison
4. Cluster size distribution

Usage: python3 make_figures.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'legend.fontsize': 10,
    'figure.figsize': (7, 5),
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
})

# ── Load data ──────────────────────────────────────────────────

# Use extended results where available
with open('rho_results_core_maths_k10.json') as f:
    core_raw = json.load(f)
with open('rho_results_ext_maths_k9.json') as f:
    ext = json.load(f)
# Use k=9 results if available, else fall back to k=8
import os
_elog_file = 'rho_results_ext_log_maths_k9.json' if os.path.exists('rho_results_ext_log_maths_k9.json') else 'rho_results_ext_log_maths_k8.json'
with open(_elog_file) as f:
    elog = json.load(f)
print(f"Using ext_log data from: {_elog_file}")
with open('rho_results_trig_maths_k9.json') as f:
    trig = json.load(f)
# core_log_maths (only up to k=8)
_clog_file = 'rho_results_core_log_maths_k8.json'
if os.path.exists(_clog_file):
    with open(_clog_file) as f:
        clog = json.load(f)
else:
    clog = None

with open('rho_decomposition_ext_log_maths_k8.json') as f:
    decomp_elog = json.load(f)
with open('rho_decomposition_trig_maths_k8.json') as f:
    decomp_trig = json.load(f)

with open('theoretical_model_all_bases.json') as f:
    model = json.load(f)


def get_rho_series(data, key_rho='rho', per_complexity=False):
    """Extract k, rho, n_functions from rho data."""
    if per_complexity:
        items = data['per_complexity']
        ks = sorted(items.keys(), key=int)
        return (
            [int(k) for k in ks],
            [items[k].get('rho', items[k].get('rho_at_k', 0)) for k in ks],
            [items[k]['n_functions'] for k in ks],
        )
    else:
        ks = sorted(data.keys(), key=int)
        return (
            [int(k) for k in ks],
            [data[k][key_rho] for k in ks],
            [data[k]['n_functions'] for k in ks],
        )

# ── Figure 1: rho(k) for all five bases ───────────────────────

fig, ax = plt.subplots(figsize=(7, 5))

bases = [
    ('core_maths', core_raw, 'o', 'C0'),
    ('ext_maths', ext, 's', 'C1'),
    ('ext_log_maths', elog, '^', 'C2'),
    ('trig_maths', trig, 'D', 'C3'),
]
if clog is not None:
    bases.insert(1, ('core_log_maths', clog, 'v', 'C4'))

for name, data, marker, color in bases:
    ks, rhos, ns = get_rho_series(data)
    # Binomial error bars (more appropriate than Poisson for a proportion)
    err = [np.sqrt(r * (1 - r) / n) if n > 0 else 0 for r, n in zip(rhos, ns)]
    # Filter to k >= 4 for cleaner plot (low-k values have very few functions)
    min_k = 3 if name == 'core_maths' else 4
    mask = [k >= min_k and n >= 10 for k, n in zip(ks, ns)]
    ks_f = [k for k, m in zip(ks, mask) if m]
    rhos_f = [r for r, m in zip(rhos, mask) if m]
    err_f = [e for e, m in zip(err, mask) if m]

    label = name.replace('_', ' ')
    ax.errorbar(ks_f, rhos_f, yerr=err_f, marker=marker, color=color,
                label=label, capsize=3, linewidth=1.5, markersize=6)

ax.set_xlabel('Complexity $k$')
ax.set_ylabel(r'$\rho(k)$')
ax.legend(loc='upper right')
ax.set_xlim(3.8, 10.2)
ax.set_ylim(0, 0.22)

fig.savefig('fig_rho_all_bases.pdf')
fig.savefig('fig_rho_all_bases.png')
print("Saved fig_rho_all_bases.pdf/png")
plt.close()


# ── Figure 2: rho decomposition (ext_log_maths) ────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# ext_log_maths decomposition
subsets_elog = ['log_only', 'exp_only', 'both_log_exp', 'neither']
labels_elog = ['log only', 'exp only', 'both log+exp', 'neither']
markers_elog = ['^', 'v', 's', 'o']
colors_elog = ['C2', 'C1', 'C4', 'C0']

for subset, label, marker, color in zip(subsets_elog, labels_elog, markers_elog, colors_elog):
    rho_data = decomp_elog['rho_by_operator'][subset]
    ks = sorted(rho_data.keys(), key=int)
    ks_int = [int(k) for k in ks if int(k) >= 4]
    rhos = [rho_data[str(k)]['rho'] for k in ks_int]
    ns = [rho_data[str(k)]['n_functions'] for k in ks_int]
    errs = [np.sqrt(r * (1 - r) / n) if n > 0 else 0 for r, n in zip(rhos, ns)]
    ax1.errorbar(ks_int, rhos, yerr=errs, marker=marker, color=color,
                 label=label, capsize=3, linewidth=1.5, markersize=6)

ax1.set_xlabel('Complexity $k$')
ax1.set_ylabel(r'$\rho(k)$')
ax1.set_title('ext log maths')
ax1.legend()
ax1.set_xlim(3.8, 8.2)
ax1.set_xticks([4, 5, 6, 7, 8])
ax1.set_ylim(0, 0.22)

# trig_maths decomposition
subsets_trig = ['sin_only', 'cos_only', 'both_sin_cos', 'neither']
labels_trig = ['sin only', 'cos only', 'both sin+cos', 'neither']
markers_trig = ['^', 'v', 's', 'o']
colors_trig = ['C3', 'C5', 'C4', 'C0']

for subset, label, marker, color in zip(subsets_trig, labels_trig, markers_trig, colors_trig):
    rho_data = decomp_trig['rho_by_operator'][subset]
    ks = sorted(rho_data.keys(), key=int)
    ks_int = [int(k) for k in ks if int(k) >= 4]
    rhos = [rho_data[str(k)]['rho'] for k in ks_int]
    ns = [rho_data[str(k)]['n_functions'] for k in ks_int]
    errs = [np.sqrt(r * (1 - r) / n) if n > 0 else 0 for r, n in zip(rhos, ns)]
    if any(r > 0 for r in rhos):
        ax2.errorbar(ks_int, rhos, yerr=errs, marker=marker, color=color,
                     label=label, capsize=3, linewidth=1.5, markersize=6)

ax2.set_xlabel('Complexity $k$')
ax2.set_ylabel(r'$\rho(k)$')
ax2.set_title('trig maths')
ax2.legend()
ax2.set_xlim(3.8, 8.2)
ax2.set_xticks([4, 5, 6, 7, 8])

fig.tight_layout()
fig.savefig('fig_rho_decomposition.pdf')
fig.savefig('fig_rho_decomposition.png')
print("Saved fig_rho_decomposition.pdf/png")
plt.close()


# ── Figure 3: Delta distribution comparison ─────────────────────

fig, ax = plt.subplots(figsize=(7, 5))

# Plot delta stats as bar chart
delta_data = {
    'ext\_log all': decomp_elog['delta_distribution']['all'],
    'ext\_log has\_log': decomp_elog['delta_distribution'].get('has_log', {}),
    'ext\_log no\_log': decomp_elog['delta_distribution'].get('no_log', {}),
    'trig all': decomp_trig['delta_distribution']['all'],
    'trig has\_trig': decomp_trig['delta_distribution'].get('has_trig', {}),
    'trig no\_trig': decomp_trig['delta_distribution'].get('no_trig', {}),
}

names = []
means = []
p_le0 = []
for name, d in delta_data.items():
    if d and 'mean_delta' in d:
        names.append(name)
        means.append(d['mean_delta'])
        p_le0.append(d['frac_delta_le0'] * 100)

x_pos = np.arange(len(names))
width = 0.35

bars1 = ax.bar(x_pos - width/2, means, width, label=r'Mean $\delta$', color='C0', alpha=0.8)
ax2 = ax.twinx()
bars2 = ax2.bar(x_pos + width/2, p_le0, width, label=r'$P(\delta \leq 0)$ (%)', color='C1', alpha=0.8)

ax.set_xticks(x_pos)
ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
ax.set_ylabel(r'Mean $\delta$')
ax2.set_ylabel(r'$P(\delta \leq 0)$ (%)')

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='lower left')

fig.tight_layout()
fig.savefig('fig_delta_distribution.pdf')
fig.savefig('fig_delta_distribution.png')
print("Saved fig_delta_distribution.pdf/png")
plt.close()


# ── Figure 4: Growth-rate crossover model ─────────────────────

fig, ax = plt.subplots(figsize=(7, 5))

for name, color, marker in [('core_maths', 'C0', 'o'), ('ext_maths', 'C1', 's'),
                              ('ext_log_maths', 'C2', '^'), ('trig_maths', 'C3', 'D')]:
    m = model[name]
    label = name.replace('_', r'\_')
    # Data points
    if name == 'core_maths':
        ks, rhos, _ = get_rho_series(core_raw)
    elif name == 'ext_maths':
        ks, rhos, _ = get_rho_series(ext)
    elif name == 'ext_log_maths':
        ks, rhos, _ = get_rho_series(elog)
    else:
        ks, rhos, _ = get_rho_series(trig)

    mask = [k >= 3 for k in ks]
    ks_f = [k for k, m_ in zip(ks, mask) if m_]
    rhos_f = [r for r, m_ in zip(rhos, mask) if m_]

    ax.plot(ks_f, rhos_f, marker=marker, color=color, label=label,
            linewidth=1.5, markersize=6)

    # Model extrapolation from the empirical peak (use k=7 for trig, not k=2)
    if name == 'trig_maths':
        # Use local peak at k=7 for extrapolation, not global k=2
        pk = 7
        pk_rho = rhos_f[ks_f.index(7)] if 7 in ks_f else rhos_f[-2]
    else:
        pk = m['peak_k']
        pk_rho = m['peak_rho']
    k_ext = np.linspace(max(ks_f), 12, 50)
    rho_ext = pk_rho * m['ratio'] ** (k_ext - pk)
    ax.plot(k_ext, rho_ext, '--', color=color, alpha=0.5, linewidth=1)

ax.set_xlabel('Complexity $k$')
ax.set_ylabel(r'$\rho(k)$')
ax.legend()
ax.set_xlim(3.8, 12.5)
ax.set_ylim(0, 0.14)

# Add alpha/beta annotations
y_offsets = {'core_maths': 0.002, 'ext_maths': -0.002,
             'ext_log_maths': 0.002, 'trig_maths': -0.002}
for name_ in ['core_maths', 'ext_maths', 'ext_log_maths', 'trig_maths']:
    m_ = model[name_]
    short = name_.split('_')[0]
    pk_ = 7 if name_ == 'trig_maths' else m_['peak_k']
    pk_rho_ = 0.051 if name_ == 'trig_maths' else m_['peak_rho']
    y_val = pk_rho_ * m_['ratio'] ** (10 - pk_) + y_offsets.get(name_, 0)
    ax.annotate(f'{short}: $\\alpha/\\beta$={m_["ratio"]:.2f}',
                xy=(10.2, y_val), fontsize=8, alpha=0.7)

fig.savefig('fig_model_extrapolation.pdf')
fig.savefig('fig_model_extrapolation.png')
print("Saved fig_model_extrapolation.pdf/png")
plt.close()

print("\nAll figures generated successfully.")
