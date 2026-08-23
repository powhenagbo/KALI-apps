#!/usr/bin/env python3
# ============================================================
# PI vs PR RF Trend — separate lines per metric
# Paul Alemoh | UALR Bioinformatics
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# ============================================================
# CONFIG
# ============================================================
INPUT_CSV  = "rf_matrix_normalized.csv"   # <-- full RF matrix
OUTPUT_PDF = "pi_pr_separate_trend.pdf"

# ============================================================
# 1. Load matrix
# ============================================================
df     = pd.read_csv(INPUT_CSV, index_col=0)
labels = df.index.tolist()
vals   = df.values.astype(float)
np.fill_diagonal(vals, np.nan)

# ============================================================
# 2. Parse labels
# ============================================================
def parse_label(l):
    if re.search(r'k\d+-k\d+', l):   # skip combined
        return None, None, None
    metric  = "PI" if l.startswith("PI_") else "PR"
    k_match = re.search(r'[_-]k(\d+)[_\-b]', l)
    b_match = re.search(r'(?:bins?|b)(\d+)', l)
    k    = int(k_match.group(1)) if k_match else None
    bins = int(b_match.group(1)) if b_match else None
    return metric, k, bins

parsed = [parse_label(l) for l in labels]

all_ks   = sorted(set(p[1] for p in parsed if p[1] is not None))
all_bins = sorted(set(p[2] for p in parsed if p[2] is not None))

# ============================================================
# 3. For each metric × k, compute mean RF against ALL others
#    at each bin size
# ============================================================
def mean_rf_for_group(group_indices, target_indices):
    sub  = vals[np.ix_(group_indices, target_indices)]
    flat = sub[~np.isnan(sub)].flatten()
    return np.nanmean(flat), np.nanstd(flat)

# color pairs: PI solid, PR dashed — same hue per k
k_colors = {
    3: '#1f77b4',
    4: '#ff7f0e',
    5: '#2ca02c',
    6: '#d62728',
    7: '#9467bd',
}

# ============================================================
# 4. Plot A — one panel per k, PI vs PR across bin sizes
# ============================================================
fig, axes = plt.subplots(1, len(all_ks), figsize=(4 * len(all_ks), 5), sharey=True)

for ax, k in zip(axes, all_ks):
    for metric, ls, marker in [("KH", "-", "o"), ("KNH", "--", "s")]:
        means, sds = [], []
        for b in all_bins:
            grp_idx = [i for i, p in enumerate(parsed)
                       if p[0] == metric and p[1] == k and p[2] == b]
            all_idx = [i for i, p in enumerate(parsed) if p[0] is not None]
            if not grp_idx:
                means.append(np.nan); sds.append(np.nan); continue
            m, s = mean_rf_for_group(grp_idx, all_idx)
            means.append(m); sds.append(s)

        means, sds = np.array(means), np.array(sds)
        color = k_colors.get(k, 'grey')
        ax.plot(all_bins, means, ls=ls, marker=marker,
                color=color, linewidth=2, label=metric)
        ax.fill_between(all_bins, means - sds, means + sds,
                        color=color, alpha=0.12)

    ax.set_title(f"k={k}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Bin size", fontsize=10)
    ax.set_xticks(all_bins)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, linestyle='--', linewidth=0.4, color='lightgrey')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=9, frameon=False)

axes[0].set_ylabel("Mean RF against all others (±1 SD)", fontsize=10)
fig.suptitle("KALI_Hash vs KALI_Non-Hash — Mean RF across bin sizes, per k\n(lower = more representative tree)",
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_PDF.replace(".pdf", "_by_k.pdf"), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_PDF.replace('.pdf', '_by_k.pdf')}")

# ============================================================
# 5. Plot B — one panel per bin size, PI vs PR across k
# ============================================================
fig, axes = plt.subplots(1, len(all_bins), figsize=(3 * len(all_bins), 5), sharey=True)

bin_colors = {
    50:   '#e41a1c',
    100:  '#ff7f00',
    200:  '#4daf4a',
    300:  '#377eb8',
    500:  '#984ea3',
    700:  '#a65628',
    1000: '#333333',
}

for ax, b in zip(axes, all_bins):
    for metric, ls, marker in [("KH", "-", "o"), ("KNH", "--", "s")]:
        means, sds = [], []
        for k in all_ks:
            grp_idx = [i for i, p in enumerate(parsed)
                       if p[0] == metric and p[1] == k and p[2] == b]
            all_idx = [i for i, p in enumerate(parsed) if p[0] is not None]
            if not grp_idx:
                means.append(np.nan); sds.append(np.nan); continue
            m, s = mean_rf_for_group(grp_idx, all_idx)
            means.append(m); sds.append(s)

        means, sds = np.array(means), np.array(sds)
        color = bin_colors.get(b, 'grey')
        ax.plot(all_ks, means, ls=ls, marker=marker,
                color=color, linewidth=2, label=metric)
        ax.fill_between(all_ks, means - sds, means + sds,
                        color=color, alpha=0.12)

    ax.set_title(f"bins={b}", fontsize=11, fontweight='bold')
    ax.set_xlabel("k", fontsize=10)
    ax.set_xticks(all_ks)
    ax.set_xticklabels([f'k{k}' for k in all_ks])
    ax.grid(True, linestyle='--', linewidth=0.4, color='lightgrey')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=9, frameon=False)

axes[0].set_ylabel("Mean RF against all others (±1 SD)", fontsize=10)
fig.suptitle("KALI_Hash vs KALI_Non-Hash — Mean RF across k values, per bin size\n(lower = more representative tree)",
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_PDF.replace(".pdf", "_by_bins.pdf"), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_PDF.replace('.pdf', '_by_bins.pdf')}")

print("\nDone.")
