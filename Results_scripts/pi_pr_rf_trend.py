#!/usr/bin/env python3
# ============================================================
# PI vs PR RF Trend Across Bin Sizes (with ±1 SD bands)
# Paul Alemoh | UALR Bioinformatics
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
INPUT_CSV  = "pi_pr_matched_rf.csv"
OUTPUT_PDF = "pi_pr_rf_trend.pdf"

# ============================================================
# 1. Load
# ============================================================
df = pd.read_csv(INPUT_CSV)
print(df.head())
print(f"\nUnique k values : {sorted(df['k'].unique())}")
print(f"Unique bin sizes: {sorted(df['bins'].unique())}")

# Drop combined k rows for the main trend (optional — set to False to keep)
DROP_COMBINED = True
if DROP_COMBINED:
    df = df[~df['k'].str.contains("combined", case=False)].copy()
    print("\nDropped combined k rows. Remaining k values:", sorted(df['k'].unique()))

# ============================================================
# 2. Sort k naturally
# ============================================================
def k_sort_key(k_str):
    nums = [int(x) for x in k_str.replace('k', '').split('-') if x.isdigit()]
    return nums[0] if nums else 99

all_ks   = sorted(df['k'].unique(), key=k_sort_key)
all_bins = sorted(df['bins'].unique())

colors = {
    'k3': '#1f77b4',
    'k4': '#ff7f0e',
    'k5': '#2ca02c',
    'k6': '#d62728',
    'k7': '#9467bd',
    'k8': '#8c564b',
    'k9': '#e377c2',
}

# ============================================================
# 3. Plot — one line per k, x-axis = bin size
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Plot A: RF by bin size, one line per k ---
ax = axes[0]
for k in all_ks:
    sub = df[df['k'] == k].sort_values('bins')
    color = colors.get(k, 'grey')
    ax.plot(sub['bins'], sub['RF'],
            marker='o', linewidth=2.0, color=color, label=k)
    # SD band: not applicable per-pair, so shade ±0 (single value per bin)
    # Instead add a marker size scaled to RF magnitude
ax.set_xlabel("Bin size", fontsize=12)
ax.set_ylabel("Normalized RF (KALI_Hash vs KALI_Non-Hash)", fontsize=12)
ax.set_title("KALI_Hash vs KALI_Non-Hash RF across bin sizes\n(one line per k)", fontsize=12, fontweight='bold')
ax.set_xticks(all_bins)
ax.tick_params(axis='x', rotation=30)
ax.grid(True, linestyle='--', linewidth=0.5, color='lightgrey')
ax.spines[['top', 'right']].set_visible(False)
ax.legend(title="k", fontsize=10, title_fontsize=10, frameon=True)

# --- Plot B: RF by k, one line per bin size ---
ax2 = axes[1]

bin_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(all_bins)))

for i, b in enumerate(all_bins):
    sub = df[df['bins'] == b].copy()
    sub['k_num'] = sub['k'].str.replace('k', '').astype(int)
    sub = sub.sort_values('k_num')
    ax2.plot(sub['k_num'], sub['RF'],
             marker='o', linewidth=2.0,
             color=bin_colors[i], label=f'bins={b}')

ax2.set_xlabel("K-mer size", fontsize=12)
ax2.set_ylabel("Normalized RF (KALI_Hash vs KALI_Non-Hash)", fontsize=12)
ax2.set_title("KALI_Hash vs KALI_Non-Hash RF across k values\n(one line per bin size)", fontsize=12, fontweight='bold')
ax2.set_xticks(sorted(df['k'].str.replace('k','').astype(int).unique()))
ax2.set_xticklabels([f'k{k}' for k in sorted(df['k'].str.replace('k','').astype(int).unique())])
ax2.grid(True, linestyle='--', linewidth=0.5, color='lightgrey')
ax2.spines[['top', 'right']].set_visible(False)
ax2.legend(title="Bin size", fontsize=9, title_fontsize=10,
           frameon=True, bbox_to_anchor=(1.01, 1), loc='upper left')

# ============================================================
# 4. Reference lines
# ============================================================
for ax in axes:
    ax.axhline(0.05, color='green', linestyle=':', linewidth=1.2, alpha=0.7,
               label='RF=0.05 threshold')
    ax.axhline(0.10, color='orange', linestyle=':', linewidth=1.2, alpha=0.7)

# ============================================================
# 5. Summary annotation
# ============================================================
mean_rf = df['RF'].mean()
fig.suptitle(
    f"KALI_Hash vs KALI_Non-Hash Matched RF Trend  |  Mean RF={mean_rf:.3f}  |  "
    f"Near-identical (RF≤0.05): {(df['RF']<=0.05).sum()}/{len(df)} pairs",
    fontsize=11, y=1.02
)

plt.tight_layout()
plt.savefig(OUTPUT_PDF, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {OUTPUT_PDF}")
