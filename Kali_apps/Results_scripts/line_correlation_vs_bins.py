"""
line_correlation_vs_bins.py
───────────────────────────
Plots Pearson r (solid) and Spearman ρ (dashed) vs bin count
for each k value, broken down by comparison subset.

Usage:
    python line_correlation_vs_bins.py --input leverage_summary_all.csv
    python line_correlation_vs_bins.py --input leverage_summary_all.csv --output my_figure.png --dpi 300
"""

import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Line plot: correlation vs bins from leverage_summary_all.csv')
parser.add_argument('--input',  required=True, help='Path to leverage_summary_all.csv')
parser.add_argument('--output', default='Line_Correlation_vs_Bins.png', help='Output image filename')
parser.add_argument('--dpi',    type=int, default=180, help='Output DPI (default 180; use 300 for publication)')
args = parser.parse_args()

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(args.input)

k_vals   = sorted(df['k'].unique())          # [3, 4, 5, 6, 7]
bin_vals = sorted(df['bins'].unique())       # [50, 100, 200, 300, 500, 700, 1000]

# Subsets to plot and their colours
subsets = [
    ('All pairs',       '#1F4E79'),   # dark blue
    ('Within-genus',    '#1F6B3A'),   # dark green
    ('Within-E. coli',  '#2171B5'),   # mid blue
    ('Within-Shigella', '#CB181D'),   # red
]

# ── Figure setup: 3-top / 2-bottom grid ──────────────────────────────────────
fig = plt.figure(figsize=(18, 11))

# Define axes positions manually for 3-top / 2-bottom layout
# Row 1: k=3, k=4, k=5  |  Row 2: k=6, k=7 (centred)
axes_positions = [
    [0.04,  0.54, 0.29, 0.38],   # k=3  (left=0.04, bottom=0.54, w=0.29, h=0.38)
    [0.36,  0.54, 0.29, 0.38],   # k=4
    [0.68,  0.54, 0.29, 0.38],   # k=5
    [0.20,  0.07, 0.29, 0.38],   # k=6  (centred in row 2)
    [0.52,  0.07, 0.29, 0.38],   # k=7
]

axes = [fig.add_axes(pos) for pos in axes_positions]

# ── Plot each k value ─────────────────────────────────────────────────────────
for idx, k in enumerate(k_vals):
    ax = axes[idx]
    sub_k = df[df['k'] == k].sort_values('bins')

    for subset, colour in subsets:
        sub = sub_k[sub_k['subset'] == subset]
        if sub.empty:
            continue

        x = sub['bins'].values
        pr  = sub['pearson_r'].values
        sp  = sub['spearman_rho'].values

        # Pearson r — solid line with filled markers
        ax.plot(x, pr,  color=colour, linestyle='-',  linewidth=2.0,
                marker='o', markersize=5, markerfacecolor=colour, zorder=4)

        # Spearman ρ — dashed line with open markers
        ax.plot(x, sp,  color=colour, linestyle='--', linewidth=1.8,
                marker='o', markersize=5, markerfacecolor='white',
                markeredgecolor=colour, markeredgewidth=1.4, zorder=3,
                alpha=0.85)

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_title(f'k = {k}', fontsize=13, fontweight='bold', pad=8)
    ax.set_xlabel('Bins', fontsize=10)
    ax.set_ylabel('Correlation', fontsize=10)
    ax.set_xticks(bin_vals)
    ax.set_xticklabels([str(b) for b in bin_vals], fontsize=8, rotation=30)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(True, alpha=0.25, linestyle=':')

    # y-axis range: pad slightly below min and cap at 1.01
    all_vals = []
    for subset, _ in subsets:
        s = sub_k[sub_k['subset'] == subset]
        if not s.empty:
            all_vals += list(s['pearson_r'].values) + list(s['spearman_rho'].values)
    if all_vals:
        ymin = max(0.55, min(all_vals) - 0.015)
        ax.set_ylim(ymin, 1.012)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ── Shared legend ─────────────────────────────────────────────────────────────
# Part 1: subset colours
legend_subset = [
    mlines.Line2D([], [], color=col, linestyle='-', linewidth=2,
                  marker='o', markersize=6, label=label)
    for label, col in subsets
]

# Part 2: metric line styles
legend_metric = [
    mlines.Line2D([], [], color='#555555', linestyle='-',  linewidth=2,
                  marker='o', markersize=6, markerfacecolor='#555555',
                  label='Pearson r (solid, filled)'),
    mlines.Line2D([], [], color='#555555', linestyle='--', linewidth=1.8,
                  marker='o', markersize=6, markerfacecolor='white',
                  markeredgecolor='#555555', markeredgewidth=1.4,
                  label='Spearman \u03c1 (dashed, open)'),
]

# Place legend below the bottom row, centred
fig.legend(
    handles=legend_subset + legend_metric,
    loc='lower center',
    ncol=3,
    fontsize=9.5,
    framealpha=0.95,
    edgecolor='#CCCCCC',
    bbox_to_anchor=(0.5, -0.01),
    title='Comparison subset and metric',
    title_fontsize=9.5
)

# ── Title ─────────────────────────────────────────────────────────────────────
fig.suptitle(
    'Pearson r (solid) and Spearman \u03c1 (dashed) vs Bin Count\nby k value and comparison subset',
    fontsize=13, fontweight='bold', y=0.98
)

# ── Save ──────────────────────────────────────────────────────────────────────
out = Path(args.output)
plt.savefig(out, dpi=args.dpi, bbox_inches='tight')
print(f"Saved: {out}  (DPI={args.dpi})")
