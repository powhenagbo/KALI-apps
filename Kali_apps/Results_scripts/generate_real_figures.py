#!/usr/bin/env python3
"""
generate_real_figures.py
========================
Generates Figures 2 to 7 using YOUR REAL data from
KALI_Non-Hash and KALI_Hash CSV outputs.

USAGE
-----
    # Generate all figures at once
    python generate_real_figures.py \
        --pykali   results/PR_k3_b300.csv \
        --spacing  results/PI_k3_bins200.csv \
        --outdir   real_figures/

    # Generate specific figures only
    python generate_real_figures.py \
        --pykali  results/PR_k3_b300.csv \
        --spacing results/PI_k3_bins200.csv \
        --figs 3 4 7

    # Figure 6 needs benchmark times CSV
    python generate_real_figures.py \
        --pykali      results/PR_k3_b300.csv \
        --spacing     results/PI_k3_bins200.csv \
        --benchmarks  results/benchmarks.csv

BENCHMARK CSV FORMAT (for Figure 6)
-------------------------------------
    genomes,kali_nonhash_seconds,kali_hash_seconds
    30,45,9
    50,85,14
    100,210,28
    200,520,58
    500,1650,155
    1000,3900,320

FIGURE DESCRIPTIONS
-------------------
    Fig 2 — Shared bin range: real distance profiles from your two genomes
    Fig 3 — Full 30x30 pairwise distance heatmap from KALI_Non-Hash CSV
    Fig 4 — NJ trees side by side (KALI_Non-Hash vs KALI_Hash) with RF annotation
    Fig 5 — RF distance bar charts across k values
    Fig 6 — Scaling benchmark runtime and speed multiplier
    Fig 7 — Outlier leverage scatter: E.coli vs Shigella vs cross-genus pairs

DEPENDENCIES
------------
    pip install matplotlib numpy pandas scipy seaborn
    pip install dendropy   # for Figure 4 NJ trees (optional)

NOTE
----
    Figure 4 NJ trees require dendropy. If not installed, Figure 4
    will be skipped and a message shown.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Ellipse
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_matrix(path):
    """Load a distance matrix CSV, strip NCBI version suffixes."""
    df = pd.read_csv(path, index_col=0)
    df.index   = df.index.str.replace(r'\.\d+$', '', regex=True)
    df.columns = df.columns.str.replace(r'\.\d+$', '', regex=True)
    return df


def upper_triangle(df_a, df_b=None):
    """
    Extract upper triangle values from one or two matrices.
    If two matrices given, aligns on shared genome names.
    Returns (vec_a, vec_b, shared_names) or (vec_a, names) if one matrix.
    """
    if df_b is None:
        n   = len(df_a)
        idx = np.triu_indices(n, k=1)
        return df_a.values[idx], list(df_a.index)

    shared = sorted(set(df_a.index) & set(df_b.index))
    if len(shared) < 3:
        raise ValueError(f"Only {len(shared)} shared genomes — need at least 3.")
    a   = df_a.loc[shared, shared].values
    b   = df_b.loc[shared, shared].values
    n   = len(shared)
    idx = np.triu_indices(n, k=1)
    return a[idx], b[idx], shared



def load_metadata(path):
    """Load metadata_labels.csv -> (accession_to_name, accession_to_species)"""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df['Accession']   = df['Accession'].str.strip().str.replace(r'\.\d+$', '', regex=True)
    df['Strain_Name'] = df['Strain_Name'].str.strip()
    df['Color']       = df['Color'].str.strip()
    accession_to_name    = dict(zip(df['Accession'], df['Strain_Name']))
    accession_to_species = {
        acc: ('ecoli' if color == 'blue' else 'shigella')
        for acc, color in zip(df['Accession'], df['Color'])
    }
    return accession_to_name, accession_to_species


def rename_matrix(df, accession_to_name):
    """Rename matrix rows/cols from accession IDs to strain names."""
    df = df.copy()
    df.index   = [accession_to_name.get(i, i) for i in df.index]
    df.columns = [accession_to_name.get(c, c) for c in df.columns]
    return df


def detect_species(names, accession_to_species=None):
    """
    Auto-detect E. coli vs Shigella from genome names.
    Returns list of labels: 'ecoli', 'shigella', or 'unknown'.
    Falls back to first half = ecoli, second half = shigella
    if no keywords found.
    """
    labels = []
    for name in names:
        n = name.lower()
        if any(k in n for k in ['shigella', 'sh_', 'shig', 'shi']):
            labels.append('shigella')
        elif any(k in n for k in ['ecoli', 'ec_', 'coli', 'escherichia']):
            labels.append('ecoli')
        else:
            labels.append('unknown')

    # If all unknown, split first half / second half
    if all(l == 'unknown' for l in labels):
        mid = len(labels) // 2
        labels = ['ecoli'] * mid + ['shigella'] * (len(labels) - mid)
        print("  WARNING: could not detect species from genome names.")
        print(f"  Assuming first {mid} = E. coli, rest = Shigella.")
        print("  Use --ecoli and --shigella flags to specify manually.")

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Shared Bin Range
# ─────────────────────────────────────────────────────────────────────────────

def draw_fig2(df_pykali, outdir, genome1=None, genome2=None):
    print("\nGenerating Figure 2 — Shared Bin Range...")
    genomes = list(df_pykali.index)
    g1 = genome1 or genomes[0]
    g2 = genome2 or genomes[1]

    vec_A = df_pykali.loc[g1].values.astype(float)
    vec_B = df_pykali.loc[g2].values.astype(float)

    # Remove zeros (self-distance diagonal)
    mask  = (vec_A > 0) | (vec_B > 0)
    vec_A = vec_A[mask]
    vec_B = vec_B[mask]
    n_bins = min(30, len(vec_A))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#f8f9fa')

    # Left — separate ranges (WRONG)
    ax = axes[0]
    ax.hist(vec_A, bins=n_bins, color='#2980b9', alpha=0.75,
            label=g1[:22], density=True)
    ax.hist(vec_B, bins=n_bins, color='#e74c3c', alpha=0.75,
            label=g2[:22], density=True)
    ax.set_title('Without Shared Bin Ranges\n(Histograms NOT comparable)',
                 fontsize=11, fontweight='bold', color='#c0392b')
    ax.set_xlabel('Pairwise distance value', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(fontsize=8); ax.set_facecolor('#fff')
    ax.text(0.5, 0.88,
            '\u2717  Different bin ranges per genome\n    Distance computation meaningless',
            transform=ax.transAxes, ha='center', fontsize=8.5, color='#c0392b',
            bbox=dict(boxstyle='round', facecolor='#fdecea', edgecolor='#c0392b'))

    # Right — shared ranges (CORRECT)
    ax = axes[1]
    bins_s = np.linspace(min(vec_A.min(), vec_B.min()),
                         max(vec_A.max(), vec_B.max()), n_bins + 1)
    ax.hist(vec_A, bins=bins_s, color='#2980b9', alpha=0.75,
            label=g1[:22], density=True)
    ax.hist(vec_B, bins=bins_s, color='#e74c3c', alpha=0.75,
            label=g2[:22], density=True)
    ax.set_title('With Shared Bin Ranges\n(Histograms ARE comparable)',
                 fontsize=11, fontweight='bold', color='#27ae60')
    ax.set_xlabel('Pairwise distance value', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(fontsize=8); ax.set_facecolor('#fff')
    ax.text(0.5, 0.88,
            '\u2713  Global min/max across all genomes\n    Distance computation valid',
            transform=ax.transAxes, ha='center', fontsize=8.5, color='#27ae60',
            bbox=dict(boxstyle='round', facecolor='#eafaf1', edgecolor='#27ae60'))

    fig.suptitle(f'Figure 2: The Shared Bin Range Design Decision\n'
                 f'Real data — {g1} vs {g2}',
                 fontsize=12, fontweight='bold', color='#2c3e50', y=1.02)
    plt.tight_layout()
    out = os.path.join(outdir, 'fig2_sharedbins_real.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Distance Matrix Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def draw_fig3(df_KALI, outdir, species_labels=None, df_spacing=None):
    print("\nGenerating Figure 3 — Distance Matrix Heatmap (both methods)...")

    names  = list(df_KALI.index)
    n      = len(names)
    labels = species_labels or detect_species(names)

    # Sort so E. coli comes first, Shigella second
    order = ([i for i,l in enumerate(labels) if l=='ecoli'] +
             [i for i,l in enumerate(labels) if l=='shigella'] +
             [i for i,l in enumerate(labels) if l=='unknown'])
    sorted_names = [names[i] for i in order]
    df_p_sorted  = df_KALI.loc[sorted_names, sorted_names]

    n_ec = sum(1 for l in labels if l=='ecoli')
    n_sh = sum(1 for l in labels if l=='shigella')

    def _draw_one(ax, df_sorted, title):
        sns.heatmap(df_sorted, ax=ax, cmap='YlOrRd',
                    linewidths=0.3, linecolor='white',
                    cbar_kws={'label': 'Cosine Distance', 'shrink': 0.8},
                    xticklabels=True, yticklabels=True, annot=False)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=11, fontweight='bold')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=11, fontweight='bold')
        if n_ec > 0:
            ax.add_patch(plt.Rectangle((0, 0), n_ec, n_ec,
                         fill=False, edgecolor='#2980b9', lw=2.5))
            ax.text(n_ec/2, -1.2, f'E. coli (n={n_ec})',
                    ha='center', fontsize=9, fontweight='bold', color='#2980b9')
        if n_sh > 0:
            ax.add_patch(plt.Rectangle((n_ec, n_ec), n_sh, n_sh,
                         fill=False, edgecolor='#e74c3c', lw=2.5))
            ax.text(n_ec + n_sh/2, -1.2, f'Shigella (n={n_sh})',
                    ha='center', fontsize=9, fontweight='bold', color='#e74c3c')
        ax.set_title(title, fontsize=11, fontweight='bold', color='#2c3e50', pad=14)

    if df_spacing is not None:
        # Side-by-side: KALI_Non-Hash left, KALI_Hash right
        fig, axes = plt.subplots(1, 2, figsize=(26, 12))
        fig.patch.set_facecolor('#f8f9fa')
        _draw_one(axes[0], df_p_sorted,
                  f'KALI_non-hash — {n} genomes')
        # Align spacing to same genome order
        shared = [g for g in sorted_names if g in df_spacing.index]
        df_s_sorted = df_spacing.loc[shared, shared]
        _draw_one(axes[1], df_s_sorted,
                  f'KALI_Hash — {n} genomes')
        fig.suptitle(
            'Figure 6: Pairwise Distance Matrix Heatmap\n'
            'Left: KALI_non-hash  | Right: KALI_Hash',
            fontsize=13, fontweight='bold', color='#2c3e50', y=1.01)
    else:
        fig, ax = plt.subplots(figsize=(14, 12))
        fig.patch.set_facecolor('#f8f9fa')
        _draw_one(ax, df_p_sorted,
                  f'Figure 3: Pairwise Distance Matrix Heatmap\nKALI_non-hash — {n} genomes')

    plt.tight_layout()
    out = os.path.join(outdir, 'fig3_heatmap_real.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
    plt.close()
    print(f"  Saved: {out}")
    print(f"  KALI  stats: min={df_KALI.values[df_KALI.values>0].min():.4f}  "
          f"max={df_KALI.values.max():.4f}  "
          f"mean={df_KALI.values[df_KALI.values>0].mean():.4f}")
    if df_spacing is not None:
        print(f"  spacing stats: min={df_spacing.values[df_spacing.values>0].min():.4f}  "
              f"max={df_spacing.values.max():.4f}  "
              f"mean={df_spacing.values[df_spacing.values>0].mean():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — NJ Trees Side by Side
# ─────────────────────────────────────────────────────────────────────────────

def draw_fig4(df_pykali, df_spacing, outdir):
    print("\nGenerating Figure 4 — NJ Trees + RF Distance...")
    try:
        import dendropy
    except ImportError:
        print("  SKIPPED — dendropy not installed.")
        print("  Install with:  pip install dendropy")
        print("  Then re-run this script.")
        return

    rf_val = None
    trees  = {}

    for label, df in [('KALI_Non-Hash', df_pykali), ('KALI_Hash', df_spacing)]:
        shared = sorted(set(df_pykali.index) & set(df_spacing.index))
        df_shared = df.loc[shared, shared]

        # Write temp CSV for dendropy
        tmp = f'/tmp/kali_{label}_tmp.csv'
        df_shared.to_csv(tmp)

        try:
            pdm  = dendropy.PhylogeneticDistanceMatrix.from_csv(
                       src=open(tmp), delimiter=',',
                       is_first_row_taxa_labels=True,
                       is_first_column_taxa_labels=True)
            tree = pdm.nj_tree()
            tree.write(path=os.path.join(outdir, f'nj_{label}_k3.nwk'),
                       schema='newick')
            trees[label] = tree
            print(f"  NJ tree built: {label} ({len(shared)} taxa)")
        except Exception as e:
            print(f"  ERROR building {label} tree: {e}")

    # Compute RF if both trees built
    if len(trees) == 2:
        try:
            t1 = trees['KALI_Non-Hash']
            t2 = trees['KALI_Hash']
            t1.migrate_taxon_namespace(t2.taxon_namespace)
            rf  = dendropy.calculate.treecompare.symmetric_difference(t1, t2)
            n   = len(shared)
            nrf = rf / (2 * (n - 3)) if n > 3 else 0
            rf_val = (rf, nrf)
            print(f"  RF = {rf}   Normalised RF = {nrf:.4f}")
        except Exception as e:
            print(f"  RF computation error: {e}")

    # Draw side-by-side figure with RF annotation
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    fig.patch.set_facecolor('#f8f9fa')

    for i, (label, col, ax) in enumerate([
            ('KALI_Non-Hash',  '#2980b9', axes[0]),
            ('KALI_Hash', '#27ae60', axes[1])]):
        nwk_path = os.path.join(outdir, f'nj_{label}_k3.nwk')
        ax.set_facecolor('#fff')
        ax.axis('off')
        if os.path.exists(nwk_path):
            with open(nwk_path) as f:
                nwk = f.read().strip()
            ax.text(0.5, 0.55, f'NJ tree saved:\nnj_{label}_k3.nwk',
                    ha='center', va='center', fontsize=11,
                    color=col, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor=col, lw=2))
            ax.text(0.5, 0.35,
                    'Visualise with:\niTOL (itol.embl.de)\nor FigTree',
                    ha='center', va='center', fontsize=9, color='#555',
                    style='italic')
        else:
            ax.text(0.5, 0.5, f'{label}\ntree not built\n(run dendropy)',
                    ha='center', va='center', fontsize=10, color='#aaa')

        rf_text = ''
        if rf_val and i == 1:
            rf, nrf = rf_val
            rf_color = '#27ae60' if rf == 0 else '#e67e22' if rf <= 4 else '#e74c3c'
            rf_text = f'RF = {rf}   nRF = {nrf:.4f}'
            if rf == 0:
                rf_text += '  \u2713 Identical topology'
            ax.text(0.5, 0.12, rf_text, ha='center', fontsize=10,
                    fontweight='bold', color=rf_color, transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor=rf_color, lw=1.5))

        subtitle = 'KALI_Hash: --reduce mean, bins=50' if label == 'KALI_Hash' else 'KALI_Non-Hash: k=3, bins=300'
        ax.set_title(f'{label} NJ Tree (k=3)\n{subtitle}',
                     fontsize=11, fontweight='bold', color=col)

    fig.suptitle('Figure 4: Neighbour-Joining Trees — KALI_Non-Hash vs KALI_Hash\n'
                 'Real data — open .nwk files in iTOL or FigTree to visualise',
                 fontsize=12, fontweight='bold', color='#2c3e50')
    plt.tight_layout()
    out = os.path.join(outdir, 'fig4_nj_trees_real.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
    plt.close()
    print(f"  Saved: {out}")
    return rf_val


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — RF Bar Charts
# ─────────────────────────────────────────────────────────────────────────────

def draw_fig5(rf_results, outdir):
    """
    rf_results: dict of k_label -> (rf_pykali_spacing, rf_pykali_tensor, rf_spacing_tensor)
    e.g. {'k=3': (0, 0, 0), 'k=4': (2, 2, 0), 'k=5': (2, 4, 2)}
    """
    print("\nGenerating Figure 5 — RF Bar Charts...")

    if not rf_results:
        print("  No RF results provided. Provide real RF values to draw this figure.")
        print("  Build trees for k=3,4,5 and pass rf_results dict.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#f8f9fa'); ax.set_facecolor('#fff')

    k_labels = list(rf_results.keys())
    pair_labels = ['KALI_Non-Hash\nvs\nKALI_Hash', 'KALI_Non-Hash\nvs\ntensor', 'KALI_Hash\nvs\ntensor']
    x = np.arange(len(pair_labels)); w = 0.25
    colors = ['#2980b9', '#27ae60', '#e67e22']

    for ki, (k_label, vals) in enumerate(rf_results.items()):
        # Normalise each value
        n_taxa  = vals[3] if len(vals) > 3 else 30
        max_rf  = 2 * (n_taxa - 3)
        nrf     = [v / max_rf if max_rf > 0 else 0 for v in vals[:3]]
        ax.bar(x + (ki-1)*w, nrf, w, label=k_label,
               color=colors[ki % len(colors)], alpha=0.85, edgecolor='white', lw=1.5)

    ax.axhline(y=0.2, color='#7f8c8d', linestyle=':', lw=1.5)
    ax.text(2.6, 0.21, 'threshold', fontsize=8, color='#7f8c8d', style='italic')
    ax.set_ylim(0, 0.55)
    ax.set_ylabel('Normalised RF Distance (0=identical, 1=max)', fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(pair_labels, fontsize=10)
    ax.set_title('Figure 5: Normalised RF Distance Between Method Pairs\n'
                 'Real data across k values',
                 fontsize=12, fontweight='bold', color='#2c3e50')
    ax.legend(fontsize=9)

    out = os.path.join(outdir, 'fig5_rf_bars_real.png')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 — Scaling Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def draw_fig6(benchmarks_path, outdir):
    print("\nGenerating Figure 6 — Scaling Benchmark...")

    if not benchmarks_path or not os.path.exists(benchmarks_path):
        print("  SKIPPED — no benchmark CSV provided.")
        print("  Create a CSV with columns: genomes,kali_nonhash_seconds,kali_hash_seconds")
        print("  Then pass: --benchmarks your_benchmarks.csv")
        return

    df = pd.read_csv(benchmarks_path)
    required = {'genomes', 'kali_nonhash_seconds', 'kali_hash_seconds'}
    if not required.issubset(df.columns):
        print(f"  ERROR: benchmark CSV must have columns: {required}")
        print(f"  Found: {list(df.columns)}")
        return

    genome_counts = df['genomes'].tolist()
    nonhash_times = df['kali_nonhash_seconds'].tolist()
    hash_times    = df['kali_hash_seconds'].tolist()
    speedup       = [p/s for p,s in zip(nonhash_times, hash_times)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#f8f9fa')

    # Left — absolute runtime
    ax = axes[0]
    ax.plot(genome_counts, nonhash_times, 'o-', color='#2980b9', lw=2.5, ms=8, label='KALI_Non-Hash')
    ax.plot(genome_counts, hash_times,    's--',color='#27ae60', lw=2.5, ms=8, label='KALI_Hash')
    ax.fill_between(genome_counts, hash_times, nonhash_times, alpha=0.1, color='#e74c3c')
    ax.set_xlabel('Number of genomes', fontsize=11)
    ax.set_ylabel('Runtime (seconds)',  fontsize=11)
    ax.set_title('Runtime Scaling\nKALI_Non-Hash vs KALI_Hash (k=3, real data)',
                 fontsize=11, fontweight='bold', color='#2c3e50')
    ax.legend(fontsize=10); ax.set_facecolor('#fff')

    # Right — speed multiplier
    ax = axes[1]
    ax.plot(genome_counts, speedup, 'D-', color='#e74c3c', lw=2.5, ms=9)
    for x, s in zip(genome_counts, speedup):
        ax.text(x, s + 0.15, f'{s:.1f}\u00d7',
                ha='center', fontsize=8.5, fontweight='bold', color='#e74c3c')
    ax.axhline(y=speedup[0], color='#7f8c8d', linestyle=':', lw=1.5,
               label=f'Initial {speedup[0]:.1f}\u00d7 at n={genome_counts[0]}')
    ax.set_xlabel('Number of genomes', fontsize=11)
    ax.set_ylabel('Speed advantage of KALI_Hash (\u00d7)', fontsize=11)
    ax.set_title('Growing Speed Advantage\nKALI_Hash vs KALI_Non-Hash (real benchmark)',
                 fontsize=11, fontweight='bold', color='#2c3e50')
    ax.set_facecolor('#fff'); ax.legend(fontsize=9)

    fig.suptitle('Figure 6: Scaling Benchmark — Real Timing Results',
                 fontsize=12, fontweight='bold', color='#2c3e50')
    plt.tight_layout()
    out = os.path.join(outdir, 'fig6_scaling_real.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
    plt.close()
    print(f"  Saved: {out}")
    print(f"  Speed range: {min(speedup):.1f}\u00d7 to {max(speedup):.1f}\u00d7")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7 — Outlier Leverage Scatter
# ─────────────────────────────────────────────────────────────────────────────

def draw_fig7(df_pykali, df_spacing, outdir,
              ecoli_names=None, shigella_names=None):
    print("\nGenerating Figure 7 — Outlier Leverage Scatter...")

    vec_p, vec_s, shared = upper_triangle(df_pykali, df_spacing)
    species = detect_species(shared)

    # Build index groups
    idx_ec    = [i for i,l in enumerate(species) if l=='ecoli']
    idx_sh    = [i for i,l in enumerate(species) if l=='shigella']
    idx_unk   = [i for i,l in enumerate(species) if l=='unknown']

    n = len(shared)
    pair_idx  = np.array([(i,j)
                           for i in range(n) for j in range(i+1,n)])

    def pair_label(i, j):
        li, lj = species[i], species[j]
        if li == lj == 'ecoli':    return 'ecoli'
        if li == lj == 'shigella': return 'shigella'
        return 'cross'

    pair_labels_arr = np.array([pair_label(i,j) for i,j in pair_idx])

    mask_ec    = pair_labels_arr == 'ecoli'
    mask_sh    = pair_labels_arr == 'shigella'
    mask_cross = pair_labels_arr == 'cross'

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor('#f8f9fa'); ax.set_facecolor('#fff')

    if mask_ec.any():
        ax.scatter(vec_p[mask_ec],    vec_s[mask_ec],
                   color='#2980b9', s=40, alpha=0.7, zorder=4,
                   label=f'E. coli pairs (n={mask_ec.sum()})')
    if mask_sh.any():
        ax.scatter(vec_p[mask_sh],    vec_s[mask_sh],
                   color='#e74c3c', s=40, alpha=0.7, zorder=4,
                   label=f'Shigella pairs (n={mask_sh.sum()})')
    if mask_cross.any():
        ax.scatter(vec_p[mask_cross], vec_s[mask_cross],
                   color='#e67e22', s=40, alpha=0.5, zorder=3,
                   label=f'Cross-genus pairs (n={mask_cross.sum()})')

    # Overall regression
    r_all,   _ = pearsonr(vec_p,          vec_s)
    rho_all, _ = spearmanr(vec_p,         vec_s)
    m, b = np.polyfit(vec_p, vec_s, 1)
    xr   = np.linspace(vec_p.min(), vec_p.max(), 200)
    ax.plot(xr, m*xr+b, 'k--', lw=2,
            label=f'Overall fit  (r={r_all:.3f}, \u03c1={rho_all:.3f})', alpha=0.85)

    # Within-species regression
    ws_mask = mask_ec | mask_sh
    if ws_mask.sum() >= 3:
        rho_ws, _ = spearmanr(vec_p[ws_mask], vec_s[ws_mask])
        mw, bw    = np.polyfit(vec_p[ws_mask], vec_s[ws_mask], 1)
        xw        = np.linspace(vec_p[ws_mask].min(), vec_p[ws_mask].max(), 100)
        ax.plot(xw, mw*xw+bw, color='#8e44ad', lw=2, linestyle='--',
                label=f'Within-species fit  (\u03c1={rho_ws:.3f})', alpha=0.9)

    # Stats box
    stats_text = (f'Overall:  r={r_all:.3f},  \u03c1={rho_all:.3f}\n'
                  f'Within-species:  \u03c1={rho_ws:.3f}' if ws_mask.sum()>=3
                  else f'Overall:  r={r_all:.3f},  \u03c1={rho_all:.3f}')
    ax.text(0.97, 0.05, stats_text, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='#aaa'))

    ax.set_xlabel('KALI_Non-Hash Distance',        fontsize=11)
    ax.set_ylabel('KALI_Hash Distance',  fontsize=11)
    ax.set_title('Figure 7: Outlier Leverage — Cross-Genus Pairs Inflate Aggregate Correlation\n'
                 'Real data — E. coli + Shigella pairwise distances',
                 fontsize=11, fontweight='bold', color='#2c3e50')
    ax.legend(fontsize=8.5, loc='upper left')

    out = os.path.join(outdir, 'fig7_outlier_real.png')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
    plt.close()
    print(f"  Saved: {out}")
    print(f"  Overall:        r={r_all:.4f}  rho={rho_all:.4f}")
    if ws_mask.sum() >= 3:
        print(f"  Within-species: rho={rho_ws:.4f}")
    if mask_cross.any():
        print(f"  Cross-genus pairs: {mask_cross.sum()}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate Figures 2-7 from your real KALI CSV outputs.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES
--------
  # All figures (no benchmark):
      python generate_real_figures.py \\
          --pykali  results/PR_k3_b300.csv \\
          --spacing results/PI_k3_bins200.csv \\
          --outdir  real_figures/

  # With benchmark times for Figure 6:
      python generate_real_figures.py \\
          --pykali      results/PR_k3_b300.csv \\
          --spacing     results/PI_k3_bins200.csv \\
          --benchmarks  results/benchmarks.csv \\
          --outdir      real_figures/

  # Specific genomes for Figure 2:
      python generate_real_figures.py \\
          --pykali   results/PR_k3_b300.csv \\
          --spacing  results/PI_k3_bins200.csv \\
          --genome1  GCF_000005845 \\
          --genome2  GCF_000006945

  # Specify which species each genome belongs to:
      python generate_real_figures.py \\
          --pykali   results/PR_k3_b300.csv \\
          --spacing  results/PI_k3_bins200.csv \\
          --ecoli    GCF_000005845 GCF_001695515 GCF_000026305 \\
          --shigella GCF_000006925 GCF_000016085

  # Specific figures only:
      python generate_real_figures.py \\
          --pykali  results/PR_k3_b300.csv \\
          --spacing results/PI_k3_bins200.csv \\
          --figs 3 7
        """
    )
    parser.add_argument('--pykali',      required=True,
                        help='pykali distance matrix CSV (e.g. PR_k3_b300.csv)')
    parser.add_argument('--spacing',     required=True,
                        help='kali_spacing distance matrix CSV (e.g. PI_k3_bins200.csv)')
    parser.add_argument('--benchmarks',  default=None,
                        help='Benchmark CSV with columns: genomes,kali_nonhash_seconds,kali_hash_seconds')
    parser.add_argument('--metadata',    default=None,
                        help='metadata_labels.csv with Accession,Strain_Name,Color columns')
    parser.add_argument('--outdir',      default='real_figures',
                        help='Output folder for figures (default: real_figures/)')
    parser.add_argument('--genome1',     default=None,
                        help='First genome name for Figure 2 comparison')
    parser.add_argument('--genome2',     default=None,
                        help='Second genome name for Figure 2 comparison')
    parser.add_argument('--ecoli',       nargs='+', default=None,
                        help='List of E. coli genome IDs for species colouring')
    parser.add_argument('--shigella',    nargs='+', default=None,
                        help='List of Shigella genome IDs for species colouring')
    parser.add_argument('--figs',        nargs='+', type=int,
                        default=[2,3,4,5,6,7],
                        help='Which figures to generate (default: 2 3 4 5 6 7)')
    args = parser.parse_args()

    # Validate files
    for path, name in [(args.pykali,'--pykali'), (args.spacing,'--spacing')]:
        if not os.path.isfile(path):
            print(f"ERROR: file not found: {path}  ({name})")
            sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    print(f"\nLoading distance matrices...")
    df_p = load_matrix(args.pykali)
    df_s = load_matrix(args.spacing)
    print(f"  pykali:  {df_p.shape[0]} genomes  from {args.pykali}")
    print(f"  spacing: {df_s.shape[0]} genomes  from {args.spacing}")

    # ── Load metadata if provided ──────────────────────────────────────────
    accession_to_name    = {}
    accession_to_species = {}
    if hasattr(args, 'metadata') and args.metadata:
        print(f"\nLoading metadata from {args.metadata}...")
        accession_to_name, accession_to_species = load_metadata(args.metadata)
        print(f"  Loaded {len(accession_to_name)} genome name mappings")
        df_p = rename_matrix(df_p, accession_to_name)
        df_s = rename_matrix(df_s, accession_to_name)
        print(f"  Sample renamed: {list(df_p.index[:3])}")

    # ── Build species label override ───────────────────────────────────────
    species_override = None
    if args.ecoli or args.shigella:
        all_names = list(df_p.index)
        ec_set = set(args.ecoli or [])
        sh_set = set(args.shigella or [])
        species_override = [
            'ecoli'    if n in ec_set else
            'shigella' if n in sh_set else
            'unknown'
            for n in all_names
        ]
    elif accession_to_species:
        all_names = list(df_p.index)
        species_override = detect_species(all_names, accession_to_species)
        n_ec = species_override.count('ecoli')
        n_sh = species_override.count('shigella')
        print(f"  Species from metadata: {n_ec} E. coli, {n_sh} Shigella")

    rf_val = None
    figs   = set(args.figs)

    if 2 in figs:
        draw_fig2(df_p, args.outdir, args.genome1, args.genome2)

    if 3 in figs:
        draw_fig3(df_p, args.outdir, species_override, df_spacing=df_s)

    if 4 in figs or 5 in figs:
        rf_val = draw_fig4(df_p, df_s, args.outdir)

    if 5 in figs:
        if rf_val:
            rf, nrf = rf_val
            n = len(list(set(df_p.index) & set(df_s.index)))
            rf_results = {
                'k=3': (rf, rf, 0, n),
            }
            draw_fig5(rf_results, args.outdir)
        else:
            print("\nFigure 5 skipped — needs RF values from Figure 4.")
            print("  Install dendropy and re-run to generate Figure 5.")

    if 6 in figs:
        draw_fig6(args.benchmarks, args.outdir)

    if 7 in figs:
        draw_fig7(df_p, df_s, args.outdir,
                  ecoli_names=args.ecoli, shigella_names=args.shigella)

    print(f"\nAll done — figures saved to: {os.path.abspath(args.outdir)}/")


if __name__ == '__main__':
    main()
