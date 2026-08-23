import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

df = pd.read_csv('/Users/pauloa/Desktop/Virus/kali_app/projects/outputs/pi_pr_matched_rf.csv')

# Separate individual k rows from combined
df_indiv = df[~df['k'].str.contains('combined')].copy()
df_comb  = df[df['k'].str.contains('combined')].copy()

k_order  = ['k3', 'k4', 'k5', 'k6', 'k7']
bins_order = [50, 100, 200, 300, 500, 700, 1000]
colors = {
    'k3': '#e74c3c',
    'k4': '#e67e22',
    'k5': '#2ecc71',
    'k6': '#2980b9',
    'k7': '#8e44ad',
}

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor('#f8f9fa')

# ── Left panel: Line chart — nRF by bin size, one line per k ─────────────────
ax = axes[0]
ax.set_facecolor('#ffffff')

for k in k_order:
    sub = df_indiv[df_indiv['k'] == k].sort_values('bins')
    ax.plot(sub['bins'], sub['RF'], marker='o', lw=2.2, ms=7,
            color=colors[k], label=k, zorder=4)
    # Annotate RF=0 points
    for _, row in sub.iterrows():
        if row['RF'] == 0.0:
            ax.annotate('RF=0', xy=(row['bins'], 0.005),
                        fontsize=7, color=colors[k], ha='center',
                        arrowprops=dict(arrowstyle='->', color=colors[k], lw=0.8),
                        xytext=(row['bins'], 0.04))

ax.axhline(0.10, color='#7f8c8d', ls=':', lw=1.5, alpha=0.7)
ax.text(1010, 0.105, 'nRF = 0.10', fontsize=8, color='#7f8c8d', va='bottom')
ax.axhline(0.0,  color='#27ae60', ls='--', lw=1.2, alpha=0.5)

ax.set_xlabel('Histogram Bin Count', fontsize=12)
ax.set_ylabel('Normalised Robinson-Foulds Distance', fontsize=12)
ax.set_title('A. nRF by Bin Size — Individual k Values\n(lower = more topologically similar)', 
             fontsize=11, fontweight='bold', color='#2c3e50')
ax.set_xticks(bins_order)
ax.set_xticklabels([str(b) for b in bins_order], fontsize=9)
ax.set_ylim(-0.02, 0.65)
ax.legend(title='k value', fontsize=10, title_fontsize=10, loc='upper right')
ax.grid(axis='y', alpha=0.3)

# ── Right panel: grouped bar — mean nRF per k, with min/max error ─────────
ax = axes[1]
ax.set_facecolor('#ffffff')

means = df_indiv.groupby('k')['RF'].mean().reindex(k_order)
mins  = df_indiv.groupby('k')['RF'].min().reindex(k_order)
maxs  = df_indiv.groupby('k')['RF'].max().reindex(k_order)

x = np.arange(len(k_order))
bars = ax.bar(x, means, color=[colors[k] for k in k_order],
              alpha=0.85, edgecolor='white', lw=1.5, width=0.55, zorder=3)

# Error bars (min to max)
ax.errorbar(x, means,
            yerr=[means - mins, maxs - means],
            fmt='none', color='#2c3e50', capsize=5, lw=1.8, zorder=5)

# Value labels
for i, (k, m) in enumerate(zip(k_order, means)):
    ax.text(i, m + 0.005, f'{m:.3f}', ha='center', va='bottom',
            fontsize=9, fontweight='bold', color=colors[k])

# Mark combined k3-k7
comb_mean = df_comb['RF'].mean()
ax.axhline(comb_mean, color='#c0392b', ls='--', lw=2, zorder=4)
ax.text(4.55, comb_mean + 0.005, f'Combined k3–k7\nmean={comb_mean:.3f}',
        fontsize=8, color='#c0392b', va='bottom', ha='right')

ax.set_xticks(x)
ax.set_xticklabels([k.upper() for k in k_order], fontsize=11)
ax.set_xlabel('k-mer Length', fontsize=12)
ax.set_ylabel('Mean Normalised RF Distance', fontsize=12)
ax.set_title('B. Mean nRF per k Value\n(error bars = min/max range across bin sizes)',
             fontsize=11, fontweight='bold', color='#2c3e50')
ax.set_ylim(0, 0.45)
ax.grid(axis='y', alpha=0.3)

fig.suptitle(
    'Figure 8c: Robinson-Foulds Topological Agreement — KALI_Hash vs KALI_Non-hash\n'
    'n=30 genomes (18 E. coli + 12 Shigella), cosine distance, reduce=mean',
    fontsize=13, fontweight='bold', color='#2c3e50', y=1.01
)

plt.tight_layout()
plt.savefig('/Users/pauloa/Desktop/Virus/kali_app/projects/outputs//fig5_rf_agreement.png',
            dpi=180, bbox_inches='tight', facecolor='#f8f9fa')
plt.close()
print("Saved: fig5_rf_agreement.png")
