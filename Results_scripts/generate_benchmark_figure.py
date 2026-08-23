import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

CSV_PATH = os.path.expanduser("~/Desktop/Virus/kali_app/projects/benchmark_timing.csv")
OUT_DIR  = os.path.expanduser("~/Desktop/Virus/kali_app/projects/new")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
print("Tool names found:", df['tool'].unique().tolist())

spacing = df[df['tool'].isin(['KALI_Hash', 'Pykali_Interval', 'kali_spacing'])].copy().sort_values('bins')
pykali  = df[df['tool'].isin(['KALI_Non-Hash', 'pykali_Restriction', 'pykali'])].copy().sort_values('bins')

shared_bins = sorted(set(spacing['bins'].tolist()) & set(pykali['bins'].tolist()))
print(f"Shared bins: {shared_bins}")

spacing = spacing[spacing['bins'].isin(shared_bins)].reset_index(drop=True)
pykali  = pykali[pykali['bins'].isin(shared_bins)].reset_index(drop=True)

speed_advantage = pykali['total'].values / spacing['total'].values
print(f"Speed advantage: {[f'{v:.0f}x' for v in speed_advantage]}")

# ── Figure 5: Both in seconds, log scale, same axis ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Figure 11: Computational Performance\n"
    "KALI_Non-Hash vs KALI_Hash | n=30 genomes, k=3-7, cosine distance",
    fontsize=13, fontweight='bold', y=1.01
)

ax1 = axes[0]
ax1.plot(spacing['bins'], spacing['total'],
         'o-', color='#27ae60', linewidth=2.2, markersize=7,
         label='KALI_Hash')
ax1.plot(pykali['bins'], pykali['total'],
         's--', color='#e74c3c', linewidth=2.2, markersize=7,
         label='KALI_Non-Hash')

for _, row in spacing.iterrows():
    ax1.annotate(f"{row['total']:.0f}s",
                 xy=(row['bins'], row['total']),
                 xytext=(0, 6), textcoords='offset points',
                 ha='center', fontsize=7.5, color='#27ae60')

for _, row in pykali.iterrows():
    ax1.annotate(f"{row['total']/3600:.1f}h",
                 xy=(row['bins'], row['total']),
                 xytext=(0, 6), textcoords='offset points',
                 ha='center', fontsize=7.5, color='#e74c3c')

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Histogram Bin Count', fontsize=12)
ax1.set_ylabel('Total Runtime (seconds, log scale)', fontsize=12)
ax1.set_title('A. Total Runtime (k=3-7 combined)\nBoth in seconds — log scale', fontsize=11, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3, which='both')

ax2 = axes[1]
bars = ax2.bar(range(len(shared_bins)), speed_advantage,
               color='#2980b9', alpha=0.8, edgecolor='#1a5276', linewidth=0.8)
for bar, val in zip(bars, speed_advantage):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 2,
             f'{val:.0f}x',
             ha='center', va='bottom', fontsize=10,
             fontweight='bold', color='#1a5276')
ax2.set_xticks(range(len(shared_bins)))
ax2.set_xticklabels([str(b) for b in shared_bins], fontsize=10)
ax2.set_xlabel('Histogram Bin Count', fontsize=12)
ax2.set_ylabel('Speed Advantage (KALI_Non-Hash / KALI_Hash)', fontsize=12)
ax2.set_title('B. Speed Advantage of KALI_Hash over KALI_Non-Hash',
              fontsize=11, fontweight='bold')
ax2.axhline(y=np.mean(speed_advantage), color='#e74c3c', linestyle='--',
            linewidth=1.5, label=f'Mean: {np.mean(speed_advantage):.0f}x')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, max(speed_advantage) * 1.15)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig5_benchmark.png"), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, "fig5_benchmark.pdf"), bbox_inches='tight')
plt.close()
print("Saved fig5_benchmark.png")

# ── Figure 6: Per-k — both in seconds, log scale ──────────────────────────
k_cols   = ['k3', 'k4', 'k5', 'k6', 'k7']
k_labels = ['k=3\n(64)', 'k=4\n(256)', 'k=5\n(1,024)', 'k=6\n(4,096)', 'k=7\n(16,384)']

sp50 = spacing[spacing['bins'] == 50].iloc[0]
py50 = pykali[pykali['bins'] == 50].iloc[0]

sp_times = [sp50[k] for k in k_cols]
py_times = [py50[k] for k in k_cols]

fig2, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(k_cols))
w = 0.35

b1 = ax.bar(x - w/2, sp_times, w,
            color='#27ae60', alpha=0.85, edgecolor='#1a5276',
            linewidth=0.8, label='KALI_Hash (seconds)')
b2 = ax.bar(x + w/2, py_times, w,
            color='#e74c3c', alpha=0.85, edgecolor='#922b21',
            linewidth=0.8, label='KALI_Non-Hash (seconds)')

for bar, t in zip(b1, sp_times):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() * 1.15,
            f'{t:.0f}s',
            ha='center', va='bottom', fontsize=9,
            color='#1e8449', fontweight='bold')

for bar, t in zip(b2, py_times):
    label = f'{t:.0f}s' if t < 3600 else f'{t/3600:.1f}h'
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() * 1.15,
            label,
            ha='center', va='bottom', fontsize=9,
            color='#922b21', fontweight='bold')

for i, (s, p) in enumerate(zip(sp_times, py_times)):
    ax.annotate(f'{p/s:.0f}x\nfaster',
                xy=(i, np.sqrt(s * p)),
                ha='center', fontsize=9, color='#2980b9', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#eaf4fb', alpha=0.8))

ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(k_labels, fontsize=11)
ax.set_xlabel('k value (number of motifs in parentheses)', fontsize=12)
ax.set_ylabel('Runtime in seconds (log scale)', fontsize=12)
ax.set_title(
    'Figure 12: Per-k Runtime Comparison\n'
    'KALI_Non-Hash vs KALI_Hash | n=30 genomes, bins=50 | Both axes in seconds',
    fontsize=12, fontweight='bold'
)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y', which='both')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig6_per_k_runtime.png"), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, "fig6_per_k_runtime.pdf"), bbox_inches='tight')
plt.close()
print("Saved fig6_per_k_runtime.png")
print("All done.")
