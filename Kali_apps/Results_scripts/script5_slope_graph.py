"""
script5_slope_graph.py
───────────────────────
Slope graph (parallel coordinates) for any two distance matrices.

Each line connects the same pairwise distance from matrix A (left axis)
to matrix B (right axis).

  Horizontal line  →  that sample pair is EQUAL in both matrices
  Upward slope     →  matrix B distance is LARGER for that pair
  Downward slope   →  matrix A distance is LARGER for that pair
  Steep slope      →  large disagreement
  Flat / clustered →  matrices are equivalent

Lines are coloured by the size of the difference:
  Blue (cool)  →  small difference  (near horizontal)
  Red  (warm)  →  large difference  (steep slope)

Usage
─────
  python script5_slope_graph.py matA.csv matB.csv
  python script5_slope_graph.py PR_k5_b200.csv PR_k3_b50.csv --name-a PI_5_bins200 --name-b PI_k3_bins50
  python script5_slope_graph.py matA.csv matB.csv --top 50   # show only 50 most-different pairs
  python script5_slope_graph.py                               # synthetic demo

Accepted formats: .csv  .tsv  .txt  .npy  .npz  .xlsx  .phylip  .phy
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy import stats

from kali_data import load_pair, synthetic_pair


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="script5_slope_graph",
        description="Slope graph — horizontal = equal, slope = different.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("matrix_a", nargs="?", help="First matrix file  (demo if omitted)")
    p.add_argument("matrix_b", nargs="?", help="Second matrix file")
    p.add_argument("--labels",  metavar="FILE", help="One label per line")
    p.add_argument("--name-a",  default="Matrix A", help="Label for left axis")
    p.add_argument("--name-b",  default="Matrix B", help="Label for right axis")
    p.add_argument("--top",     type=int, default=None,
                   help="Show only the N most-different pairs (default: all)")
    p.add_argument("--alpha",   type=float, default=None,
                   help="Line transparency (default: auto based on N pairs)")
    p.add_argument("--out",     default="slope_graph.png", help="Output file")
    p.add_argument("--dpi",     type=int, default=150, help="Image DPI")
    return p


def main() -> None:
    args = build_parser().parse_args()

    # ── Load ──────────────────────────────────────────────────────────────────
    if args.matrix_a and args.matrix_b:
        mat_a, mat_b, labels = load_pair(args.matrix_a, args.matrix_b, args.labels)
        na = args.name_a if args.name_a != "Matrix A" else args.matrix_a
        nb = args.name_b if args.name_b != "Matrix B" else args.matrix_b
    elif args.matrix_a or args.matrix_b:
        print("[error] Provide both files, or neither for demo.", file=sys.stderr)
        sys.exit(1)
    else:
        print("[demo] Using synthetic data.", file=sys.stderr)
        mat_a, mat_b, labels = synthetic_pair()
        na, nb = "Matrix A (demo)", "Matrix B (demo)"

    n   = mat_a.shape[0]
    idx = np.triu_indices(n, k=1)
    a_vals = mat_a[idx]
    b_vals = mat_b[idx]
    diffs  = a_vals - b_vals
    abs_d  = np.abs(diffs)

    # ── Optionally keep only top-N most different pairs ───────────────────────
    order = np.argsort(abs_d)[::-1]
    if args.top is not None and args.top < len(order):
        order    = order[:args.top]
        a_vals   = a_vals[order]
        b_vals   = b_vals[order]
        diffs    = diffs[order]
        abs_d    = abs_d[order]
        subtitle = f"Top {args.top} most-different pairs  (of {len(idx[0])} total)"
    else:
        subtitle = f"All {len(idx[0])} pairwise distances  ·  n={n} samples"

    N_lines = len(a_vals)
    alpha   = args.alpha if args.alpha is not None else max(0.03, min(0.6, 40 / N_lines))

    # ── Colour map: blue (small diff) → red (large diff) ─────────────────────
    norm     = mcolors.Normalize(vmin=0, vmax=abs_d.max() or 1)
    cmap     = cm.get_cmap("coolwarm")
    colours  = [cmap(norm(d)) for d in abs_d]

    # ── Statistics ────────────────────────────────────────────────────────────
    n_horizontal = int(np.sum(abs_d < 1e-6))           # exactly equal
    n_slope_up   = int(np.sum(diffs < -1e-6))           # B > A
    n_slope_down = int(np.sum(diffs >  1e-6))           # A > B
    mean_diff    = float(diffs.mean())
    pct_equal    = 100 * n_horizontal / N_lines

    # Bias size relative to the data's own scale (not an absolute cutoff —
    # KALI distances live in the ~0-0.01 range, so a fixed threshold like
    # 0.05 would never trigger regardless of the real relationship)
    data_scale = float(np.mean(np.concatenate([a_vals, b_vals])))
    rel_bias   = abs(mean_diff) / data_scale if data_scale > 0 else 0.0

    # Sign test: is the direction (A>B vs B>A) consistent across pairs,
    # or roughly 50/50 as you'd expect from pure noise?
    n_directional = n_slope_down + n_slope_up
    if n_directional > 0:
        sign_p = stats.binomtest(
            max(n_slope_down, n_slope_up), n_directional, p=0.5
        ).pvalue
    else:
        sign_p = 1.0

    print(f"Total pairs     : {N_lines}")
    print(f"Horizontal (=)  : {n_horizontal}  ({pct_equal:.1f}%)")
    print(f"Slope down (A>B): {n_slope_down}")
    print(f"Slope up   (B>A): {n_slope_up}")
    print(f"Mean bias (A−B) : {mean_diff:+.4f}  ({rel_bias*100:.1f}% of mean value)")
    print(f"Max |diff|      : {abs_d.max():.4f}")
    print(f"Sign test p     : {sign_p:.3e}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    DARK  = "#0d1117"
    TEXT  = "#c9d1d9"
    MUTED = "#8b949e"

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor("#0d1117")

    # Draw one line per pair
    for i in range(N_lines):
        ax.plot(
            [0, 1],
            [a_vals[i], b_vals[i]],
            color    = colours[i],
            alpha    = alpha,
            linewidth= 0.8,
            solid_capstyle="round",
        )

    # Axis labels (left = A, right = B)
    ax.set_xlim(-0.15, 1.15)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([na, nb], fontsize=13, fontweight="bold", color=TEXT)

    # Vertical axis lines
    ax.axvline(0, color="#334155", linewidth=1.5, zorder=0)
    ax.axvline(1, color="#334155", linewidth=1.5, zorder=0)

    ax.set_ylabel("Pairwise Distance", color=MUTED, labelpad=8)
    ax.tick_params(axis="y", colors=MUTED, labelsize=8)
    ax.tick_params(axis="x", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(axis="y", color="#1e293b", linewidth=0.5, zorder=0)

    # Colour bar
    sm  = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb  = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.6, aspect=30)
    cb.set_label("|A − B|  difference", color=MUTED, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8)

    # Stats box
    # Verdict grounded in the sign test (is there a real, consistent direction?)
    # and the bias size relative to the data's own scale (is it big enough to
    # matter?) — rather than an absolute cutoff that ignores data scale.
    if sign_p < 0.05 and rel_bias > 0.01:
        verdict = "Matrices differ systematically"
    elif sign_p < 0.05:
        verdict = "Consistent but small bias"
    else:
        verdict = "Matrices are statistically equivalent"

    ax.text(0.5, 1.01,
            f"Horizontal = equal  ·  Slope = different  ·  {verdict}",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9, color=MUTED, style="italic")

    ax.text(0.02, 0.98,
            f"Mean bias (A−B): {mean_diff:+.4f}  ({rel_bias*100:.1f}%)\n"
            f"Max |diff|     : {abs_d.max():.4f}\n"
            f"Slope down (A>B): {n_slope_down}\n"
            f"Slope up   (B>A): {n_slope_up}\n"
            f"Sign test p    : {sign_p:.2e}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=9, color="#22d3ee", family="monospace",
            bbox=dict(facecolor="#0f172a", edgecolor="#1e3a5f", pad=6, alpha=0.9))

    ax.set_title(
        f"Slope Graph  ·  {na}  vs  {nb}\n"
        f"{subtitle}",
        color=TEXT, fontsize=13, fontweight="bold", pad=14,
    )

    plt.tight_layout()
    plt.savefig(args.out, dpi=args.dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
