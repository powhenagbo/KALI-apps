#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, itertools, os, re, time
import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy.spatial import distance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import gzip as _gzip

def _kali_non_hash_file_format(path: str) -> str:
    p = path.lower()
    if p.endswith(".gz"):
        p = p[:-3]
    return "fastq" if p.endswith((".fastq", ".fq")) else "fasta"

def _kali_non_hash_open(path: str):
    return _gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")


class Genome:
    def __init__(self, path, name=""):
        fmt = _kali_non_hash_file_format(path)
        with _kali_non_hash_open(path) as fh:
            self.records = tuple(SeqIO.parse(fh, fmt))
        if not self.records:
            raise ValueError(f"No sequence records found in {path}")
        self.name = name or self.records[0].id

    def restrict(self, motif):
        fragments, pattern = [], re.compile(motif, re.I)
        for record in self.records:
            seq, begin = str(record.seq), 0
            for m in pattern.finditer(seq):
                fragments.append(m.start() - begin)
                begin = m.end()
            fragments.append(len(seq) - begin)
        return fragments


class GenomeSet:
    """Ordered collection of Genome objects.

    Preserves insertion order so that genome labels and distance-matrix
    rows/columns always correspond to the same sequence.  The previous
    implementation subclassed ``set``, whose iteration order is
    non-deterministic, which could silently misalign labels with distances.
    """

    def __init__(self):
        self._list: list = []
        self._seen: set  = set()

    def add(self, genome) -> None:
        if genome.name not in self._seen:
            self._list.append(genome)
            self._seen.add(genome.name)

    def __iter__(self):
        return iter(self._list)

    def __len__(self):
        return len(self._list)

    def names(self):
        return tuple(g.name for g in self._list)

    def electrophorese(self, motif, bins):
        frags = [g.restrict(motif) for g in self._list]
        rng = (min(map(np.min, frags)), max(map(np.max, frags)))
        lanes = [np.histogram(f, bins=bins, range=rng) for f in frags]
        labels = self.names()
        result = type('Bands', (), {
            'labels': labels,
            'lanes': np.array([l[0] for l in lanes]),
            'bin_edges': np.round(lanes[0][1])
        })()
        return result

    def distance_matrix(self, motif, bins, metric):
        bands = self.electrophorese(motif, bins)
        return distance.pdist(bands.lanes, metric)


def resolve_inputs(path):
    # All extensions that KALI_non-hash can open via Biopython (fasta/fastq, plain and gzipped)
    EXTS = ("*.fasta", "*.fa", "*.fna", "*.fas", "*.ffn",
            "*.fastq", "*.fq",
            "*.fasta.gz", "*.fa.gz", "*.fna.gz", "*.fastq.gz", "*.fq.gz")
    if os.path.isdir(path):
        flat = sorted(f for e in EXTS for f in glob.glob(os.path.join(path, e)))
    else:
        flat = sorted(glob.glob(path))
    if not flat:
        raise ValueError(
            "No sequence files found. Supported: "
            ".fasta .fa .fna .fas .ffn .fastq .fq (and .gz compressed versions)"
        )
    return flat


def vprint(verbose, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)


def save_band_plot(bands, out_path):
    """Save electrophoresis band heatmap with contrast-stretched shading.

    Raw fragment-count data is heavily right-skewed (a handful of bins carry
    most of the mass), so a plain linear scale from min->max crushes nearly
    everything to one end of the colormap and the real banding pattern
    disappears. We fix that two ways:
      - PowerNorm(gamma<1) stretches the low/mid range so subtle differences
        become visible instead of being flattened near white.
      - vmax is clipped to the 98th percentile (not the raw max) so one or
        two outlier bins don't compress the rest of the scale.
    bands.lanes rows=genomes, columns=bin edges (right edges).
    """
    from matplotlib.colors import PowerNorm

    df = pd.DataFrame(bands.lanes, index=bands.labels, columns=bands.bin_edges[1:])

    vals = df.values.astype(float)
    vmax = np.percentile(vals, 98) if vals.max() > 0 else 1.0
    vmax = max(vmax, 1e-9)  # guard against an all-zero matrix

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(df.index))))
    sns.heatmap(
        df, cmap='viridis', ax=ax,
        norm=PowerNorm(gamma=0.3, vmin=0, vmax=vmax),
        cbar_kws={'label': 'fragment count'},
    )
    ax.set_xlabel("fragment length bin (bp)")
    ax.set_ylabel("genome")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_gel_plot(bands, out_path, top_n=15, sample_n=None, seed=42):
    """Render restriction fragments as a simulated agarose gel image.

    Each genome becomes a lane; each of its top_n most abundant fragment-
    length bins becomes a band. Migration distance is log-scaled in fragment
    length (small fragments migrate farther, matching a real gel), and band
    darkness/thickness reflects fragment count at that length, normalized
    against the global max count so lanes stay comparable to one another.

    Note: unlike a real gel (one distinct fragment per band), each genome
    here produces hundreds-to-thousands of fragments across the whole
    chromosome for a single motif, binned by length. Showing every non-empty
    bin would render as a near-solid smear in the short-fragment region, so
    this view keeps only the top_n bins per genome by count -- a readable,
    presentation-style summary, not the full underlying distribution.
    For the complete, non-truncated picture use save_band_plot() (heatmap).

    """
    bin_edges = np.asarray(bands.bin_edges, dtype=float)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    full_lanes = np.asarray(bands.lanes, dtype=float)
    full_labels = list(bands.labels)
    full_n_genomes = full_lanes.shape[0]

    # global vmax computed BEFORE any subsampling so intensity stays
    # comparable to a full, unsampled run of the same data
    vmax = full_lanes.max() if full_lanes.max() > 0 else 1.0

    if sample_n is not None and sample_n < full_n_genomes:
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(full_n_genomes, size=sample_n, replace=False))
        lanes = full_lanes[sel]
        labels = [full_labels[j] for j in sel]
        sampled_note = f"  (random {sample_n} of {full_n_genomes} genomes, seed={seed})"
    else:
        lanes = full_lanes
        labels = full_labels
        sampled_note = ""

    n_genomes, n_bins = lanes.shape

    # Select each genome's top_n bands FIRST, so the axis range below can be
    # zoomed to where the data actually lands -- not the full histogram range.
    # (Fixing the earlier issue where lo/hi came from bin_centers.min()/.max(),
    # which can span orders of magnitude beyond what top_n actually plots,
    # wasting most of the gel as empty space when only short fragments dominate.)
    per_genome_keep = []
    plotted_bin_idx = set()
    for i in range(n_genomes):
        row = lanes[i]
        keep = np.argsort(row)[::-1][:top_n]
        keep = keep[row[keep] > 0]
        per_genome_keep.append(keep)
        plotted_bin_idx.update(keep.tolist())

    if plotted_bin_idx:
        plotted_lengths = bin_centers[sorted(plotted_bin_idx)]
        data_lo, data_hi = plotted_lengths.min(), plotted_lengths.max()
    else:
        data_lo, data_hi = bin_centers.min(), bin_centers.max()

    # Pad the zoomed range by 15% in log-space on each side so the extreme
    # bands don't sit flush against the well/dye-front edges, and collapse to
    # a small fixed window if every plotted band happens to be the same length.
    if data_hi > data_lo:
        pad = (np.log10(data_hi) - np.log10(data_lo)) * 0.15
    else:
        pad = 0.3
    lo = max(10 ** (np.log10(max(data_lo, 1e-9)) - pad), 1.0)
    hi = 10 ** (np.log10(max(data_hi, 1e-9)) + pad)
    gel_len = 15.0  # cm, purely cosmetic scale to read like a real gel

    # A band's thickness maxes out at 0.12 + 0.10 = 0.22 cm (see the loop below),
    # so its half-thickness (0.11 cm) is the largest amount any band can extend
    # past its center point. Without a margin, a band centered exactly at frac=0
    # or frac=1 would have half its thickness rendered outside the lane
    # rectangle (y ∈ [0, gel_len]) and appear clipped. Reserving that same
    # half-thickness as margin on both ends guarantees every band -- even at
    # maximum thickness -- stays fully inside the lane, regardless of gel_len.
    max_thickness = 0.12 + 0.10
    margin_frac = (max_thickness / 2) / gel_len

    def migration_from_top(length_bp):
        # 0 = well (large fragments barely move), gel_len = dye front (small fragments)
        frac_small = 1 - (np.log10(np.clip(length_bp, lo, hi)) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
        frac_small = margin_frac + frac_small * (1 - 2 * margin_frac)
        return frac_small * gel_len

    lane_w, gap, well_h = 0.7, 0.35, 0.25
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * n_genomes), 8))
    bg, lane_bg, band_color = "#0d2a44", "#dbe9f7", (0.13, 0.09, 0.30)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    for i in range(n_genomes):
        x0 = i * (lane_w + gap)
        ax.add_patch(plt.Rectangle((x0, 0), lane_w, gel_len, facecolor=lane_bg, edgecolor="none", zorder=1))
        ax.add_patch(plt.Rectangle((x0, -0.5), lane_w, well_h, facecolor="#b8cbe0", edgecolor="none", zorder=2))
        ax.text(x0 + lane_w / 2, -0.65, labels[i], ha="center", va="top",
                fontsize=9, color="white", fontweight="bold", rotation=0)

        # Fragment count = lanes[i].sum() (histogram is a complete partition of
        # that genome's fragments, so this equals the raw fragment count from
        # restrict()). A genome the motif never cuts returns exactly one
        # fragment (the whole sequence), so total == 1 unambiguously flags
        # "no cut site found" -- distinct from merely having few visible bands.
        total_fragments = int(row_sum) if (row_sum := lanes[i].sum()) else 0
        if total_fragments <= 1:
            ax.text(x0 + lane_w / 2, -1.05, "no cut", ha="center", va="top",
                    fontsize=7.5, color="#ff6b6b", fontstyle="italic")
        else:
            ax.text(x0 + lane_w / 2, -1.05, f"{total_fragments-1:,} cuts", ha="center", va="top",
                    fontsize=7.5, color="#9fb3cc", fontstyle="italic")

        row = lanes[i]
        keep = per_genome_keep[i]
        for b in keep:
            count = row[b]
            y = migration_from_top(bin_centers[b])
            intensity = 0.15 + 0.75 * (count / vmax)
            thickness = 0.12 + 0.10 * (count / vmax)
            ax.add_patch(plt.Rectangle(
                (x0 + 0.05, y - thickness / 2), lane_w - 0.10, thickness,
                facecolor=band_color, alpha=min(intensity, 1.0), edgecolor="none", zorder=3
            ))

    # reference ladder ticks on the y-axis, labeled by fragment length rather than raw cm
    ladder_vals = [v for v in [10, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700,
                                1000, 1500, 2000, 3000, 5000, 10000]
                   if lo <= v <= hi]
    if len(ladder_vals) < 3:
        ladder_vals = np.geomspace(lo, hi, 6).round().astype(int).tolist()
    ax.set_yticks([migration_from_top(v) for v in ladder_vals])
    ax.set_yticklabels([f"{v:,} bp" for v in ladder_vals], color="white", fontsize=9)

    ax.set_xlim(-0.4, n_genomes * (lane_w + gap))
    ax.set_ylim(gel_len + 1.0, -1.4)
    ax.set_ylabel("Fragment length (migration)", color="white", fontsize=11)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_title(f"Simulated Restriction Fragment Gel  (top {top_n} bins/genome){sampled_note}",
                 color="white", fontsize=13, fontweight="bold", pad=14)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def save_html_report(matrix_df, out_path, k, bins, metric, reduce, label=None):
    """Generate a self-contained HTML heatmap report from a distance matrix."""
    labels    = list(matrix_df.index)
    data_rows = matrix_df.values.tolist()
    vals      = matrix_df.values
    max_val   = float(vals[vals > 0].max()) if (vals > 0).any() else 1.0
    n         = len(labels)
    js_labels = str(labels)
    js_data   = str(data_rows)
    k_label   = label if label else f"k={k}"
    motif_str = f"{4**k:,}" if k > 0 else "combined"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>KALI Restriction Fragment {k_label} bins={bins}</title>
<style>
  body  {{ font-family: Arial, sans-serif; margin: 2rem; color: #1a1a1a; background: #fff; }}
  h1    {{ font-size: 1.2rem; font-weight: 600; margin-bottom: 0.3rem; }}
  .sub  {{ font-size: 0.85rem; color: #666; margin-bottom: 1.5rem; }}
  .cards{{ display: grid; grid-template-columns: repeat(5, auto); gap: 10px; margin-bottom: 1.5rem; width: fit-content; }}
  .card {{ background: #f5f5f3; border-radius: 8px; padding: 0.6rem 1rem; }}
  .cl   {{ font-size: 11px; color: #888; margin-bottom: 3px; }}
  .cv   {{ font-size: 1.2rem; font-weight: 600; }}
  .wrap {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; font-size: 11px; }}
  th, td{{ border: 0.5px solid #ddd; padding: 5px 8px; text-align: center; white-space: nowrap; }}
  th    {{ background: #f5f5f3; font-weight: 500; color: #555; }}
  th.rh {{ text-align: left; min-width: 140px; max-width: 220px; overflow: hidden; text-overflow: ellipsis; }}
  td.diag{{ background: #f5f5f3; color: #aaa; }}
  .legend{{ margin-top: 10px; font-size: 11px; color: #888; display: flex; align-items: center; gap: 8px; }}
  .swatch{{ width: 100px; height: 10px; border-radius: 3px;
             background: linear-gradient(to right, rgba(83,74,183,0.65), rgba(83,74,183,0.05)); }}
</style>
</head>
<body>
<h1>KALI restriction fragment distance matrix</h1>
<p class="sub">{k_label} &nbsp;|&nbsp; bins = {bins} &nbsp;|&nbsp; metric = {metric} &nbsp;|&nbsp; reduce = {reduce} &nbsp;|&nbsp; sequences = {n} &nbsp;|&nbsp; motifs = {motif_str}</p>
<div class="cards">
  <div class="card"><div class="cl">Sequences</div><div class="cv">{n}</div></div>
  <div class="card"><div class="cl">k / range</div><div class="cv">{k_label}</div></div>
  <div class="card"><div class="cl">Bins</div><div class="cv">{bins}</div></div>
  <div class="card"><div class="cl">Metric</div><div class="cv">{metric}</div></div>
  <div class="card"><div class="cl">Reduce</div><div class="cv">{reduce}</div></div>
</div>
<div class="wrap"><table><thead><tr id="hrow"></tr></thead><tbody id="tbody"></tbody></table></div>
<div class="legend"><span>Similar</span><div class="swatch"></div><span>Distant</span></div>
<script>
const labels = {js_labels};
const data   = {js_data};
const maxVal = {max_val};
const hrow = document.getElementById('hrow');
const th0  = document.createElement('th'); th0.className='rh'; hrow.appendChild(th0);
labels.forEach(l => {{
  const th = document.createElement('th');
  th.textContent = l.length > 20 ? l.slice(0,18)+'\u2026' : l;
  th.title = l; hrow.appendChild(th);
}});
function cellColor(v) {{
  return 'background:rgba(83,74,183,' + ((1-v/maxVal)*0.65).toFixed(2) + ')';
}}
const tbody = document.getElementById('tbody');
data.forEach((row, i) => {{
  const tr = document.createElement('tr');
  const th = document.createElement('th'); th.className='rh'; th.title=labels[i];
  th.textContent = labels[i].length > 28 ? labels[i].slice(0,26)+'\u2026' : labels[i];
  tr.appendChild(th);
  row.forEach((v, j) => {{
    const td = document.createElement('td');
    if (i===j) {{ td.className='diag'; td.textContent='0.000000'; }}
    else {{ td.textContent = v.toFixed(6); td.style.cssText = cellColor(v); }}
    tr.appendChild(td);
  }});
  tbody.appendChild(tr);
}});
</script>
</body></html>"""
    with open(out_path, "w") as fh:
        fh.write(html)

def main():
    p = argparse.ArgumentParser(description="In silico restriction fragment distance matrices")
    p.add_argument("-g", "--genome",    required=True, help="Folder or wildcard to FASTA files")
    p.add_argument("-k", "--kmer_list", nargs='+', type=int, default=[3,4,5])
    p.add_argument("-b", "--bin",       type=int, default=50,
                   help="Number of histogram bins for fragment length distribution (default: 50)")
    p.add_argument("-d", "--distance",  default="cosine",
                   choices=["cosine","euclidean","jaccard"],
                   help="Distance metric (default: cosine)")
    p.add_argument("-r", "--reduce",    default="mean",
                   choices=["mean", "median"],
                   help="Reduction method across motifs: mean or median (default: mean)")
    p.add_argument("-o", "--output",    default="kali_output",
                   help="Output base path")
    p.add_argument("--combine",         action="store_true",
                   help="Average all per-k RE matrices into one combined matrix")
    p.add_argument("-p", "--plot",      action="store_true",
                   help="Save electrophoresis band heatmap PNG for each k")
    p.add_argument("--gel",             action="store_true",
                   help="Render the band plot as a simulated gel image (top-N bands per lane) instead of a heatmap")
    p.add_argument("--top-n",           type=int, default=15,
                   help="Bands shown per lane in --gel mode (default: 15)")
    p.add_argument("--sample-genomes",  type=int, default=None,
                   help="In --gel mode, randomly show only this many genomes (lanes) instead of all of them, for readability with large sets")
    p.add_argument("--seed",            type=int, default=42,
                   help="Random seed used for --sample-genomes (default: 42, for reproducible sampling)")
    p.add_argument("-v", "--verbose",   action="store_true")
    args = p.parse_args()
    v = args.verbose
    t0 = time.time()

    vprint(v, f"\n{'='*70}\nKALI - In Silico Restriction Fragment Analysis\n{'='*70}")
    vprint(v, f"\nConfig: genomes={args.genome} | k={args.kmer_list} | bins={args.bin} | dist={args.distance} | reduce={args.reduce}")

    files = resolve_inputs(args.genome)
    vprint(v, f"\n[Step 1] Loading {len(files)} genomes...")

    genomeset = GenomeSet()
    for f in files:
        g = Genome(f)
        genomeset.add(g)
        vprint(v, f"  Loaded: {os.path.basename(f)} ({len(g.records)} seq)")

    names = genomeset.names()
    vprint(v, f"  Genomes: {', '.join(names)}")

    output_files, total_motifs = [], 0
    all_avg_matrices = []   # accumulate per-k condensed distance arrays for combined matrix
    vprint(v, f"\n[Step 2] Processing k-mers...")

    for k in args.kmer_list:
        t1 = time.time()
        motifs = [''.join(pm) for pm in itertools.product('ATGC', repeat=k)]
        vprint(v, f"\n  k={k} | {len(motifs):,} motifs", end="", flush=True)

        dists = []
        last_bands = None
        for i, motif in enumerate(motifs):
            if v and i % 100 == 0:
                print(f"\r  k={k} | {i}/{len(motifs)} ({100*i/len(motifs):.0f}%)", end="", flush=True)
            last_bands = genomeset.electrophorese(motif, args.bin)
            dists.append(distance.pdist(last_bands.lanes, args.distance))

        # --- reduce across all motifs: mean or median ---
        dists_array = np.array(dists)
        if args.reduce == "mean":
            avg = np.mean(dists_array, axis=0)
        else:  # median
            avg = np.median(dists_array, axis=0)

        all_avg_matrices.append(avg)

        out      = f"{args.output}_k{k}_b{args.bin}.csv"
        html_out = f"{args.output}_k{k}_b{args.bin}.html"
        matrix_df = pd.DataFrame(distance.squareform(avg), columns=names, index=names)
        matrix_df.to_csv(out)
        save_html_report(matrix_df, html_out, k, args.bin, args.distance, args.reduce)

        # --- optional electrophoresis band plot (last motif's bands, same as zip) ---
        if args.plot and last_bands is not None:
            if args.gel:
                band_png = f"{args.output}_k{k}_b{args.bin}_gel.png"
                save_gel_plot(last_bands, band_png, top_n=args.top_n,
                               sample_n=args.sample_genomes, seed=args.seed)
                vprint(v, f"\n  Gel plot → {band_png}")
            else:
                band_png = f"{args.output}_k{k}_b{args.bin}_bands.png"
                save_band_plot(last_bands, band_png)
                vprint(v, f"\n  Band plot → {band_png}")
            output_files.append(band_png)

        total_motifs += len(motifs)
        output_files.append(out)
        k_time = time.time() - t1
        vprint(v, f"\r  k={k} done in {k_time:.1f}s | dist range [{avg.min():.4f}, {avg.max():.4f}] → {out}")

    # Combined matrix — only when --combine is passed
    if args.combine and len(all_avg_matrices) > 1:
        if args.reduce == "mean":
            combined_avg = np.mean(all_avg_matrices, axis=0)
        else:
            combined_avg = np.median(all_avg_matrices, axis=0)
        k_range       = f"k{args.kmer_list[0]}-k{args.kmer_list[-1]}"
        combined_csv  = f"{args.output}_{k_range}_b{args.bin}_combined.csv"
        combined_html = f"{args.output}_{k_range}_b{args.bin}_combined.html"
        combined_df   = pd.DataFrame(distance.squareform(combined_avg), columns=names, index=names)
        combined_df.to_csv(combined_csv)
        save_html_report(combined_df, combined_html,
                         k=0, bins=args.bin, metric=args.distance,
                         reduce=args.reduce, label=k_range)
        output_files.append(combined_csv)
        vprint(v, f"\n  Combined matrix ({k_range}) → {combined_csv}")

    total = time.time() - t0
    if v:
        print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
        print(f"  Motifs processed : {total_motifs:,}")
        print(f"  Reduce method    : {args.reduce}")
        print(f"  Files generated  : {len(output_files)}")
        print(f"  Total time       : {total:.2f}s")
        print(f"  Avg time per k   : {total/len(args.kmer_list):.2f}s")
        for f in output_files:
            print(f"  ✓ {f} ({os.path.getsize(f):,} bytes)")
        print(f"{'='*70}\nCOMPLETE\n{'='*70}")
    else:
        print(f"\nProcessed k={', '.join(map(str, args.kmer_list))} | reduce={args.reduce} | Total time: {total:.2f}s")
        for f in output_files:
            if f.endswith(".csv"):
                html_f = f[:-4] + ".html"
                print(f"  - {f}")
                print(f"  - {html_f}")
            else:
                print(f"  - {f}")


if __name__ == "__main__":
    main()
