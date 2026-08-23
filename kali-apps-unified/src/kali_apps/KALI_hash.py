#!/usr/bin/env python3
"""
KALI_hash.py — Spacing-histogram distance method for the KALI platform

Computes pairwise genome distances using k-mer SPACING DISTRIBUTION vectors.

WHY THIS METHOD
---------------
Standard k-mer frequency (pykali_hash.py / KALI_non-hash) captures HOW OFTEN each k-mer
appears. 


USAGE
-----
# Single k
python KALI_hash.py -g Ecoli/ -k 4 --bins 200 -o results/spacing

# Multiple k values — saves per-k matrices + combined
python KALI_hash.py -g Ecoli/ -k 3 4 5 --bins 200 --combine -o results/spacing

# Multi-k with vector concatenation (richer signal)
python KALI_hash.py -g Ecoli/ -k 3 4 5 --bins 200 --combine --combine-method concat -o results/spacing

ARGUMENTS
---------
  -g / --genome         folder, single file, or wildcard (e.g. Ecoli/*.fna)
  -k / --kmer           k-mer size(s) (default: 4). Accept multiple: -k 3 4 5
  --bins                histogram bins per k-mer (default: 200)
  --combine             when multiple k given, also save a combined matrix
  --combine-method      average  : average per-k distance matrices (default)
                        concat   : concatenate vectors before distance
                                   (captures multi-resolution signal)
  -d / --distance       cosine | euclidean | jaccard (default: cosine)
  -o / --output         output base path (default: spacing_out)
  -v / --verbose        print per-genome stats and timing

OUTPUTS
-------
  <output>_k<k>_bins<bins>.csv              per-k distance matrix
  <output>_k3-k5_bins<bins>_combined.csv    combined matrix (if --combine)

DEPENDENCIES
------------
  pip install biopython numpy pandas scipy
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy.spatial import distance

__version__ = "2.0.0"
DNA_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}


# ─────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────

def resolve_inputs(input_path: str) -> List[str]:
    """Accept a folder, single file, or glob wildcard."""
    if os.path.isdir(input_path):
        files: List[str] = []
        for ext in ("*.fasta", "*.fa", "*.fna", "*.fas", "*.ffn"):
            files.extend(glob.glob(os.path.join(input_path, ext)))
    else:
        files = glob.glob(input_path)
    files = sorted(set(files))
    if not files:
        raise ValueError(f"No FASTA files found at: {input_path}")
    return files


class Genome:
    def __init__(self, path: str) -> None:
        self.path = path
        self.name = Path(path).stem
        self.sequence = self._read(path)
        if not self.sequence:
            raise ValueError(f"No valid sequence in {path}")

    @staticmethod
    def _read(path: str) -> str:
        parts: List[str] = []
        for record in SeqIO.parse(path, "fasta"):
            parts.append(str(record.seq).upper())
        return re.sub(r"[^ATGCN]", "N", "".join(parts))


# ─────────────────────────────────────────────
# Core: rolling k-mer hash
# ─────────────────────────────────────────────

def rolling_kmer_hashes(sequence: str, k: int) -> np.ndarray:
    """
    Convert a DNA sequence to an integer hash array in O(n) time.
    Each k-mer maps to a unique value in [0, 4^k).
    N and non-ACGT characters reset the hash.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if len(sequence) < k:
        return np.array([], dtype=np.int32)

    hashes: List[int] = []
    base = 4 ** (k - 1)
    curr = 0
    run = 0

    for ch in sequence:
        val = DNA_MAP.get(ch)
        if val is None:
            curr = 0
            run = 0
            continue
        if run < k:
            curr = curr * 4 + val
            run += 1
            if run == k:
                hashes.append(curr)
        else:
            curr = (curr % base) * 4 + val
            hashes.append(curr)

    return np.array(hashes, dtype=np.int32)


# ─────────────────────────────────────────────
# Core: spacing histograms
# ─────────────────────────────────────────────

def get_spacings_by_kmer(
    hashes: np.ndarray,
    vector_size: int,
) -> Dict[int, np.ndarray]:
    """
    For each k-mer that appears in the hash array, return the array of
    inter-occurrence distances (positions between consecutive occurrences).

    """
    n = len(hashes)
    if n == 0:
        return {}

    order = np.argsort(hashes, kind="stable")
    sorted_h = hashes[order]
    sorted_pos = order.astype(np.float32)

    boundaries = np.concatenate(
        [[0], np.where(np.diff(sorted_h))[0] + 1, [n]]
    )

    result: Dict[int, np.ndarray] = {}
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        pos = np.sort(sorted_pos[s:e])
        kid = int(sorted_h[s])
        result[kid] = np.diff(pos) if len(pos) > 1 else np.array([float(n)])

    return result


def build_spacing_vectors(
    genomes: Sequence[Genome],
    k: int,
    bins: int,
    verbose: bool = False,
) -> np.ndarray:
    """
    Build one spacing-histogram vector per genome.

    
    Parameters
    ----------
    bins : int
        Histogram bins per k-mer (default: 200).
    """
    vector_size = 4 ** k
    t0 = time.perf_counter()

    # Pass 1 — collect all spacings and hash arrays
    if verbose:
        print(f"  Pass 1: computing k-mer hashes and spacings...")
    all_hashes = [rolling_kmer_hashes(g.sequence, k) for g in genomes]
    all_spacings = [get_spacings_by_kmer(h, vector_size) for h in all_hashes]

    if verbose:
        for g, h in zip(genomes, all_hashes):
            print(f"    {g.name}: {len(g.sequence):,} bp, {len(h):,} valid {k}-mers")

    # Pass 2 — find global bin range per k-mer (shared across all genomes)
    if verbose:
        print(f"  Pass 2: computing shared bin ranges across all genomes...")
    global_lo = np.zeros(vector_size, dtype=np.float32)
    global_hi = np.ones(vector_size, dtype=np.float32)

    for kmer_id in range(vector_size):
        parts = [sp[kmer_id] for sp in all_spacings if kmer_id in sp]
        if parts:
            combined = np.concatenate(parts)
            global_lo[kmer_id] = float(combined.min())
            global_hi[kmer_id] = float(combined.max()) + 1.0

    # Pass 3 — build histogram vectors with shared ranges
    if verbose:
        print(f"  Pass 3: building {bins}-bin spacing histograms per k-mer...")
    vectors: List[np.ndarray] = []
    for sp_dict in all_spacings:
        vec: List[np.ndarray] = []
        for kmer_id in range(vector_size):
            sp = sp_dict.get(kmer_id, np.array([global_lo[kmer_id]]))
            hist, _ = np.histogram(
                sp,
                bins=bins,
                range=(global_lo[kmer_id], global_hi[kmer_id]),
            )
            hist = hist.astype(np.float32)
            if hist.sum() > 0:
                hist /= hist.sum()
            vec.append(hist)
        vectors.append(np.stack(vec))   # shape: (4^k, bins)

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"  Vector build: {elapsed:.2f}s  |  "
              f"vector size: {vector_size} k-mers × {bins} bins = {vector_size * bins} features")

    # shape: (n_genomes, 4^k, bins)
    return np.stack(vectors)


# ─────────────────────────────────────────────
# Distance matrix
# ─────────────────────────────────────────────

def spacing_distance_matrix(
    genomes: Sequence[Genome],
    k: int,
    bins: int,
    metric: str,
    reduce: str = "concat",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Build pairwise distance matrix using spacing-histogram vectors.

    reduce="concat"  : concatenate all motif histograms, one pdist (original behaviour)
    reduce="mean"    : pdist per motif, mean across motifs   (pykali-equivalent)
    reduce="median"  : pdist per motif, median across motifs (robust to outlier motifs)
    """
    labels = [g.name for g in genomes]

    # shape: (n_genomes, 4^k, bins)
    vectors_3d = build_spacing_vectors(genomes, k=k, bins=bins, verbose=verbose)
    n_genomes, n_motifs, _ = vectors_3d.shape

    t0 = time.perf_counter()

    if reduce == "concat":
        vectors_2d = vectors_3d.reshape(n_genomes, -1)
        condensed = distance.pdist(vectors_2d, metric=metric)
    else:
        all_dists = []
        for motif_id in range(n_motifs):
            motif_vecs = vectors_3d[:, motif_id, :]
            all_dists.append(distance.pdist(motif_vecs, metric))
        dists_array = np.array(all_dists)
        if reduce == "mean":
            condensed = np.mean(dists_array, axis=0)
        else:
            condensed = np.median(dists_array, axis=0)

    square = distance.squareform(condensed)
    elapsed = time.perf_counter() - t0

    if verbose:
        print(f"  Distance computation ({metric}, reduce={reduce}): {elapsed:.3f}s")

    return pd.DataFrame(square, index=labels, columns=labels)


# ─────────────────────────────────────────────
# Multi-k combine
# ─────────────────────────────────────────────

def combine_by_average(
    matrices: List[pd.DataFrame],
) -> pd.DataFrame:
    """
    Average multiple distance matrices element-wise.

    Same approach as pykali_hash.py --combine: run spacing at k=3, k=4, k=5
    separately, then average. Each k captures different resolution:
      k=3 → global composition   (64 k-mers)
      k=4 → local patterns       (256 k-mers)
      k=5 → specific sequences   (1024 k-mers)
    Averaging blends all three signals equally.
    """
    labels = matrices[0].index
    stacked = np.stack([m.to_numpy(dtype=float) for m in matrices], axis=0)
    return pd.DataFrame(stacked.mean(axis=0), index=labels, columns=labels)


def combine_by_concat(
    genomes: Sequence[Genome],
    k_list: List[int],
    bins: int,
    metric: str,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Concatenate spacing vectors across all k values, then compute distance once.

    More powerful than averaging because the distance is computed on the
    full multi-resolution fingerprint rather than averaged over separate
    single-resolution distances.

    Vector size = sum(4^k × bins for k in k_list)
    e.g. k=[3,4,5], bins=200: (64+256+1024)×200 = 268,800 features

    k=3 catches global composition shifts
    k=4 catches local motif clustering  (pykali default)
    k=5 catches fine-grained sequence patterns
    All three combined gives a richer fingerprint than any single k.
    """
    labels = [g.name for g in genomes]
    all_vecs: List[np.ndarray] = [np.array([]) for _ in genomes]

    for k in k_list:
        if verbose:
            print(f"  Building vectors for k={k}...")
        vecs = build_spacing_vectors(genomes, k=k, bins=bins, verbose=verbose)
        for i in range(len(genomes)):
            if all_vecs[i].size == 0:
                all_vecs[i] = vecs[i]
            else:
                all_vecs[i] = np.concatenate([all_vecs[i], vecs[i]])

    matrix = np.vstack(all_vecs)
    if verbose:
        print(f"  Combined vector size: {matrix.shape[1]} features")
        print(f"  Computing pairwise {metric} distances...")

    t0 = time.perf_counter()
    condensed = distance.pdist(matrix, metric=metric)
    square = distance.squareform(condensed)
    if verbose:
        print(f"  Distance computation: {time.perf_counter()-t0:.3f}s")

    return pd.DataFrame(square, index=labels, columns=labels)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "KALI_hash.py v2 — spacing-histogram genome distance\n\n"
            "Captures the spatial distribution of k-mer occurrences across\n"
            "the genome. Supports multiple k values with --combine.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "-g", "--genome", required=True,
        help="Folder, FASTA file, or wildcard (e.g. Ecoli/*.fna)",
    )
    p.add_argument(
        "-k", "--kmer", type=int, nargs="+", default=[4],
        help="k-mer size(s). Single: -k 4  Multiple: -k 3 4 5 (default: 4)",
    )
    p.add_argument(
        "--bins", type=int, default=200,
        help="Histogram bins per k-mer (default: 200).",
    )
    p.add_argument(
        "--combine", action="store_true",
        help="When multiple k given, also produce a combined distance matrix.",
    )
    p.add_argument(
        "--combine-method", choices=["average", "concat"], default="average",
        help="How to combine multiple k values (default: average). "
             "average: average per-k distance matrices. "
             "concat: concatenate vectors then compute distance once.",
    )
    p.add_argument(
        "-r", "--reduce",
        choices=["concat", "mean", "median"],
        default="concat",
        help="How to combine motif histograms: "
             "concat (default, one pdist on full vector), "
             "mean (pdist per motif then mean — pykali-equivalent), "
             "median (pdist per motif then median — robust to outlier motifs)",
    )
    p.add_argument(
        "-d", "--distance",
        choices=["cosine", "euclidean", "jaccard"],
        default="cosine",
        help="Pairwise distance metric (default: cosine)",
    )
    p.add_argument(
        "-o", "--output", default="spacing_out",
        help="Output base path (default: spacing_out)",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print per-genome stats and timing",
    )
    p.add_argument("--version", action="version", version=f"KALI_hash {__version__}")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_total = time.perf_counter()

    # ── Load genomes ───────────────────────────────────────────
    files = resolve_inputs(args.genome)
    print(f"Found {len(files)} genome file(s)")

    genomes = []
    for f in files:
        try:
            g = Genome(f)
            genomes.append(g)
        except Exception as e:
            print(f"  Warning: skipping {f} — {e}", file=sys.stderr)

    if len(genomes) < 2:
        print("Error: need at least 2 valid genomes.", file=sys.stderr)
        sys.exit(1)

    k_list = sorted(set(args.kmer))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(genomes)} genomes  |  "
          f"k={k_list}  bins={args.bins}  metric={args.distance}  reduce={args.reduce}"
          + (f"  combine={args.combine_method}" if args.combine and len(k_list) > 1 else "")
          + "\n")

    # ── Per-k matrices ─────────────────────────────────────────
    per_k_dfs: List[pd.DataFrame] = []

    for k in k_list:
        print(f"--- k={k} ---")
        df = spacing_distance_matrix(
            genomes,
            k=k,
            bins=args.bins,
            metric=args.distance,
            reduce=args.reduce,
            verbose=args.verbose,
        )
        per_k_dfs.append(df)

        csv_name = f"{out_path}_k{k}_bins{args.bins}.csv"
        df.to_csv(csv_name)
        print(f"Saved: {csv_name}")

        if args.verbose:
            labels = list(df.index)
            n = len(labels)
            pairs = [
                (labels[i], labels[j], float(df.iloc[i, j]))
                for i in range(n) for j in range(i + 1, n)
            ]
            pairs.sort(key=lambda x: x[2])
            print("  Top-5 closest:")
            for g1, g2, d in pairs[:5]:
                print(f"    {g1} ↔ {g2}: {d:.6f}")
        print()

    # ── Combined matrix ────────────────────────────────────────
    if args.combine and len(k_list) > 1:
        k_range = f"k{k_list[0]}-k{k_list[-1]}"
        print(f"--- combined ({k_range}, method={args.combine_method}) ---")

        if args.combine_method == "average":
            combined_df = combine_by_average(per_k_dfs)
            print(f"  Averaged {len(k_list)} distance matrices")
        else:  # concat
            combined_df = combine_by_concat(
                genomes, k_list, args.bins, args.distance, args.verbose
            )

        combined_name = f"{out_path}_{k_range}_bins{args.bins}_combined.csv"
        combined_df.to_csv(combined_name)
        print(f"Saved: {combined_name}")
        print()

    # ── Summary ────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_total
    print(f"Total time: {elapsed:.1f}s  |  "
          f"genomes: {len(genomes)}  |  "
          f"pairs: {len(genomes) * (len(genomes) - 1) // 2}")


if __name__ == "__main__":
    main()
