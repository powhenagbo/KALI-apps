#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import gzip
import os
import time
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy.spatial import distance

DNA_MAP = {
    'A': 0,
    'C': 1,
    'G': 2,
    'T': 3,
}
VALID_BASES = set(DNA_MAP)


def log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg)


class Genome:
    def __init__(self, name: str, sequence: str) -> None:
        self.name = name
        self.sequence = sequence.upper()
        if not self.sequence:
            raise ValueError(f"Empty sequence for {name}")

    @classmethod
    def from_file_genome(cls, path: str) -> "Genome":
        """One genome per file — join all records. Supports FASTA/FASTQ and .gz."""
        fmt = _file_format(path)
        stem = Path(path).stem.replace(".fastq","").replace(".fasta","").replace(".fq","").replace(".fa","")
        with _open_file(path) as fh:
            seq = "".join(str(r.seq) for r in SeqIO.parse(fh, fmt))
        return cls(stem, seq)

    @classmethod
    def from_file_metagenomics(cls, path: str) -> List["Genome"]:
        """One record per sequence — for metagenomic contigs/reads. Supports FASTA/FASTQ and .gz."""
        fmt  = _file_format(path)
        stem = Path(path).stem.replace(".fastq","").replace(".fasta","").replace(".fq","").replace(".fa","")
        genomes = []
        with _open_file(path) as fh:
            for record in SeqIO.parse(fh, fmt):
                name = f"{stem}::{record.id}"
                genomes.append(cls(name, str(record.seq)))
        return genomes


FASTA_EXTS  = {".fasta", ".fa", ".fna", ".fas", ".ffn"}
FASTQ_EXTS  = {".fastq", ".fq"}
COMPRESSED  = {".gz"}

def _file_format(path: str) -> str:
    """Detect whether a file is FASTA or FASTQ, handling .gz compression."""
    p = path.lower()
    if p.endswith(".gz"):
        p = p[:-3]
    ext = os.path.splitext(p)[1]
    if ext in FASTQ_EXTS:
        return "fastq"
    return "fasta"


def _open_file(path: str):
    """Return a file handle — transparent gzip support."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def _resolve_single(input_path: str) -> List[str]:
    """Resolve one path entry — directory, wildcard, or single file."""
    all_exts = FASTA_EXTS | FASTQ_EXTS
    gz_exts  = {e + ".gz" for e in all_exts}

    if os.path.isdir(input_path):
        files = []
        for ext in sorted(all_exts | gz_exts):
            files += glob.glob(os.path.join(input_path, "*" + ext))
    else:
        files = glob.glob(input_path)
    return files


def resolve_inputs(input_paths: List[str]) -> List[str]:
    """Accept one or more paths — directories, wildcards, or explicit files."""
    files = []
    for p in input_paths:
        files += _resolve_single(p)

    files = sorted(set(files))
    if not files:
        raise ValueError(
            "No sequence files found. Supported: "
            ".fasta .fa .fna .fas .ffn .fastq .fq (and .gz compressed versions)"
        )
    return files


def rolling_kmer_hashes(sequence: str, k: int) -> np.ndarray:
    """
    Convert a DNA sequence into base-4 k-mer hash values.
    Uses numpy stride tricks — supports k up to 9 efficiently.
    Any k-mer containing a non-ACGT character is skipped.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if k > 9:
        raise ValueError(
            f"k={k} requires vector_size={4**k:,} which exceeds the supported maximum. "
            f"Use k ≤ 9 for pykali_hash.py full vectors."
        )
    if len(sequence) < k:
        return np.array([], dtype=np.int32)

    # Map bases to 0-3, everything else to -1
    _bmap = np.full(128, -1, dtype=np.int8)
    for ch, val in [('A',0),('C',1),('G',2),('T',3),
                    ('a',0),('c',1),('g',2),('t',3)]:
        _bmap[ord(ch)] = val

    arr     = np.frombuffer(sequence.encode('ascii', errors='replace'), dtype=np.uint8)
    encoded = _bmap[arr]   # shape: (L,)

    L = len(encoded)
    if L < k:
        return np.array([], dtype=np.int32)

    # Build k-mer matrix via stride tricks: shape (L-k+1, k)
    shape   = (L - k + 1, k)
    strides = (encoded.strides[0], encoded.strides[0])
    kmers   = np.lib.stride_tricks.as_strided(encoded, shape=shape, strides=strides)

    # Drop k-mers with any ambiguous base
    valid_mask = (kmers >= 0).all(axis=1)
    kmers      = kmers[valid_mask]

    if kmers.shape[0] == 0:
        return np.array([], dtype=np.int32)

    # Compute base-4 hash: each k-mer → integer index
    powers  = (4 ** np.arange(k - 1, -1, -1)).astype(np.int64)
    hashes  = (kmers.astype(np.int64) * powers).sum(axis=1)

    return hashes.astype(np.int32)


def block_vectors_from_hashes(
    hashes: np.ndarray,
    vector_size: int,
    bin_size: int,
    normalize: str = "block",
) -> np.ndarray:
    """
    Build one vector per block.

    normalize:
      - block : divide each block vector by its own valid k-mer count
      - none  : keep raw counts
    """
    if bin_size < 1:
        raise ValueError("bin_size must be >= 1")

    if hashes.size == 0:
        return np.zeros((1, vector_size), dtype=np.float32)

    vectors: List[np.ndarray] = []
    for start in range(0, len(hashes), bin_size):
        block = hashes[start:start + bin_size]
        counts = np.bincount(block, minlength=vector_size).astype(np.float32)
        if normalize == "block" and len(block) > 0:
            counts /= float(len(block))
        vectors.append(counts)

    return np.vstack(vectors)


def reduce_blocks(block_matrix: np.ndarray, method: str) -> np.ndarray:
    if method == "mean":
        return block_matrix.mean(axis=0)
    if method == "median":
        return np.median(block_matrix, axis=0)
    raise ValueError("method must be 'mean' or 'median'")


def build_genome_vector(
    genome: Genome,
    k: int,
    bin_size: int,
    reduce_method: str,
    normalize: str,
    verbose: bool,
) -> np.ndarray:
    log(f"Reading genome: {genome.name}", verbose)
    hashes = rolling_kmer_hashes(genome.sequence, k)
    log(f"  Sequence length: {len(genome.sequence):,}", verbose)
    log(f"  Valid {k}-mers: {len(hashes):,}", verbose)

    vector_size = 4 ** k
    blocks = block_vectors_from_hashes(hashes, vector_size, bin_size, normalize)
    log(f"  Number of blocks: {blocks.shape[0]:,}", verbose)

    final_vector = reduce_blocks(blocks, reduce_method).astype(np.float32)
    return final_vector


def compute_distance_matrix(
    labels: Sequence[str],
    vectors: np.ndarray,
    metric: str,
) -> pd.DataFrame:
    condensed = distance.pdist(vectors, metric=metric)
    square = distance.squareform(condensed)
    return pd.DataFrame(square, index=labels, columns=labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standardized block-based k-mer pipeline based on the PowerPoint logic: "
            "read genomes, hash k-mers, group into blocks, reduce block vectors, "
            "and compute pairwise genome distances."
        )
    )
    parser.add_argument(
        "-g", "--genome", required=True, nargs='+',
        help="One or more folders, files, or wildcard paths (e.g. -g genomes/ reads/*.fq.gz)"
    )
    parser.add_argument(
        "-k", "--kmer", type=int, nargs='+', default=[3],
        help="One or more k-mer sizes (e.g. -k 3 4 5). Max k=9 (vector_size=4^k, up to 262,144)."
    )
    parser.add_argument(
        "-b", "--bin-size", type=int, default=50,
        help="Number of k-mers per block (default: 50)"
    )
    parser.add_argument(
        "-r", "--reduce", choices=["mean", "median"], default="mean",
        help="How to reduce block vectors into one genome vector"
    )
    parser.add_argument(
        "-d", "--distance", choices=["cosine", "euclidean", "jaccard"], default="cosine",
        help="Distance metric for pairwise genome comparison"
    )
    parser.add_argument(
        "-n", "--normalize", choices=["block", "none"], default="block",
        help="Normalize each block by its own size, or keep raw counts"
    )
    parser.add_argument(
        "-m", "--mode", choices=["genome", "metagenomics"], default="genome",
        help="genome: one vector per file | metagenomics: one vector per FASTA record"
    )
    parser.add_argument(
        "-o", "--output", default="kali_pipeline_output",
        help="Base name for output CSVs — one file per k (e.g. kali_pipeline_output_k3.csv)"
    )
    parser.add_argument(
        "--vectors-output", default=None,
        help="Optional base name to save per-k genome vectors (e.g. vectors_k3.csv)"
    )
    parser.add_argument(
        "--combine", action="store_true",
        help="Average all per-k distance matrices into one combined matrix"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print step-by-step progress"
    )
    return parser.parse_args()



def save_html_report(distance_df: pd.DataFrame, out_path: str,
                     k: int, bin_size: int, metric: str, mode: str,
                     label: str = None) -> None:
    """Generate a self-contained HTML heatmap report from a distance matrix."""
    labels = list(distance_df.index)
    data_rows = distance_df.values.tolist()
    vals = distance_df.values
    max_val = float(vals[vals > 0].max()) if (vals > 0).any() else 1.0
    n = len(labels)
    k_label = label if label else f"k={k}"

    # Build JS arrays
    js_labels = str(labels)
    js_data   = str(data_rows)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>KALI Distance Matrix — {k_label} b={bin_size}</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 2rem; color: #1a1a1a; background: #fff; }}
  h1   {{ font-size: 1.2rem; font-weight: 600; margin-bottom: 0.3rem; }}
  .sub {{ font-size: 0.85rem; color: #666; margin-bottom: 1.5rem; }}
  .cards {{ display: grid; grid-template-columns: repeat(4, auto); gap: 10px; margin-bottom: 1.5rem; width: fit-content; }}
  .card  {{ background: #f5f5f3; border-radius: 8px; padding: 0.6rem 1rem; }}
  .cl    {{ font-size: 11px; color: #888; margin-bottom: 3px; }}
  .cv    {{ font-size: 1.2rem; font-weight: 600; }}
  .wrap  {{ overflow-x: auto; }}
  table  {{ border-collapse: collapse; font-size: 11px; }}
  th, td {{ border: 0.5px solid #ddd; padding: 5px 8px; text-align: center; white-space: nowrap; }}
  th     {{ background: #f5f5f3; font-weight: 500; color: #555; }}
  th.rh  {{ text-align: left; min-width: 140px; max-width: 200px; overflow: hidden; text-overflow: ellipsis; }}
  td.diag{{ background: #f5f5f3; color: #aaa; }}
  .legend{{ margin-top: 10px; font-size: 11px; color: #888; display: flex; align-items: center; gap: 8px; }}
  .swatch{{ width: 100px; height: 10px; border-radius: 3px;
            background: linear-gradient(to right, rgba(29,158,117,0.65), rgba(29,158,117,0.05)); }}
</style>
</head>
<body>
<h1>KALI distance matrix</h1>
<p class="sub">k = {k} &nbsp;|&nbsp; block size = {bin_size} &nbsp;|&nbsp; metric = {metric} &nbsp;|&nbsp; mode = {mode} &nbsp;|&nbsp; sequences = {n}</p>

<div class="cards">
  <div class="card"><div class="cl">Sequences</div><div class="cv">{n}</div></div>
  <div class="card"><div class="cl">k-mer size</div><div class="cv">{k}</div></div>
  <div class="card"><div class="cl">Block size</div><div class="cv">{bin_size}</div></div>
  <div class="card"><div class="cl">Metric</div><div class="cv">{metric}</div></div>
</div>

<div class="wrap"><table><thead><tr id="header-row"></tr></thead><tbody id="tbody"></tbody></table></div>
<div class="legend"><span>Similar</span><div class="swatch"></div><span>Distant</span></div>

<script>
const labels  = {js_labels};
const data    = {js_data};
const maxVal  = {max_val};

const hdr = document.getElementById('header-row');
const th0 = document.createElement('th'); th0.className = 'rh'; hdr.appendChild(th0);
labels.forEach(l => {{
  const th = document.createElement('th');
  th.textContent = l.length > 20 ? l.slice(0,18) + '…' : l;
  th.title = l;
  hdr.appendChild(th);
}});

function cellColor(v) {{
  const t = v / maxVal;
  return 'background:rgba(29,158,117,' + ((1-t)*0.65).toFixed(2) + ')';
}}

const tbody = document.getElementById('tbody');
data.forEach((row, i) => {{
  const tr = document.createElement('tr');
  const th = document.createElement('th');
  th.className = 'rh'; th.title = labels[i];
  th.textContent = labels[i].length > 28 ? labels[i].slice(0,26) + '…' : labels[i];
  tr.appendChild(th);
  row.forEach((v, j) => {{
    const td = document.createElement('td');
    if (i === j) {{ td.className = 'diag'; td.textContent = '0.000000'; }}
    else {{ td.textContent = v.toFixed(6); td.style.cssText = cellColor(v); }}
    tr.appendChild(td);
  }});
  tbody.appendChild(tr);
}});
</script>
</body></html>"""

    with open(out_path, "w") as fh:
        fh.write(html)

def main() -> None:
    args = parse_args()
    t0   = time.time()

    files = resolve_inputs(args.genome)
    log(f"Found {len(files)} genome file(s)", args.verbose)

    if args.mode == "genome":
        genomes = [Genome.from_file_genome(path) for path in files]
    else:
        genomes = [g for path in files for g in Genome.from_file_metagenomics(path)]
    labels = [g.name for g in genomes]
    log(f"Mode: {args.mode} | Sequences loaded: {len(genomes)}", args.verbose)

    all_distance_dfs: List[pd.DataFrame] = []
    output_files: List[str] = []

    for k in args.kmer:
        t1 = time.time()
        log(f"\n--- Processing k={k} ---", args.verbose)

        vectors = []
        for genome in genomes:
            vec = build_genome_vector(
                genome=genome,
                k=k,
                bin_size=args.bin_size,
                reduce_method=args.reduce,
                normalize=args.normalize,
                verbose=args.verbose,
            )
            vectors.append(vec)

        vector_matrix = np.vstack(vectors)
        distance_df   = compute_distance_matrix(labels, vector_matrix, args.distance)
        all_distance_dfs.append(distance_df)

        out_path  = f"{args.output}_k{k}_b{args.bin_size}.csv"
        html_path = f"{args.output}_k{k}_b{args.bin_size}.html"
        distance_df.to_csv(out_path)
        save_html_report(distance_df, html_path, k, args.bin_size, args.distance, args.mode)
        output_files.append(out_path)

        k_time = time.time() - t1
        print(f"Saved distance matrix: {out_path}  ({k_time:.1f}s)")
        print(f"Saved HTML report:     {html_path}")

        if args.vectors_output:
            vectors_df = pd.DataFrame(vector_matrix, index=labels)
            vec_path   = f"{args.vectors_output}_k{k}_b{args.bin_size}.csv"
            vectors_df.to_csv(vec_path)
            print(f"Saved genome vectors:  {vec_path}")

    # Combined matrix — only when --combine is passed and multiple k values used
    if args.combine and len(all_distance_dfs) > 1:
        combined_vals = np.mean([df.values for df in all_distance_dfs], axis=0)
        combined_df   = pd.DataFrame(combined_vals, index=labels, columns=labels)
        k_range       = f"k{args.kmer[0]}-k{args.kmer[-1]}"
        combined_csv  = f"{args.output}_{k_range}_b{args.bin_size}_combined.csv"
        combined_html = f"{args.output}_{k_range}_b{args.bin_size}_combined.html"
        combined_df.to_csv(combined_csv)
        save_html_report(combined_df, combined_html,
                         k=0, bin_size=args.bin_size,
                         metric=args.distance, mode=args.mode,
                         label=k_range)
        output_files.append(combined_csv)
        print(f"Saved combined matrix: {combined_csv}")
        print(f"Saved combined HTML:   {combined_html}")

    total = time.time() - t0
    print(f"\nTotal time: {total:.2f}s  |  avg per k: {total/len(args.kmer):.2f}s")


if __name__ == "__main__":
    main()
