#!/usr/bin/env python3
"""
kali_tensor.py — Phase 5 of the KALI platform

Multi-scale distance tensor for genomic analysis.

CONCEPT
-------


METHODS
-------
1. Build distance matrix D_k for each k value (cosine distance)
2. Stack into tensor T of shape (N, N, K)
3. Compute per-scale variance explained (how much each k contributes)
4. Apply PCA across the K dimension for each genome pair to get
   a K-dimensional "distance profile" per pair
5. Fuse using data-driven weights (variance-based or learned)
6. Output: fused matrix, per-scale matrices, weight vector, HTML report

USAGE
-----
python kali_tensor.py \
  -g genomes/*.fasta \
  -k 2 3 4 5 \
  -b 500 \
  -o results/tensor_output

python kali_tensor.py \
  -g genomes/*.fasta \
  -k 2 3 4 5 \
  -b 500 \
  --fusion variance \
  --compare-single \
  -o results/tensor_output
"""
from __future__ import annotations

import argparse
import json
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# ── K-mer helpers ─────────────────────────────────────────────────────────────

BASE_MAP = {'A':0,'C':1,'G':2,'T':3,'a':0,'c':1,'g':2,'t':3}

def kmer_vector(seq: str, k: int) -> np.ndarray:
    size = 4 ** k
    vec  = np.zeros(size, dtype=np.float64)
    for i in range(len(seq) - k + 1):
        idx, valid = 0, True
        for j in range(k):
            b = BASE_MAP.get(seq[i+j], -1)
            if b == -1: valid = False; break
            idx = idx * 4 + b
        if valid:
            vec[idx] += 1
    total = vec.sum()
    return vec / total if total > 0 else vec


def read_fasta(path: str) -> str:
    """Return concatenated sequence from all contigs."""
    seqs = []
    with open(path) as fh:
        buf = []
        for line in fh:
            line = line.strip()
            if line.startswith('>'):
                if buf: seqs.append(''.join(buf))
                buf = []
            else:
                buf.append(line.upper())
        if buf: seqs.append(''.join(buf))
    return ''.join(seqs)


# ── Tensor construction ───────────────────────────────────────────────────────

def build_distance_matrix(vectors: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Build pairwise distance matrix from feature vectors."""
    n = len(vectors)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            if metric == "cosine":
                d = distance.cosine(vectors[i], vectors[j])
            elif metric == "euclidean":
                d = distance.euclidean(vectors[i], vectors[j])
            else:
                d = distance.cosine(vectors[i], vectors[j])
            D[i,j] = D[j,i] = d
    return D


def build_tensor(genome_paths: list[str], k_list: list[int],
                 bin_size: int, verbose: bool = False) -> tuple:
    """
    Build the multi-scale distance tensor.

    Returns:
        tensor    : ndarray (N, N, K) — distance matrix per k
        names     : list of genome names
        k_list    : list of k values used
        vectors   : dict {k: ndarray (N, 4^k)} — feature vectors per k
    """
    names = [Path(p).stem for p in genome_paths]
    N     = len(names)
    K     = len(k_list)

    if verbose:
        print(f"  Building tensor: {N} genomes × {N} genomes × {K} k-scales")

    # Load sequences
    sequences = []
    for i, path in enumerate(genome_paths):
        seq = read_fasta(path)
        original_len = len(seq)
        # Long genomes are subsampled to 50 evenly-spaced blocks so that all
        # genomes contribute equal-length sequences to the k-mer vectors.
        # A warning is printed so the caller knows this happened.
        if original_len > bin_size * 10:
            step  = original_len // 50
            parts = [seq[j:j+bin_size] for j in range(0, original_len-bin_size, step)]
            seq   = ''.join(parts[:50])
            if verbose:
                print(f"    [{i+1}/{N}] {Path(path).stem}: long genome "
                      f"({original_len:,} bp) — subsampled to "
                      f"{len(seq):,} bp for tensor analysis")
        else:
            if verbose:
                print(f"    [{i+1}/{N}] {Path(path).stem} ({original_len:,} bp)")
        sequences.append(seq)

    # Build vectors and matrices for each k
    all_vectors = {}
    tensor      = np.zeros((N, N, K))

    for ki, k in enumerate(k_list):
        if verbose:
            print(f"  Computing k={k} distance matrix...")
        vecs = np.array([kmer_vector(seq, k) for seq in sequences])
        all_vectors[k] = vecs
        D = build_distance_matrix(vecs)
        tensor[:, :, ki] = D

    return tensor, names, k_list, all_vectors


# ── Tensor analysis ───────────────────────────────────────────────────────────

def compute_scale_weights(tensor: np.ndarray,
                          method: str = "variance") -> np.ndarray:
    """
    Compute per-scale weights for fusion.

    Methods:
      variance  : weight by variance of each k-scale matrix
                  (higher variance = more discriminating)
      uniform   : equal weights
      pca       : first PCA component across scales per pair,
                  use explained variance as weights
    """
    N, _, K = tensor.shape

    if method == "uniform":
        return np.ones(K) / K

    if method == "variance":
        variances = np.array([
            np.var(tensor[:, :, k])
            for k in range(K)
        ])
        variances = np.maximum(variances, 1e-12)
        return variances / variances.sum()

    if method == "pca":
        # Flatten each (N,N) matrix to a vector, stack into (K, N*N) matrix
        scale_vecs = np.array([tensor[:,:,k].flatten() for k in range(K)])
        pca = PCA(n_components=1)
        pca.fit(scale_vecs.T)
        # Use absolute loading of each k on first PC as weight
        loadings = np.abs(pca.components_[0])
        loadings = np.maximum(loadings, 0)
        s = loadings.sum()
        return loadings / s if s > 0 else np.ones(K) / K

    return np.ones(K) / K


def fuse_tensor(tensor: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted linear fusion of per-scale matrices into one distance matrix."""
    N, _, K = tensor.shape
    fused = np.zeros((N, N))
    for ki in range(K):
        fused += weights[ki] * tensor[:, :, ki]
    return fused


def tensor_pca(tensor: np.ndarray, k_list: list[int]) -> dict:
    """
    PCA across the K dimension.
    Each genome pair becomes a K-dimensional point.
    Returns variance explained per component and loadings.
    """
    N, _, K = tensor.shape
    # Extract upper triangle (pairwise distances)
    idx     = np.triu_indices(N, k=1)
    pairs   = np.array([tensor[idx[0], idx[1], ki] for ki in range(K)]).T
    # (n_pairs, K) matrix

    if pairs.shape[0] < 2:
        return {"variance_explained": [], "loadings": [], "n_pairs": 0}

    scaler = StandardScaler()
    pairs_s = scaler.fit_transform(pairs)

    n_comp = min(K, pairs_s.shape[0])
    pca    = PCA(n_components=n_comp)
    pca.fit(pairs_s)

    return {
        "variance_explained": pca.explained_variance_ratio_.tolist(),
        "loadings":           pca.components_.tolist(),
        "k_labels":           [str(k) for k in k_list],
        "n_pairs":            int(pairs.shape[0]),
    }


def compare_single_vs_tensor(tensor: np.ndarray, fused: np.ndarray,
                               k_list: list[int]) -> dict:
    """
    Compare fused tensor matrix against each single-k matrix.
    Uses Spearman correlation of upper-triangle distances as the metric.
    Higher correlation with fused = similar to tensor.
    The key finding: fused captures more of the total variance.
    """
    N = tensor.shape[0]
    idx = np.triu_indices(N, k=1)
    fused_vec = fused[idx]

    results = {}
    for ki, k in enumerate(k_list):
        single_vec = tensor[:, :, ki][idx]
        if fused_vec.std() > 0 and single_vec.std() > 0:
            corr, pval = spearmanr(fused_vec, single_vec)
        else:
            corr, pval = 0.0, 1.0
        results[str(k)] = {
            "spearman_r":   float(corr),
            "p_value":      float(pval),
            "mean_distance": float(single_vec.mean()),
            "std_distance":  float(single_vec.std()),
        }

    # Variance of fused vs each single
    fused_var = float(np.var(fused_vec))
    for ki, k in enumerate(k_list):
        single_var = float(np.var(tensor[:,:,ki][idx]))
        results[str(k)]["variance_ratio"] = fused_var / single_var if single_var > 0 else 1.0

    return results


# ── Clustering quality ────────────────────────────────────────────────────────

def evaluate_clustering_quality(tensor: np.ndarray, fused: np.ndarray,
                                  labels: list[str], k_list) -> dict:
    """
    

    This directly answers the question: 'Does the fused tensor produce better
    genus separation than any single-k matrix?'
    """
    if len(set(labels)) < 2:
        return {"error": "Need at least 2 distinct labels for silhouette score"}

    results = {}
    K = tensor.shape[2]

    for ki in range(K):
        D = tensor[:, :, ki].copy()
        # Clip small negatives from floating point
        D = np.clip(D, 0, None)
        try:
            score = silhouette_score(D, labels, metric="precomputed")
            lbl = str(k_list[ki])
            results[lbl] = {
                "silhouette": round(float(score), 4),
                "interpretation": (
                    "excellent" if score > 0.7 else
                    "good"      if score > 0.5 else
                    "fair"      if score > 0.25 else
                    "poor"
                )
            }
        except Exception as e:
            results[str(k_list[ki])] = {"error": str(e)}

    # Fused matrix silhouette
    fused_clipped = np.clip(fused, 0, None)
    try:
        fused_score = silhouette_score(fused_clipped, labels, metric="precomputed")
        results["fused"] = {
            "silhouette": round(float(fused_score), 4),
            "interpretation": (
                "excellent" if fused_score > 0.7 else
                "good"      if fused_score > 0.5 else
                "fair"      if fused_score > 0.25 else
                "poor"
            )
        }
        # The key comparison: is fused better than the best single-k?
        single_scores = [
            v["silhouette"] for k, v in results.items()
            if k != "fused" and "silhouette" in v
        ]
        if single_scores:
            best_single = max(single_scores)
            best_single_k = max(
                (k for k in results if k != "fused" and "silhouette" in results[k]),
                key=lambda k: results[k]["silhouette"]
            )
            results["summary"] = {
                "fused_silhouette":       round(float(fused_score), 4),
                "best_single_silhouette": round(best_single, 4),
                "best_single_k":          best_single_k,
                "improvement":            round(float(fused_score - best_single), 4),
                "fused_is_best":          fused_score >= best_single,
            }
    except Exception as e:
        results["fused"] = {"error": str(e)}

    return results


# ── CSV export ────────────────────────────────────────────────────────────────

def save_matrix_csv(matrix: np.ndarray, names: list[str], path: str) -> None:
    df = pd.DataFrame(matrix, index=names, columns=names)
    df.to_csv(path)


def save_tensor_csv(tensor: np.ndarray, names: list[str],
                    k_list: list[int], path: str) -> None:
    """Save all per-scale matrices as a long-format CSV."""
    rows = []
    N = len(names)
    for ki, k in enumerate(k_list):
        for i in range(N):
            for j in range(i+1, N):
                rows.append({
                    "k":         k,
                    "genome_i":  names[i],
                    "genome_j":  names[j],
                    "distance":  round(tensor[i, j, ki], 6),
                })
    pd.DataFrame(rows).to_csv(path, index=False)


# ── HTML report ───────────────────────────────────────────────────────────────

def save_html_report(tensor: np.ndarray, fused: np.ndarray,
                     names: list[str], k_list: list[int],
                     weights: np.ndarray, pca_result: dict,
                     comparison: dict, out_path: str,
                     args_info: dict,
                     clustering_quality: dict | None = None) -> None:

    payload = json.dumps({
        "names":      names,
        "k_list":     k_list,
        "weights":    weights.tolist(),
        "fused":      fused.tolist(),
        "per_k":      {str(k): tensor[:,:,ki].tolist()
                       for ki, k in enumerate(k_list)},
        "pca":        pca_result,
        "comparison": comparison,
        "clustering": clustering_quality or {},
        "args":       args_info,
        "n_genomes":  len(names),
    })

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>KALI Tensor Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/react/18.2.0/umd/react.production.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/react-dom/18.2.0/umd/react-dom.production.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/babel-standalone/7.23.2/babel.min.js"></script>
<script id="kali-tensor" type="application/json">""" + payload + """</script>
<style>
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:'IBM Plex Sans',Arial,sans-serif;background:#0d0f14;color:#e2e8f8;padding:1.5rem;}
h1{font-size:1rem;font-weight:500;color:#9b72f8;font-family:'IBM Plex Mono',monospace;margin-bottom:.25rem;}
.meta{font-size:11px;color:#6b7a9e;font-family:'IBM Plex Mono',monospace;margin-bottom:1.25rem;}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:1.5rem;}
.card{background:#1c2133;border:1px solid #2a3050;border-radius:8px;padding:.6rem .9rem;}
.cl{font-size:10px;color:#6b7a9e;margin-bottom:2px;text-transform:uppercase;letter-spacing:.08em;}
.cv{font-size:1.2rem;font-weight:600;}
h2{font-size:.8rem;font-weight:500;color:#6b7a9e;font-family:'IBM Plex Mono',monospace;
   text-transform:uppercase;letter-spacing:.1em;margin:1.5rem 0 .75rem;}
.k-tabs{display:flex;gap:4px;margin-bottom:.75rem;flex-wrap:wrap;}
.ktab{padding:4px 12px;border-radius:4px;font-size:11px;font-family:'IBM Plex Mono',monospace;
  border:1px solid #2a3050;background:#1c2133;color:#6b7a9e;cursor:pointer;transition:all .12s;}
.ktab:hover{border-color:#9b72f8;color:#9b72f8;}
.ktab.active{border-color:#9b72f8;color:#9b72f8;background:rgba(155,114,248,.1);}
.ktab.fused{border-color:#f5a623;color:#f5a623;}
.ktab.fused.active{background:rgba(245,166,35,.1);}
table{width:100%;border-collapse:collapse;font-size:11px;font-family:'IBM Plex Mono',monospace;}
th,td{border:.5px solid #2a3050;padding:5px 10px;text-align:left;}
th{background:#161a24;color:#6b7a9e;font-weight:500;}
tr:nth-child(even) td{background:#161a24;}
.weight-bar{display:flex;align-items:center;gap:8px;}
.wbg{flex:1;height:6px;background:#2a3050;border-radius:3px;overflow:hidden;}
.wfill{height:6px;border-radius:3px;background:#9b72f8;}
</style>
</head>
<body>
<div id="root"></div>
<script type="text/babel">
const D = JSON.parse(document.getElementById('kali-tensor').textContent);

function App() {
  const [activeK, setActiveK] = React.useState('fused');
  const topWeight = Math.max(...D.weights);
  const topKidx   = D.weights.indexOf(topWeight);
  const topK      = D.k_list[topKidx];

  return (
    <div>
      <h1>KALI multi-scale distance tensor</h1>
      <p className="meta">
        {D.n_genomes} genomes | layers={D.k_list.join(',')} |
        bin={D.args.bin_size} bp | fusion={D.args.fusion}
        {Object.entries(D.args)
          .filter(([k]) => !['k_list','bin_size','fusion','n_genomes','mode','layers'].includes(k))
          .map(([k,v]) => ` | ${k}: ${v}`).join('')}
      </p>

      <div className="cards">
        <div className="card">
          <div className="cl">Genomes</div>
          <div className="cv" style={{color:'#1dc9a0'}}>{D.n_genomes}</div>
        </div>
        <div className="card">
          <div className="cl">k-scales</div>
          <div className="cv" style={{color:'#9b72f8'}}>{D.k_list.length}</div>
        </div>
        <div className="card">
          <div className="cl">Dominant scale</div>
          <div className="cv" style={{color:'#f5a623'}}>{topK}</div>
        </div>
        <div className="card">
          <div className="cl">PC1 variance</div>
          <div className="cv" style={{color:'#5b8df8'}}>
            {D.pca.variance_explained.length > 0
              ? (D.pca.variance_explained[0]*100).toFixed(1)+'%'
              : 'N/A'}
          </div>
        </div>
      </div>

      <h2>Scale weights ({D.args.fusion} fusion)</h2>
      <table>
        <thead>
          <tr><th>k</th><th>Weight</th><th>Relative contribution</th>
              <th>Mean distance</th><th>Variance</th></tr>
        </thead>
        <tbody>
          {D.k_list.map((k,i) => {
            const comp = D.comparison[String(k)] || {};
            return (
              <tr key={k}>
                <td style={{color:'#9b72f8',fontWeight:500}}>{k}</td>
                <td>{(D.weights[i]*100).toFixed(1)}%</td>
                <td>
                  <div className="weight-bar">
                    <div className="wbg">
                      <div className="wfill"
                        style={{width:(D.weights[i]*100)+'%'}} />
                    </div>
                  </div>
                </td>
                <td>{comp.mean_distance ? comp.mean_distance.toFixed(4) : '—'}</td>
                <td>{comp.std_distance  ? comp.std_distance.toFixed(4)  : '—'}</td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <h2>Single-k vs tensor comparison</h2>
      <table>
        <thead>
          <tr><th>k</th><th>Spearman r vs fused</th>
              <th>Variance ratio (fused/single)</th></tr>
        </thead>
        <tbody>
          {D.k_list.map(k => {
            const comp = D.comparison[String(k)] || {};
            const r    = comp.spearman_r || 0;
            const vr   = comp.variance_ratio || 1;
            return (
              <tr key={k}>
                <td style={{color:'#9b72f8'}}>{k}</td>
                <td style={{color: r > 0.9 ? '#1dc9a0' : r > 0.7 ? '#f5a623' : '#f05050'}}>
                  {r.toFixed(4)}
                </td>
                <td style={{color: vr > 1 ? '#1dc9a0' : '#6b7a9e'}}>
                  {vr.toFixed(3)}x {vr > 1 ? '↑ more variance' : ''}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      {D.clustering && D.clustering.summary && (
        <div>
          <h2>Silhouette scores — genus separation quality</h2>
          <p style={{fontSize:11,color:'#6b7a9e',fontFamily:'monospace',marginBottom:'0.75rem'}}>
            Silhouette score [-1,1]: higher = better genus cluster separation. 
            Answers: does the fused tensor improve over single-k?
          </p>
          <table>
            <thead>
              <tr><th>Scale</th><th>Silhouette score</th><th>Quality</th><th>Bar</th></tr>
            </thead>
            <tbody>
              {Object.entries(D.clustering)
                .filter(([k]) => k !== 'summary' && D.clustering[k].silhouette !== undefined)
                .map(([k, v]) => (
                <tr key={k}>
                  <td style={{color: k==='fused'?'#f5a623':'#9b72f8',fontWeight:500}}>
                    {k==='fused'?'★ Fused tensor':k}
                  </td>
                  <td style={{color: v.silhouette>0.5?'#1dc9a0':v.silhouette>0.25?'#f5a623':'#f05050',
                    fontWeight:500}}>
                    {v.silhouette.toFixed(4)}
                  </td>
                  <td style={{color:'#6b7a9e',fontSize:10}}>{v.interpretation}</td>
                  <td style={{width:120}}>
                    <div style={{background:'#2a3050',borderRadius:3,height:8}}>
                      <div style={{background:k==='fused'?'#f5a623':'#9b72f8',height:8,
                        borderRadius:3,width:Math.max(0,v.silhouette*100)+'%'}}/>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{marginTop:'0.5rem',padding:'8px 12px',background:'#161a24',
            border:'1px solid #2a3050',borderRadius:6,fontSize:11,fontFamily:'monospace'}}>
            <span style={{color:'#6b7a9e'}}>Summary: </span>
            <span style={{color: D.clustering.summary.fused_is_best?'#1dc9a0':'#f05050'}}>
              {D.clustering.summary.fused_is_best
                ? `✓ Fused tensor (${D.clustering.summary.fused_silhouette}) outperforms best single-k ` +
                  `(${D.clustering.summary.best_single_k}: ${D.clustering.summary.best_single_silhouette}) ` +
                  `by ${D.clustering.summary.improvement > 0 ? '+' : ''}${D.clustering.summary.improvement.toFixed(4)}`
                : `✗ Best single-k (${D.clustering.summary.best_single_k}: ${D.clustering.summary.best_single_silhouette}) ` +
                  `outperforms fused (${D.clustering.summary.fused_silhouette})`
              }
            </span>
          </div>
        </div>
      )}

      <h2>PCA across k-scales</h2>
      {D.pca.variance_explained.length > 0 ? (
        <table>
          <thead>
            <tr><th>Component</th><th>Variance explained</th>
                {D.k_list.map(k => <th key={k}>{k} loading</th>)}
            </tr>
          </thead>
          <tbody>
            {D.pca.variance_explained.map((ve,ci) => (
              <tr key={ci}>
                <td>PC{ci+1}</td>
                <td style={{color:'#5b8df8'}}>{(ve*100).toFixed(1)}%</td>
                {D.pca.loadings[ci].map((l,li) => (
                  <td key={li} style={{
                    color: Math.abs(l) > 0.5 ? '#9b72f8' : '#6b7a9e'
                  }}>
                    {l.toFixed(3)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p style={{fontSize:11,color:'#6b7a9e'}}>
          Not enough genome pairs for PCA.
        </p>
      )}

      <h2>Distance heatmap</h2>
      <div className="k-tabs">
        <button className={'ktab fused' + (activeK==='fused'?' active':'')}
          onClick={() => setActiveK('fused')}>Fused tensor</button>
        {D.k_list.map(k => (
          <button key={k}
            className={'ktab' + (activeK===String(k)?' active':'')}
            onClick={() => setActiveK(String(k))}>{k}</button>
        ))}
      </div>
      <Heatmap
        key={activeK}
        matrix={activeK==='fused' ? D.fused : D.per_k[activeK]}
        names={D.names}
        title={activeK==='fused' ? 'Fused tensor distance' : activeK+' distance'}
      />
    </div>
  );
}

function Heatmap({matrix, names, title}) {
  const ref = React.useRef(null);
  React.useEffect(() => {
    if (!ref.current || !matrix) return;
    const W = ref.current.clientWidth || 700;
    const N = names.length;
    const margin = {l: 110, r: 20, t: 20, b: 110};
    const size   = Math.min(W - margin.l - margin.r, 500);
    const cell   = size / N;
    const H      = size + margin.t + margin.b;

    d3.select(ref.current).selectAll('*').remove();
    const svg = d3.select(ref.current).append('svg')
      .attr('width', W).attr('height', H);
    const g = svg.append('g')
      .attr('transform', `translate(${margin.l},${margin.t})`);

    const flat = matrix.flat().filter(v => v > 0);
    const maxV = d3.quantile(flat.sort(d3.ascending), 0.95) || 1;
    const color = d3.scaleSequential([0, maxV], d3.interpolateViridis);

    names.forEach((ri, i) => {
      names.forEach((ci, j) => {
        g.append('rect')
          .attr('x', j*cell).attr('y', i*cell)
          .attr('width', cell).attr('height', cell)
          .attr('fill', color(matrix[i][j]));
      });
    });

    const fontSize = Math.max(7, Math.min(11, cell * 0.7));
    names.forEach((n,i) => {
      const label = n.length > 14 ? n.slice(0,13)+'…' : n;
      g.append('text').attr('x', -5).attr('y', i*cell + cell/2)
        .attr('text-anchor','end').attr('dominant-baseline','middle')
        .style('font-size', fontSize+'px').style('fill','#9b9a94')
        .text(label);
      g.append('text').attr('x', i*cell+cell/2).attr('y', size+10)
        .attr('text-anchor','start').attr('dominant-baseline','hanging')
        .attr('transform',`rotate(45,${i*cell+cell/2},${size+10})`)
        .style('font-size', fontSize+'px').style('fill','#9b9a94')
        .text(label);
    });

    // Colorbar
    const cbW = 120, cbH = 10;
    const cbX = size - cbW, cbY = size + margin.b - 20;
    const cbScale = d3.scaleLinear([0, cbW], [0, maxV]);
    const cbAxis  = d3.axisBottom(d3.scaleLinear([0, cbW],[0,maxV]))
      .ticks(3).tickFormat(d3.format('.2f'));
    for (let px = 0; px < cbW; px++) {
      g.append('rect').attr('x',cbX+px).attr('y',cbY)
        .attr('width',1).attr('height',cbH)
        .attr('fill', color(cbScale(px)));
    }
    g.append('g').attr('transform',`translate(${cbX},${cbY+cbH})`)
      .call(cbAxis).selectAll('text').style('fill','#6b7a9e').style('font-size','9px');
    g.append('text').attr('x',cbX+cbW/2).attr('y',cbY-4)
      .attr('text-anchor','middle').style('font-size','9px').style('fill','#6b7a9e')
      .text('distance');
  }, [matrix, names]);

  return <div ref={ref} style={{width:'100%'}} />;
}

ReactDOM.createRoot(document.getElementById('root')).render(<App/>);
</script>
</body>
</html>"""

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)


# ── CLI ───────────────────────────────────────────────────────────────────────

def load_matrices_from_csvs(csv_paths: list[str],
                             labels: list[str] | None) -> tuple:
    """
    Load pre-computed distance matrix CSVs (from Hash or KALI_non-hash)
    and stack into a tensor.

    Returns:
        tensor : ndarray (N, N, K)
        names  : list of genome names
        labels : list of layer labels (e.g. "Hash k=4", "KALI_non-hash k=3")
    """
    matrices = []
    names    = None
    auto_labels = []

    for i, path in enumerate(csv_paths):
        df = pd.read_csv(path, index_col=0)
        mat = df.values.astype(float)
        # Symmetrise and zero diagonal
        mat = (mat + mat.T) / 2
        np.fill_diagonal(mat, 0)

        # Strip NCBI version suffixes (.1 .2 .3) so matrices built from
        # files with different versioning still align correctly.
        import re as _re
        df.index = [_re.sub(r'\.\d+$', '', str(n).strip()) for n in df.index]

        if names is None:
            names = list(df.index)
        else:
            # Reindex to common names
            common = [n for n in names if n in df.index]
            if len(common) < 2:
                raise ValueError(
                    f"Matrix {path} shares < 2 genome names with first matrix.\n"
                    f"  First matrix names (sample): {names[:5]}\n"
                    f"  This matrix names  (sample): {list(df.index)[:5]}"
                )
            names = common
            idx = [list(df.index).index(n) for n in names]
            mat = mat[np.ix_(idx, idx)]

        matrices.append(mat)
        lbl = labels[i] if (labels and i < len(labels)) else Path(path).stem
        auto_labels.append(lbl)

    N = len(names)
    K = len(matrices)
    tensor = np.zeros((N, N, K))
    for ki, mat in enumerate(matrices):
        n = min(N, mat.shape[0])
        tensor[:n, :n, ki] = mat[:n, :n]

    return tensor, names, auto_labels


def parse_args():
    p = argparse.ArgumentParser(
        description="KALI Phase 5 — Multi-scale distance tensor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MODES
-----
Mode 1 — Raw FASTA (builds matrices internally from genomes):
  python kali_tensor.py -g genomes/*.fasta -k 2 3 4 5 -b 500 -o results/tensor

Mode 2 — Pre-computed matrices (use Hash and/or KALI_non-hash output CSVs):
  python kali_tensor.py \
    --matrices hash_k3.csv hash_k4.csv hash_k5.csv kali_non_hash_k3.csv \
    --labels "Hash k=3" "Hash k=4" "Hash k=5" "KALI_non-hash k=3" \
    --fusion variance -o results/tensor

Mode 2 is recommended — uses validated Hash/KALI_non-hash results directly.
"""
    )
    # Mode 1 — raw FASTA
    p.add_argument('-g', '--genomes', nargs='+', default=None,
                   help='FASTA file(s) — Mode 1 only')
    p.add_argument('-k', '--k', nargs='+', type=int, default=[2, 3, 4, 5],
                   help='k-mer sizes for Mode 1 (default: 2 3 4 5)')
    p.add_argument('-b', '--bin-size', type=int, default=500,
                   help='Block size for Mode 1 (default: 500)')

    # Mode 2 — pre-computed matrices
    p.add_argument('--matrices', nargs='+', default=None,
                   help='Pre-computed distance matrix CSV files from Hash/KALI_non-hash — Mode 2')
    p.add_argument('--labels', nargs='+', default=None,
                   help='Layer labels for --matrices (e.g. "Hash k=4" "KALI_non-hash k=3")')

    # Shared
    p.add_argument('--fusion', choices=['variance','uniform','pca'],
                   default='variance',
                   help='Fusion method (default: variance)')
    p.add_argument('--compare-single', action='store_true',
                   help='Compare fused matrix vs each layer')
    p.add_argument('--labels-col', default=None,
                   metavar='CSV_PATH',
                   help='CSV with genome_id,label columns — enables silhouette score '
                        'computation showing whether fused tensor improves genus separation '
                        '(e.g. the same metadata CSV used for the classifier)')
    p.add_argument('-o', '--output', required=True,
                   help='Output base path')
    p.add_argument('-v', '--verbose', action='store_true')
    p.add_argument('--meta', nargs='+', default=None,
                   metavar='KEY=VALUE',
                   help='Custom metadata fields shown in the HTML report '
                        '(e.g. --meta project=MyStudy analyst=JaneDoe note="first run")')
    p.add_argument('--name-map', nargs='+', default=None,
                   metavar='ID=NAME',
                   help='Replace genome IDs with display names in all outputs '
                        '(e.g. --name-map NC_011748="E. coli 55989"). '
                        'Applied to CSV rows/columns and the HTML report.')
    return p.parse_args()


def main():
    args = parse_args()
    out_base = Path(args.output)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n── KALI Multi-scale Distance Tensor ──────────────────")

    # ── Mode 2: pre-computed matrices ────────────────────────────────────────
    if args.matrices:
        csv_paths = [p for p in args.matrices if Path(p).exists()]
        missing   = [p for p in args.matrices if not Path(p).exists()]
        if missing:
            for m in missing: print(f"  WARNING: matrix not found: {m}")
        if len(csv_paths) < 2:
            print("Error: need at least 2 matrix CSV files")
            sys.exit(1)

        print(f"   Mode:    Pre-computed matrices")
        print(f"   Layers:  {len(csv_paths)}")
        print(f"   Fusion:  {args.fusion}")
        print()

        tensor, names, layer_labels = load_matrices_from_csvs(
            csv_paths, args.labels
        )
        k_list    = layer_labels  # use filename stems as layer identifiers
        args_info = {
            "k_list":     layer_labels,
            "bin_size": "pre-computed",
            "fusion":     args.fusion,
            "n_genomes":  len(names),
            "mode":       "matrices",
            "layers":     layer_labels,
        }
        print(f"   Genomes: {len(names)}")
        for i, lbl in enumerate(layer_labels):
            print(f"   Layer {i+1}: {lbl}")
        print()

    # ── Mode 1: raw FASTA ─────────────────────────────────────────────────────
    else:
        if not args.genomes:
            print("Error: provide either -g/--genomes or --matrices")
            sys.exit(1)

        genome_paths = []
        for g in args.genomes:
            expanded = glob(g)
            genome_paths.extend(expanded if expanded else [g])
        genome_paths = [p for p in genome_paths if Path(p).exists()]

        if len(genome_paths) < 2:
            print("Error: need at least 2 genome files")
            sys.exit(1)

        print(f"   Mode:    Raw FASTA")
        print(f"   Genomes: {len(genome_paths)}")
        print(f"   k:       {args.k}")
        print(f"   Block:   {args.bin_size:,} bp")
        print(f"   Fusion:  {args.fusion}")
        print()

        layer_labels = [f"k={k}" for k in args.k]
        args_info = {
            "k_list":     args.k,
            "bin_size": args.bin_size,
            "fusion":     args.fusion,
            "n_genomes":  len(genome_paths),
            "mode":       "fasta",
        }

        tensor, names, k_list, _ = build_tensor(
            genome_paths, args.k, args.bin_size, args.verbose
        )
        k_list = args.k

    # Apply genome name map if provided (replaces IDs in all outputs)
    if args.name_map:
        nmap = {}
        for item in args.name_map:
            if '=' in item:
                k, v = item.split('=', 1)
                nmap[k.strip()] = v.strip()
        if nmap:
            names = [nmap.get(n, n) for n in names]
            if args.verbose:
                mapped = sum(1 for n in names if n in nmap.values())
                print(f'   Name map: {mapped}/{len(names)} genomes renamed')

    # Merge any custom --meta KEY=VALUE pairs into args_info
    if args.meta:
        import re as _re2
        for item in args.meta:
            if '=' in item:
                key, val = item.split('=', 1)
                m = _re2.match(r'^label_(\d+)$', key.strip())
                if m:
                    idx2 = int(m.group(1))
                    if 0 <= idx2 < len(layer_labels):
                        layer_labels[idx2] = val.strip()
                else:
                    args_info[key.strip()] = val.strip()
            else:
                print(f"  WARNING: --meta entry ignored (no '='): {item!r}")

    # Compute weights and fuse
    print("Computing scale weights...")
    weights = compute_scale_weights(tensor, method=args.fusion)
    for ki, lbl in enumerate(layer_labels):
        print(f"  {lbl}: weight={weights[ki]*100:.1f}%")

    print("\nFusing matrices...")
    fused = fuse_tensor(tensor, weights)

    # PCA across k-scales
    print("Running tensor PCA...")
    # Use string layer_labels throughout for display (avoids double "k=" in Mode 2)
    k_list_for_pca = layer_labels
    pca_result = tensor_pca(tensor, k_list_for_pca)
    if pca_result["variance_explained"]:
        for ci, ve in enumerate(pca_result["variance_explained"]):
            print(f"  PC{ci+1}: {ve*100:.1f}% variance explained")

    # Comparison
    comparison = {}
    if args.compare_single or True:  # always run
        print("\nComparing single-k vs fused tensor...")
        # compare_single_vs_tensor keys results as f"k={element}" for each
        # element in k_list_for_pca.  Build the same key here directly so we
        # never need a fallback lookup.
        comparison = compare_single_vs_tensor(tensor, fused, k_list_for_pca)
        for lbl in layer_labels:
            key  = lbl
            comp = comparison.get(key, {})
            if comp:
                print(f"  {lbl}: Spearman r={comp.get('spearman_r',0):.4f}  "
                      f"variance ratio={comp.get('variance_ratio',1):.3f}x")

    # Silhouette scores — answers 'does the tensor actually help clustering?'
    clustering_quality = {}
    if args.labels_col:
        print(f"\nComputing silhouette scores (genus labels from --labels-col)...")
        try:
            label_meta = pd.read_csv(args.labels_col, encoding="utf-8-sig")
            id_col   = label_meta.columns[0]
            lbl_col  = label_meta.columns[1] if label_meta.shape[1] > 1 else label_meta.columns[0]
            lbl_map  = dict(zip(
                label_meta[id_col].astype(str).str.strip(),
                label_meta[lbl_col].astype(str).str.strip()
            ))
            genome_labels = [lbl_map.get(n, "Unknown") for n in names]
            clustering_quality = evaluate_clustering_quality(
                tensor, fused, genome_labels, k_list_for_pca
            )
            if "summary" in clustering_quality:
                s = clustering_quality["summary"]
                print(f"  Fused silhouette:      {s['fused_silhouette']:.4f}")
                print(f"  Best single-k ({s['best_single_k']}): {s['best_single_silhouette']:.4f}")
                marker = "✓ IMPROVEMENT" if s["fused_is_best"] else "✗ no improvement"
                print(f"  Improvement:           {s['improvement']:+.4f}  {marker}")
        except Exception as e:
            print(f"  Silhouette skipped: {e}")
            clustering_quality = {"error": str(e)}

    # Save outputs
    print("\nSaving outputs...")

    # Fused matrix CSV
    fused_csv = str(out_base) + "_tensor_fused.csv"
    save_matrix_csv(fused, names, fused_csv)
    print(f"  Fused matrix: {fused_csv}")

    # Per-k matrices CSV
    tensor_csv = str(out_base) + "_tensor_perk.csv"
    save_tensor_csv(tensor, names, k_list, tensor_csv)
    print(f"  Per-k tensor: {tensor_csv}")

    # Weights CSV
    weights_csv = str(out_base) + "_tensor_weights.csv"
    pd.DataFrame({
        "k":      k_list,
        "weight": weights.tolist(),
    }).to_csv(weights_csv, index=False)
    print(f"  Weights:      {weights_csv}")

    # HTML report
    html_path = str(out_base) + "_tensor.html"
    save_html_report(
        tensor, fused, names, k_list_for_pca, weights,
        pca_result, comparison, html_path, args_info,
        clustering_quality=clustering_quality
    )
    print(f"  HTML report:  {html_path}")

    # Silhouette CSV for supplementary data
    if clustering_quality and "summary" in clustering_quality:
        sil_rows = []
        for k, v in clustering_quality.items():
            if k != "summary" and "silhouette" in v:
                sil_rows.append({"scale": k, "silhouette": v["silhouette"],
                                  "interpretation": v["interpretation"]})
        if sil_rows:
            sil_csv = str(out_base) + "_silhouette.csv"
            pd.DataFrame(sil_rows).to_csv(sil_csv, index=False)
            print(f"  Silhouette:   {sil_csv}")

    print(f"\nDone.\n")


if __name__ == "__main__":
    main()
