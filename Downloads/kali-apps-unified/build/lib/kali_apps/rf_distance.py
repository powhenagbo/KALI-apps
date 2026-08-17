"""
Robinson-Foulds Distance Calculator
====================================
Computes pairwise unweighted RF, weighted RF, and normalized RF distances
between Newick tree files using DendroPy.

Requirements:
    pip install dendropy

Usage:
    python rf_distance.py tree1.nwk tree2.nwk tree3.nwk ...

Notes:
    - Duplicate leaf taxa (e.g. multiple "Shigella_sonnei") are automatically
      renamed with _2, _3 suffixes in positional order so they are consistent
      across all trees.
    - Normalized RF = RF / 2(n-3), where n = number of taxa.
"""

import re
import sys
import itertools
import collections
import dendropy
from dendropy.calculate import treecompare


# ── helpers ──────────────────────────────────────────────────────────────────

def deduplicate_newick(nwk_path: str) -> str:
    """
    Read a Newick file and rename duplicate leaf taxa with positional suffixes
    (_2, _3, …) so every taxon label is unique.

    The first occurrence keeps its original name; subsequent occurrences get
    _2, _3, … appended.  Applying the same function to all trees in a dataset
    ensures consistent renaming as long as duplicate counts match.
    """
    nwk = open(nwk_path).read()
    counts: dict[str, int] = {}

    def replacer(m: re.Match) -> str:
        name, branch = m.group(1), m.group(2)
        counts[name] = counts.get(name, 0) + 1
        suffix = f"_{counts[name]}" if counts[name] > 1 else ""
        return f"{name}{suffix}:{branch}"

    return re.sub(r'([A-Za-z0-9_\-\.]+):([0-9eE+\-.]+)', replacer, nwk)


def load_trees(paths: list[str]) -> tuple[dendropy.TaxonNamespace, dict[str, dendropy.Tree]]:
    """Load all Newick trees into a shared taxon namespace."""
    tns = dendropy.TaxonNamespace()
    trees: dict[str, dendropy.Tree] = {}
    for path in paths:
        label = path  # use full path as label; trimmed below in display
        nwk = deduplicate_newick(path)
        t = dendropy.Tree.get(data=nwk, schema="newick", taxon_namespace=tns)
        t.encode_bipartitions()
        trees[label] = t
    return tns, trees


# ── main ─────────────────────────────────────────────────────────────────────

def main(paths: list[str]) -> None:
    if len(paths) < 2:
        print("Usage: python rf_distance.py tree1.nwk tree2.nwk ...")
        sys.exit(1)

    # Use short names for display (basename without leading timestamp)
    def short(p: str) -> str:
        base = p.split("/")[-1]          # basename
        base = re.sub(r'^\d+_kali_', '', base)  # strip leading timestamp_kali_
        base = base.replace("_tree.nwk", ".nwk").replace(".nwk", "")
        return base

    tns, trees = load_trees(paths)
    n_taxa = len(tns)
    max_rf  = 2 * (n_taxa - 3)          # maximum RF for unrooted trees

    # Check for duplicates that were renamed
    dups = {t: c for t, c in
            collections.Counter(re.findall(r'([A-Za-z0-9_\-\.]+):',
                                           open(paths[0]).read())).items()
            if c > 1}
    if dups:
        print(f"⚠  Duplicate taxa renamed with positional suffixes: "
              f"{', '.join(dups.keys())}\n")

    print(f"Taxa : {n_taxa}   |   Max RF : {max_rf}\n")

    labels  = list(trees.keys())
    col_w   = 30
    header  = (f"{'Tree A':<{col_w}} {'Tree B':<{col_w}} "
               f"{'RF':>6}  {'RF_norm':>8}  {'wRF':>10}")
    print(header)
    print("─" * len(header))

    for a, b in itertools.combinations(labels, 2):
        rf      = treecompare.symmetric_difference(trees[a], trees[b])
        rf_norm = rf / max_rf if max_rf > 0 else float("nan")
        wrf     = treecompare.weighted_robinson_foulds_distance(trees[a], trees[b])
        sa, sb  = short(a), short(b)
        print(f"{sa:<{col_w}} {sb:<{col_w}} "
              f"{rf:>6}  {rf_norm:>8.3f}  {wrf:>10.6f}")

    print()
    print("Columns")
    print("  RF       – unweighted Robinson-Foulds (symmetric difference)")
    print("  RF_norm  – RF / 2(n-3)  [0 = identical topology, 1 = maximally different]")
    print("  wRF      – weighted RF  (accounts for branch-length differences)")


if __name__ == "__main__":
    main(sys.argv[1:])
