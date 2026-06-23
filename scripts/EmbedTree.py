#!/usr/bin/env python3
"""
EmbedTree.py — Build a neighbor-joining tree from protein embeddings.

Loads embeddings produced by EmbedProteins.py, computes a pairwise
distance matrix, and constructs a neighbor-joining tree using BioPython.
The tree is written in Newick format. Optionally saves the full pairwise
distance matrix as a TSV.

Distance metrics (--distance):
    cosine    — 1 - cosine_similarity  (default; recommended for embeddings)
    euclidean — L2 distance

Requirements:
    pip install numpy scipy biopython

Usage
-----
    EmbedTree.py --embeddings embeddings.esm2_650M.npy --output tree_650M
    EmbedTree.py --embeddings embeddings.esm2_650M.npy --output tree_650M \\
                 --distance euclidean
    EmbedTree.py --embeddings embeddings.esm2_650M.npy --output tree_650M \\
                 --save_matrix
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

VERSION = "v0.1.0"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr)


def _check_import(package: str, install: str) -> None:
    import importlib
    if importlib.util.find_spec(package) is None:
        print(f"ERROR: '{package}' not found. Install with: pip install {install}",
              file=sys.stderr)
        sys.exit(1)


# ── Input ──────────────────────────────────────────────────────────────────────

def load_embeddings(npy_path: Path) -> "np.ndarray":
    import numpy as np
    _log(f"Loading embeddings from {npy_path.name} ...")
    embeddings = np.load(npy_path)
    _log(f"  Shape: {embeddings.shape}")
    return embeddings


def load_metadata(npy_path: Path, metadata_path: "Path | None") -> dict:
    """Load metadata JSON. If not provided, look for a .json sibling of the .npy file."""
    if metadata_path is None:
        stem = npy_path.name
        if stem.endswith(".npy"):
            stem = stem[:-4]
        metadata_path = npy_path.parent / f"{stem}.json"

    if not metadata_path.exists():
        print(f"ERROR: metadata file not found: {metadata_path}\n"
              f"       Provide it explicitly with --metadata.", file=sys.stderr)
        sys.exit(1)

    with open(metadata_path) as fh:
        meta = json.load(fh)
    _log(f"Metadata loaded from {metadata_path.name}  "
         f"({meta.get('n_sequences', '?')} sequences, "
         f"model: {meta.get('model_short', '?')})")
    return meta


# ── Distance matrix ────────────────────────────────────────────────────────────

def compute_distances(embeddings: "np.ndarray", metric: str) -> "np.ndarray":
    from scipy.spatial.distance import cdist
    _log(f"Computing pairwise {metric} distances ({embeddings.shape[0]} x {embeddings.shape[0]}) ...")
    dist = cdist(embeddings, embeddings, metric=metric)
    # Ensure symmetry and zero diagonal (numerical noise)
    dist = (dist + dist.T) / 2.0
    import numpy as np
    np.fill_diagonal(dist, 0.0)
    _log(f"  Distance range: [{dist.min():.6f}, {dist.max():.6f}]")
    return dist


# ── Tree construction ──────────────────────────────────────────────────────────

def build_nj_tree(dist_matrix: "np.ndarray", seq_ids: list):
    """Build a neighbor-joining tree using BioPython. Returns a Bio.Phylo tree."""
    from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor

    n = len(seq_ids)
    # BioPython requires a lower-triangular matrix (list of lists)
    lower_tri = []
    for i in range(n):
        row = [float(dist_matrix[i][j]) for j in range(i + 1)]
        lower_tri.append(row)

    dm          = DistanceMatrix(seq_ids, lower_tri)
    constructor = DistanceTreeConstructor()
    _log("Building neighbor-joining tree ...")
    tree = constructor.nj(dm)
    return tree


# ── Output ─────────────────────────────────────────────────────────────────────

def save_newick(tree, path: Path) -> None:
    from Bio import Phylo
    import io
    buf = io.StringIO()
    Phylo.write(tree, buf, "newick")
    newick_str = buf.getvalue().strip()
    path.write_text(newick_str + "\n")
    _log(f"Newick tree saved: {path}")


def save_distance_matrix(dist_matrix: "np.ndarray", seq_ids: list, path: Path) -> None:
    import numpy as np
    _log(f"Saving distance matrix: {path}")
    header = "\t".join([""] + seq_ids)
    rows   = [header]
    for i, sid in enumerate(seq_ids):
        row = "\t".join([sid] + [f"{dist_matrix[i][j]:.8f}" for j in range(len(seq_ids))])
        rows.append(row)
    path.write_text("\n".join(rows) + "\n")


def save_run_info(meta: dict, metric: str, n_sequences: int,
                  newick_path: Path, output_base: str) -> None:
    info = {
        "date":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_short":   meta.get("model_short"),
        "model_full":    meta.get("model_full"),
        "embedding_dim": meta.get("embedding_dim"),
        "pooling":       meta.get("pooling", "mean"),
        "n_sequences":   n_sequences,
        "distance":      metric,
        "tree_method":   "neighbor-joining",
        "newick_file":   str(newick_path),
    }
    info_path = Path(f"{output_base}.info.json")
    with open(info_path, "w") as fh:
        json.dump(info, fh, indent=2)
        fh.write("\n")
    _log(f"Run info saved:    {info_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="EmbedTree",
        description="Build a neighbor-joining tree from protein embeddings.",
    )
    ap.add_argument("--embeddings", required=True, type=Path,
                    help="Embeddings .npy file from EmbedProteins.py")
    ap.add_argument("--metadata",   type=Path, default=None,
                    help="Metadata .json file (default: auto-detected from --embeddings path)")
    ap.add_argument("--output",     required=True,
                    help="Output basename (e.g. 'tree_650M' -> tree_650M.nw)")
    ap.add_argument("--distance",   choices=["cosine", "euclidean"], default="cosine",
                    help="Pairwise distance metric (default: cosine)")
    ap.add_argument("--save_matrix", action="store_true",
                    help="Also save the full pairwise distance matrix as TSV")
    ap.add_argument("--version",    action="version", version=f"%(prog)s {VERSION}")
    args = ap.parse_args(argv)

    if not args.embeddings.exists():
        print(f"ERROR: --embeddings file not found: {args.embeddings}", file=sys.stderr)
        sys.exit(1)

    for pkg, install in [("numpy", "numpy"), ("scipy", "scipy"),
                         ("Bio", "biopython")]:
        _check_import(pkg, install)

    _log(f"EmbedTree.py  {VERSION}")
    _log(f"Distance metric: {args.distance}")

    embeddings = load_embeddings(args.embeddings)
    meta       = load_metadata(args.embeddings, args.metadata)
    seq_ids    = meta.get("sequence_ids")

    if seq_ids is None:
        print("ERROR: 'sequence_ids' not found in metadata.", file=sys.stderr)
        sys.exit(1)

    if len(seq_ids) != embeddings.shape[0]:
        print(f"ERROR: {len(seq_ids)} sequence IDs but {embeddings.shape[0]} embeddings.",
              file=sys.stderr)
        sys.exit(1)

    dist_matrix = compute_distances(embeddings, args.distance)
    tree        = build_nj_tree(dist_matrix, seq_ids)

    newick_path = Path(f"{args.output}.nw")
    save_newick(tree, newick_path)

    if args.save_matrix:
        matrix_path = Path(f"{args.output}.dist.tsv")
        save_distance_matrix(dist_matrix, seq_ids, matrix_path)

    save_run_info(meta, args.distance, len(seq_ids), newick_path, args.output)

    _log("Done.")


if __name__ == "__main__":
    main()
