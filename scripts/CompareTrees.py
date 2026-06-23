#!/usr/bin/env python3
"""
CompareTrees.py — Compare two phylogenetic trees (topology and branch lengths).

Takes two Newick trees (e.g., an embedding-based tree from EmbedTree.py and
an alignment-based tree from AlignTree.py) and computes standard tree distance
metrics for benchmarking.

Metrics reported:
    rf              — Robinson-Foulds (symmetric difference) distance
    rf_max          — Maximum possible RF for the given leaf count
    rf_normalized   — RF / rf_max  (0 = identical topology, 1 = maximally different)
    wrf             — Weighted Robinson-Foulds (branch-length sensitive)
    euclidean       — Euclidean branch-length distance (Kuhner-Felsenstein)
    n_taxa_t1       — Leaf count in tree1
    n_taxa_t2       — Leaf count in tree2
    n_taxa_shared   — Leaves present in both trees
    n_taxa_only_t1  — Leaves present only in tree1
    n_taxa_only_t2  — Leaves present only in tree2

If the trees contain different leaf sets, use --prune to restrict comparison
to the shared set. Otherwise the script exits with an error.

Requirements:
    pip install dendropy

Output formats (--format, comma-separated):
    tsv   — two-column metric/value table  (default)
    json  — dictionary of all metrics
    txt   — human-readable report with ASCII tables

Usage
-----
    CompareTrees.py --tree1 embed.nw --tree2 align.nw --output comparison
    CompareTrees.py --tree1 embed.nw --tree2 align.nw --output comparison \\
                    --label1 ESM2-650M --label2 IQ-TREE
    CompareTrees.py --tree1 embed.nw --tree2 align.nw --output comparison \\
                    --format tsv,json --prune
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

VERSION = "v0.1.0"

TOPOLOGY_METRICS  = ["rf", "rf_max", "rf_normalized"]
BRANCHLEN_METRICS = ["wrf", "euclidean"]
TAXA_METRICS      = ["n_taxa_t1", "n_taxa_t2", "n_taxa_shared",
                     "n_taxa_only_t1", "n_taxa_only_t2"]
ALL_METRICS       = TOPOLOGY_METRICS + BRANCHLEN_METRICS + TAXA_METRICS


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


def _parse_formats(fmt_arg: str) -> list:
    valid = {"tsv", "json", "txt"}
    chosen = []
    for f in fmt_arg.split(","):
        f = f.strip().lower()
        if f not in valid:
            print(f"ERROR: unknown format '{f}'. Valid: {', '.join(sorted(valid))}",
                  file=sys.stderr)
            sys.exit(1)
        if f not in chosen:
            chosen.append(f)
    return chosen


# ── Tree loading ───────────────────────────────────────────────────────────────

def load_trees(path1: Path, path2: Path):
    """Load two Newick trees into a shared taxon namespace."""
    import dendropy
    _log(f"Loading tree 1: {path1.name}")
    _log(f"Loading tree 2: {path2.name}")
    tns = dendropy.TaxonNamespace()
    t1 = dendropy.Tree.get(path=str(path1), schema="newick", taxon_namespace=tns)
    t2 = dendropy.Tree.get(path=str(path2), schema="newick", taxon_namespace=tns)
    return t1, t2


def get_leaf_labels(tree) -> set:
    return {leaf.taxon.label for leaf in tree.leaf_node_iter()}


def prune_to_shared(t1, t2, only1: set, only2: set):
    t1p = t1.clone(depth=1)
    t2p = t2.clone(depth=1)
    if only1:
        t1p.prune_taxa_with_labels(list(only1))
    if only2:
        t2p.prune_taxa_with_labels(list(only2))
    return t1p, t2p


# ── Distance metrics ───────────────────────────────────────────────────────────

def compute_metrics(t1, t2, n_taxa_t1: int, n_taxa_t2: int,
                    n_shared: int, n_only1: int, n_only2: int) -> dict:
    from dendropy.calculate import treecompare

    n = n_shared  # number of shared (comparison) taxa
    # Maximum RF for an unrooted binary tree with n leaves = 2*(n-3)
    rf_max = max(2 * (n - 3), 0)

    _log(f"Computing Robinson-Foulds distance ...")
    rf = treecompare.symmetric_difference(t1, t2)
    rf_norm = rf / rf_max if rf_max > 0 else 0.0

    _log(f"Computing weighted Robinson-Foulds distance ...")
    wrf = treecompare.weighted_robinson_foulds_distance(t1, t2)

    _log(f"Computing Euclidean branch-length distance ...")
    euc = treecompare.euclidean_distance(t1, t2)

    return {
        "rf":             int(rf),
        "rf_max":         int(rf_max),
        "rf_normalized":  round(rf_norm, 6),
        "wrf":            round(float(wrf), 6),
        "euclidean":      round(float(euc), 6),
        "n_taxa_t1":      n_taxa_t1,
        "n_taxa_t2":      n_taxa_t2,
        "n_taxa_shared":  n_shared,
        "n_taxa_only_t1": n_only1,
        "n_taxa_only_t2": n_only2,
    }


# ── ASCII table ────────────────────────────────────────────────────────────────

def _ascii_table(headers: list, rows: list, title: str = "") -> list:
    """Return a list of strings forming a Unicode box table."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def _hline(left, mid, right, fill="─"):
        return left + mid.join(fill * (w + 2) for w in widths) + right

    top    = _hline("┌", "┬", "┐")
    head   = "│" + "│".join(f" {h:<{widths[i]}} " for i, h in enumerate(headers)) + "│"
    sep    = _hline("├", "┼", "┤")
    bot    = _hline("└", "┴", "┘")

    lines = []
    if title:
        lines.append(title)
    lines.append(top)
    lines.append(head)
    lines.append(sep)
    for row in rows:
        lines.append("│" + "│".join(f" {str(v):<{widths[i]}} " for i, v in enumerate(row)) + "│")
    lines.append(bot)
    return lines


def print_stderr(metrics: dict, label1: str, label2: str,
                 path1: Path, path2: Path) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nCompareTrees.py  {VERSION}  {ts}", file=sys.stderr)
    print(f"  Tree 1 : {label1}  ({path1})", file=sys.stderr)
    print(f"  Tree 2 : {label2}  ({path2})", file=sys.stderr)

    topo_rows   = [(k, metrics[k]) for k in TOPOLOGY_METRICS]
    bl_rows     = [(k, metrics[k]) for k in BRANCHLEN_METRICS]
    taxa_rows   = [(k, metrics[k]) for k in TAXA_METRICS]

    for table_lines in [
        _ascii_table(["Topology metric", "Value"],      topo_rows),
        _ascii_table(["Branch-length metric", "Value"], bl_rows),
        _ascii_table(["Taxa metric", "Count"],          taxa_rows),
    ]:
        print("", file=sys.stderr)
        for line in table_lines:
            print(line, file=sys.stderr)
    print("", file=sys.stderr)


# ── Writers ───────────────────────────────────────────────────────────────────

def write_tsv(metrics: dict, output_base: str) -> None:
    path = Path(f"{output_base}.tsv")
    with open(path, "w") as fh:
        fh.write("metric\tvalue\n")
        for k in ALL_METRICS:
            fh.write(f"{k}\t{metrics[k]}\n")
    _log(f"TSV written: {path}")


def write_json(metrics: dict, output_base: str, label1: str, label2: str,
               path1: Path, path2: Path) -> None:
    path = Path(f"{output_base}.json")
    payload = {
        "date":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tree1_label": label1,
        "tree2_label": label2,
        "tree1_file":  str(path1),
        "tree2_file":  str(path2),
        "metrics":     metrics,
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    _log(f"JSON written: {path}")


def write_txt(metrics: dict, output_base: str, label1: str, label2: str,
              path1: Path, path2: Path) -> None:
    path = Path(f"{output_base}.txt")
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    topo_rows = [(k, metrics[k]) for k in TOPOLOGY_METRICS]
    bl_rows   = [(k, metrics[k]) for k in BRANCHLEN_METRICS]
    taxa_rows = [(k, metrics[k]) for k in TAXA_METRICS]

    lines = [
        f"CompareTrees.py  {VERSION}",
        f"Date   : {ts}",
        f"Tree 1 : {label1}  ({path1})",
        f"Tree 2 : {label2}  ({path2})",
        "",
    ]
    for title, rows in [
        ("Topology metrics",      topo_rows),
        ("Branch-length metrics", bl_rows),
        ("Taxa",                  taxa_rows),
    ]:
        for l in _ascii_table(["Metric", "Value"], rows, title=title):
            lines.append(l)
        lines.append("")

    path.write_text("\n".join(lines))
    _log(f"TXT written: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="CompareTrees",
        description="Compare two phylogenetic trees (topology and branch lengths).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--tree1",   required=True, type=Path,
                    help="First Newick tree file (e.g., embedding-based from EmbedTree.py)")
    ap.add_argument("--tree2",   required=True, type=Path,
                    help="Second Newick tree file (e.g., alignment-based from AlignTree.py)")
    ap.add_argument("--output",  required=True,
                    help="Output basename")
    ap.add_argument("--label1",  default="tree1",
                    help="Label for tree1 in reports (default: tree1)")
    ap.add_argument("--label2",  default="tree2",
                    help="Label for tree2 in reports (default: tree2)")
    ap.add_argument("--format",  default="tsv",
                    help="Output format(s), comma-separated: tsv,json,txt (default: tsv)")
    ap.add_argument("--prune",   action="store_true",
                    help="Prune to shared leaf set if trees have different taxa")
    ap.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    args = ap.parse_args(argv)

    for p in (args.tree1, args.tree2):
        if not p.exists():
            print(f"ERROR: tree file not found: {p}", file=sys.stderr)
            sys.exit(1)

    _check_import("dendropy", "dendropy")

    formats = _parse_formats(args.format)

    _log(f"CompareTrees.py  {VERSION}")

    t1, t2 = load_trees(args.tree1, args.tree2)

    leaves1 = get_leaf_labels(t1)
    leaves2 = get_leaf_labels(t2)
    shared  = leaves1 & leaves2
    only1   = leaves1 - leaves2
    only2   = leaves2 - leaves1

    _log(f"Tree 1 leaves: {len(leaves1)}  |  Tree 2 leaves: {len(leaves2)}  |  Shared: {len(shared)}")

    if only1 or only2:
        msg = (f"Leaf sets differ: {len(only1)} only in tree1, {len(only2)} only in tree2.")
        if not args.prune:
            print(f"ERROR: {msg}\n"
                  f"       Use --prune to restrict comparison to the {len(shared)} shared leaves.",
                  file=sys.stderr)
            sys.exit(1)
        _log(f"WARNING: {msg} Pruning to shared set ({len(shared)} leaves).")
        t1, t2 = prune_to_shared(t1, t2, only1, only2)

    if len(shared) < 4:
        print(f"ERROR: at least 4 shared leaves required for meaningful RF distance "
              f"(found {len(shared)}).", file=sys.stderr)
        sys.exit(1)

    metrics = compute_metrics(t1, t2,
                              n_taxa_t1=len(leaves1),
                              n_taxa_t2=len(leaves2),
                              n_shared=len(shared),
                              n_only1=len(only1),
                              n_only2=len(only2))

    print_stderr(metrics, args.label1, args.label2, args.tree1, args.tree2)

    if "tsv" in formats:
        write_tsv(metrics, args.output)
    if "json" in formats:
        write_json(metrics, args.output, args.label1, args.label2, args.tree1, args.tree2)
    if "txt" in formats:
        write_txt(metrics, args.output, args.label1, args.label2, args.tree1, args.tree2)

    _log("Done.")


if __name__ == "__main__":
    main()
