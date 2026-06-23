#!/usr/bin/env python3
"""
PlotTrees.py — Side-by-side visualization of two phylogenetic trees.

Creates a publication-quality figure comparing two Newick trees (e.g., an
embedding-based tree from EmbedTree.py vs an alignment-based tree from
AlignTree.py). Leaves can be colored by group and the Robinson-Foulds
distance from CompareTrees.py can be overlaid as an annotation.

Leaf coloring (--color_file):
    Two-column TSV, no header: sequence_id<TAB>group_name
    One color per unique group; up to 20 groups supported (tab10 + tab20).

RF annotation (--comparison):
    JSON file produced by CompareTrees.py; adds RF / RF-norm to the figure.

Output formats (--format, comma-separated):
    pdf   — vector, best for publication  (default)
    png   — raster, 150 dpi
    svg   — vector, editable

Requirements:
    pip install biopython matplotlib

Usage
-----
    PlotTrees.py --tree1 embed.nw --tree2 align.nw --output trees_plot
    PlotTrees.py --tree1 embed.nw --tree2 align.nw --output trees_plot \\
                 --label1 "ESM2-650M" --label2 "MAFFT+FastTree" \\
                 --comparison comparison.json --color_file groups.tsv
    PlotTrees.py --tree1 embed.nw --tree2 align.nw --output trees_plot \\
                 --format pdf,png --width 18 --height 10
"""

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

VERSION = "v0.1.0"

DEFAULT_LEAF_COLOR = "#333333"
DEFAULT_BRANCH_COLOR = "#555555"


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
    valid = {"pdf", "png", "svg"}
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

def load_newick(path: Path):
    from Bio import Phylo
    import io
    tree = Phylo.read(io.StringIO(path.read_text().strip()), "newick")
    leaves = [c.name for c in tree.get_terminals()]
    _log(f"  {path.name}: {len(leaves)} leaves")
    return tree


# ── Color file ─────────────────────────────────────────────────────────────────

def load_color_file(color_file: Path) -> dict:
    """Return {seq_id: group_name} from a two-column TSV (no header)."""
    groups = {}
    with open(color_file) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                print(f"WARNING: color_file line {lineno} has <2 columns, skipping",
                      file=sys.stderr)
                continue
            groups[parts[0]] = parts[1]
    _log(f"Color file: {len(groups)} sequences in {len(set(groups.values()))} groups")
    return groups


def build_leaf_color_map(groups: dict) -> tuple:
    """
    Returns (leaf_color_dict, group_color_dict).
    leaf_color_dict: {seq_id: hex_color}
    group_color_dict: {group_name: hex_color} for the legend
    """
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm

    unique_groups = sorted(set(groups.values()))
    n = len(unique_groups)
    if n <= 10:
        cmap = cm.get_cmap("tab10", 10)
    else:
        cmap = cm.get_cmap("tab20", 20)

    group_colors = {g: mcolors.to_hex(cmap(i % cmap.N))
                    for i, g in enumerate(unique_groups)}
    leaf_colors  = {sid: group_colors[grp] for sid, grp in groups.items()}
    return leaf_colors, group_colors


# ── Cladogram conversion ───────────────────────────────────────────────────────

def to_cladogram(tree):
    """Return a deep copy of the tree with all branch lengths set to 1."""
    t = deepcopy(tree)
    for clade in t.find_clades():
        clade.branch_length = 1.0
    return t


# ── Plotting ───────────────────────────────────────────────────────────────────

def _draw_tree(tree, ax, label: str, leaf_color_func, show_confidence: bool) -> None:
    from Bio import Phylo

    Phylo.draw(
        tree,
        axes=ax,
        do_show=False,
        show_confidence=show_confidence,
        label_func=lambda c: c.name if c.is_terminal() else "",
        label_colors=leaf_color_func,
        branch_labels=None,
    )
    ax.set_title(label, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Branch length", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    # Remove top/right spines for cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _dynamic_height(t1, t2) -> float:
    n = max(
        len(list(t1.get_terminals())),
        len(list(t2.get_terminals())),
    )
    return max(7.0, n * 0.35)


def plot_trees(t1, t2, label1: str, label2: str,
               leaf_colors: dict, group_colors: dict,
               rf_metrics: dict | None,
               figsize: tuple, cladogram: bool,
               output_base: str, formats: list) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if cladogram:
        _log("Converting to cladogram (equal branch lengths) ...")
        t1 = to_cladogram(t1)
        t2 = to_cladogram(t2)

    leaf_color_func = (lambda name: leaf_colors.get(name, DEFAULT_LEAF_COLOR)
                       if leaf_colors else lambda name: DEFAULT_LEAF_COLOR)

    # Determine figure height dynamically if not overridden
    width, height = figsize
    if height is None:
        height = _dynamic_height(t1, t2)
    figsize = (width, height)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    _log(f"Drawing tree 1 ({label1}) ...")
    _draw_tree(t1, ax1, label1, leaf_color_func, show_confidence=False)

    _log(f"Drawing tree 2 ({label2}) ...")
    _draw_tree(t2, ax2, label2, leaf_color_func, show_confidence=False)

    # RF annotation footer
    if rf_metrics:
        rf      = rf_metrics.get("rf", "?")
        rf_norm = rf_metrics.get("rf_normalized", "?")
        rf_max  = rf_metrics.get("rf_max", "?")
        footer  = (f"Robinson-Foulds distance:  RF = {rf} / {rf_max}"
                   f"   |   RF normalized = {rf_norm:.4f}"
                   if isinstance(rf_norm, float) else
                   f"Robinson-Foulds distance:  RF = {rf}")
        fig.text(0.5, 0.01, footer,
                 ha="center", va="bottom", fontsize=9,
                 color="#444444", style="italic")

    # Group legend
    if group_colors:
        patches = [mpatches.Patch(color=col, label=grp)
                   for grp, col in sorted(group_colors.items())]
        fig.legend(handles=patches,
                   loc="lower center",
                   bbox_to_anchor=(0.5, 0.04 if rf_metrics else 0.01),
                   ncol=min(len(patches), 6),
                   fontsize=8,
                   frameon=True,
                   title="Groups",
                   title_fontsize=8)

    bottom_margin = 0.08 if (rf_metrics or group_colors) else 0.04
    fig.subplots_adjust(bottom=bottom_margin, wspace=0.35)

    for fmt in formats:
        out_path = Path(f"{output_base}.{fmt}")
        dpi = 150 if fmt == "png" else None
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        _log(f"Saved: {out_path}")

    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="PlotTrees",
        description="Side-by-side visualization of two phylogenetic trees.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--tree1",      required=True, type=Path,
                    help="First Newick tree (e.g., embedding-based from EmbedTree.py)")
    ap.add_argument("--tree2",      required=True, type=Path,
                    help="Second Newick tree (e.g., alignment-based from AlignTree.py)")
    ap.add_argument("--output",     required=True,
                    help="Output basename (e.g. 'trees_plot' -> trees_plot.pdf)")
    ap.add_argument("--label1",     default=None,
                    help="Title for tree1 panel (default: filename stem)")
    ap.add_argument("--label2",     default=None,
                    help="Title for tree2 panel (default: filename stem)")
    ap.add_argument("--comparison", type=Path, default=None,
                    help="CompareTrees.py JSON output for RF annotation")
    ap.add_argument("--color_file", type=Path, default=None,
                    help="Two-column TSV (seq_id<TAB>group) for leaf coloring")
    ap.add_argument("--format",     default="pdf",
                    help="Output format(s), comma-separated: pdf,png,svg (default: pdf)")
    ap.add_argument("--width",      type=float, default=14.0,
                    help="Figure width in inches (default: 14)")
    ap.add_argument("--height",     type=float, default=None,
                    help="Figure height in inches (default: auto from leaf count)")
    ap.add_argument("--cladogram",  action="store_true",
                    help="Draw as cladogram (equal branch lengths, topology only)")
    ap.add_argument("--version",    action="version", version=f"%(prog)s {VERSION}")
    args = ap.parse_args(argv)

    for p in (args.tree1, args.tree2):
        if not p.exists():
            print(f"ERROR: tree file not found: {p}", file=sys.stderr)
            sys.exit(1)

    if args.comparison and not args.comparison.exists():
        print(f"ERROR: --comparison file not found: {args.comparison}", file=sys.stderr)
        sys.exit(1)

    if args.color_file and not args.color_file.exists():
        print(f"ERROR: --color_file not found: {args.color_file}", file=sys.stderr)
        sys.exit(1)

    for pkg, install in [("Bio", "biopython"), ("matplotlib", "matplotlib")]:
        _check_import(pkg, install)

    formats = _parse_formats(args.format)

    label1 = args.label1 or args.tree1.stem
    label2 = args.label2 or args.tree2.stem

    _log(f"PlotTrees.py  {VERSION}")
    _log(f"Loading trees ...")
    t1 = load_newick(args.tree1)
    t2 = load_newick(args.tree2)

    # Load optional color file
    leaf_colors  = {}
    group_colors = {}
    if args.color_file:
        groups = load_color_file(args.color_file)
        leaf_colors, group_colors = build_leaf_color_map(groups)

    # Load optional RF metrics
    rf_metrics = None
    if args.comparison:
        with open(args.comparison) as fh:
            data = json.load(fh)
        rf_metrics = data.get("metrics")
        _log(f"RF annotation loaded: RF={rf_metrics.get('rf')}, "
             f"RF_norm={rf_metrics.get('rf_normalized')}")

    plot_trees(
        t1, t2, label1, label2,
        leaf_colors, group_colors,
        rf_metrics,
        figsize=(args.width, args.height),
        cladogram=args.cladogram,
        output_base=args.output,
        formats=formats,
    )

    _log("Done.")


if __name__ == "__main__":
    main()
