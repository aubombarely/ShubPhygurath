#!/usr/bin/env python3
"""
AlignTree.py — Build an alignment-based phylogenetic tree from protein sequences.

Runs a multiple sequence alignment followed by tree inference, producing
a Newick tree for comparison with embedding-based trees from EmbedTree.py.

Alignment tools (--aligner):
    mafft   — MAFFT --auto mode  (default)
    muscle  — MUSCLE v3 or v5 (auto-detected)

Tree inference tools (--tree_tool):
    fasttree — FastTree with LG model  (default; fast)
    iqtree   — IQ-TREE 2 with model selection (slower; more accurate)

Model options (--model):
    FastTree : lg (default), wag, jtt
    IQ-TREE  : any valid model string or TEST for auto-selection (default: TEST)

Requirements:
    External tools: mafft or muscle, fasttree or iqtree2/iqtree
    Python: no extra packages needed beyond stdlib

Usage
-----
    AlignTree.py --fasta proteins.fasta --output align_tree
    AlignTree.py --fasta proteins.fasta --output align_tree \\
                 --aligner muscle --tree_tool iqtree
    AlignTree.py --fasta proteins.fasta --output align_tree \\
                 --tree_tool iqtree --model LG+G4 --bootstrap 1000 --threads 8
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

VERSION = "v0.1.0"

FASTTREE_MODELS = {"lg": "-lg", "wag": "-wag", "jtt": "-jtt"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr)


def _require_tool(name: str) -> str:
    """Return the full path of a tool or exit with a helpful message."""
    path = shutil.which(name)
    if path is None:
        print(f"ERROR: '{name}' not found in PATH. Please install it and retry.",
              file=sys.stderr)
        sys.exit(1)
    return path


def _run(cmd: list, capture_stdout: bool = False) -> subprocess.CompletedProcess:
    _log(f"Running: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE if capture_stdout else None,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: command failed (exit {result.returncode}):", file=sys.stderr)
        print(result.stderr[-2000:], file=sys.stderr)
        sys.exit(1)
    return result


# ── Alignment ──────────────────────────────────────────────────────────────────

def run_mafft(fasta: Path, aln_out: Path, threads: int) -> None:
    tool = _require_tool("mafft")
    result = _run([tool, "--auto", "--thread", str(threads), "--quiet", str(fasta)],
                  capture_stdout=True)
    aln_out.write_text(result.stdout)
    _log(f"Alignment written: {aln_out}")


def _detect_muscle_version(tool: str) -> int:
    result = subprocess.run([tool, "--version"], capture_output=True, text=True)
    output = result.stdout + result.stderr
    if "MUSCLE v5" in output or "muscle 5" in output.lower():
        return 5
    return 3


def run_muscle(fasta: Path, aln_out: Path) -> None:
    tool = _require_tool("muscle")
    version = _detect_muscle_version(tool)
    _log(f"MUSCLE version detected: {version}")
    if version >= 5:
        _run([tool, "-align", str(fasta), "-output", str(aln_out)])
    else:
        _run([tool, "-in", str(fasta), "-out", str(aln_out)])
    _log(f"Alignment written: {aln_out}")


ALIGNERS = {
    "mafft":  run_mafft,
    "muscle": run_muscle,
}


# ── Tree inference ─────────────────────────────────────────────────────────────

def run_fasttree(aln: Path, nw_out: Path, model: str) -> None:
    tool = _require_tool("FastTree")
    model_flag = FASTTREE_MODELS.get(model.lower())
    if model_flag is None:
        print(f"ERROR: unknown FastTree model '{model}'. Valid: {', '.join(FASTTREE_MODELS)}",
              file=sys.stderr)
        sys.exit(1)
    result = _run([tool, model_flag, str(aln)], capture_stdout=True)
    nw_out.write_text(result.stdout.strip() + "\n")
    _log(f"Newick tree written: {nw_out}")


def run_iqtree(aln: Path, nw_out: Path, model: str, bootstrap: int,
               threads: int) -> None:
    # Try iqtree2 first, fall back to iqtree
    tool = shutil.which("iqtree2") or shutil.which("iqtree")
    if tool is None:
        print("ERROR: neither 'iqtree2' nor 'iqtree' found in PATH.", file=sys.stderr)
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = str(Path(tmpdir) / "iqtree_run")
        cmd = [tool, "-s", str(aln), "-m", model,
               "-T", str(threads), "--prefix", prefix, "--quiet"]
        if bootstrap > 0:
            cmd += ["-bb", str(bootstrap)]
        _run(cmd)

        treefile = Path(f"{prefix}.treefile")
        if not treefile.exists():
            print(f"ERROR: IQ-TREE treefile not found: {treefile}", file=sys.stderr)
            sys.exit(1)
        nw_out.write_text(treefile.read_text())

    _log(f"Newick tree written: {nw_out}")


# ── Run info ───────────────────────────────────────────────────────────────────

def save_run_info(fasta: Path, aln_out: Path, nw_out: Path, aligner: str,
                  tree_tool: str, model: str, bootstrap: int,
                  threads: int, output_base: str) -> None:
    info = {
        "date":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_fasta": str(fasta),
        "alignment":  str(aln_out),
        "newick":     str(nw_out),
        "aligner":    aligner,
        "tree_tool":  tree_tool,
        "model":      model,
        "bootstrap":  bootstrap,
        "threads":    threads,
    }
    info_path = Path(f"{output_base}.info.json")
    with open(info_path, "w") as fh:
        json.dump(info, fh, indent=2)
        fh.write("\n")
    _log(f"Run info saved: {info_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="AlignTree",
        description="Build an alignment-based phylogenetic tree from protein sequences.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--fasta",      required=True, type=Path,
                    help="Input protein FASTA file")
    ap.add_argument("--output",     required=True,
                    help="Output basename (e.g. 'align_tree')")
    ap.add_argument("--aligner",    choices=["mafft", "muscle"], default="mafft",
                    help="Alignment tool (default: mafft)")
    ap.add_argument("--tree_tool",  choices=["fasttree", "iqtree"], default="fasttree",
                    help="Tree inference tool (default: fasttree)")
    ap.add_argument("--model",      default=None,
                    help="Substitution model. FastTree: lg/wag/jtt (default: lg). "
                         "IQ-TREE: any model string (default: TEST)")
    ap.add_argument("--bootstrap",  type=int, default=0,
                    help="UFBoot replicates for IQ-TREE (default: 0 = no bootstrap; "
                         "ignored for FastTree)")
    ap.add_argument("--threads",    type=int, default=4,
                    help="CPU threads (default: 4)")
    ap.add_argument("--version",    action="version", version=f"%(prog)s {VERSION}")
    args = ap.parse_args(argv)

    if not args.fasta.exists():
        print(f"ERROR: --fasta file not found: {args.fasta}", file=sys.stderr)
        sys.exit(1)

    # Set default model per tree tool
    if args.model is None:
        args.model = "lg" if args.tree_tool == "fasttree" else "TEST"

    _log(f"AlignTree.py  {VERSION}")
    _log(f"Aligner: {args.aligner}  |  Tree tool: {args.tree_tool}  |  Model: {args.model}")

    aln_out = Path(f"{args.output}.aln.fasta")
    nw_out  = Path(f"{args.output}.nw")

    # Step 1 — Alignment
    _log("── Step 1: Multiple sequence alignment ──")
    if args.aligner == "mafft":
        run_mafft(args.fasta, aln_out, args.threads)
    else:
        run_muscle(args.fasta, aln_out)

    # Step 2 — Tree inference
    _log("── Step 2: Tree inference ──")
    if args.tree_tool == "fasttree":
        run_fasttree(aln_out, nw_out, args.model)
    else:
        run_iqtree(aln_out, nw_out, args.model, args.bootstrap, args.threads)

    save_run_info(args.fasta, aln_out, nw_out, args.aligner, args.tree_tool,
                  args.model, args.bootstrap, args.threads, args.output)

    _log("Done.")


if __name__ == "__main__":
    main()
