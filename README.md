# ShubPhygurath

**Benchmarking protein language model embeddings against alignment-based phylogenetics.**

ShubPhygurath provides a five-script pipeline that constructs phylogenetic trees from protein sequences via two independent routes — ESM-2 embeddings and multiple sequence alignment — and quantifies how closely those trees agree using standard topological and branch-length distance metrics.

> **Name:** Lovecraft Mythos convention — *Shub-Niggurath* + *Phylo*gurath

---

## Background

Alignment-free methods based on protein language model (PLM) embeddings have emerged as a fast alternative to alignment-based phylogenetics. ShubPhygurath operationalises a direct comparison:

| Route | Tools | Output |
|---|---|---|
| **Embedding-based** | ESM-2 (8M – 15B) → pairwise distance matrix → NJ tree | `EmbedProteins.py` + `EmbedTree.py` |
| **Alignment-based** | MAFFT / MUSCLE → IQ-TREE / FastTree | `AlignTree.py` |
| **Comparison** | Robinson-Foulds, weighted RF, Euclidean | `CompareTrees.py` |
| **Visualisation** | Side-by-side Newick figure | `PlotTrees.py` |

---

## Requirements

### Conda environment

Create a dedicated environment and install all dependencies in one go:

```bash
conda create -n shubphygurath python=3.10
conda activate shubphygurath

# Scientific core + bioinformatics packages
conda install -c conda-forge -c bioconda \
    numpy scipy matplotlib biopython dendropy

# PyTorch — choose one:
# CPU only
conda install -c pytorch pytorch cpuonly
# GPU (CUDA 12.1)
conda install -c pytorch -c nvidia pytorch pytorch-cuda=12.1

# Alignment and tree inference tools
conda install -c bioconda mafft muscle fasttree iqtree

# ESM-2 (not on conda; install via pip into the active environment)
pip install fair-esm
```

| Package | Channel | Used by |
|---|---|---|
| `numpy`, `scipy` | conda-forge | EmbedProteins, EmbedTree |
| `matplotlib` | conda-forge | PlotTrees |
| `biopython` | bioconda | EmbedTree, PlotTrees |
| `dendropy` | conda-forge | CompareTrees |
| `pytorch` | pytorch | EmbedProteins |
| `fair-esm` | pip | EmbedProteins |
| `mafft` | bioconda | AlignTree |
| `muscle` | bioconda | AlignTree |
| `fasttree` | bioconda | AlignTree |
| `iqtree` | bioconda | AlignTree |

---

## Scripts

### 1. `EmbedProteins.py` — Generate ESM-2 protein embeddings

Encodes each protein sequence as a fixed-length vector by mean-pooling the per-residue representations from the last transformer layer of an ESM-2 model.

```
EmbedProteins.py --fasta <proteins.fasta> --output <basename>
                 [--model esm2_650M] [--batch_size 4]
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--fasta` | required | Input protein FASTA |
| `--output` | required | Output basename |
| `--model` | `esm2_650M` | ESM-2 model(s); comma-separated or `all` |
| `--batch_size` | `4` | Sequences per GPU batch (reduce on OOM) |

**Supported models:**

| Short name | Parameters | Embedding dim |
|---|---|---|
| `esm2_8M` | 8 M | 320 |
| `esm2_35M` | 35 M | 480 |
| `esm2_150M` | 150 M | 640 |
| `esm2_650M` | 650 M | 1 280 |
| `esm2_3B` | 3 B | 2 560 |
| `esm2_15B` | 15 B | 5 120 |

**Output per model:**

```
{output}.{model}.npy    # embeddings array, shape (n_seqs, embed_dim)
{output}.{model}.json   # metadata: sequence IDs, model info, run parameters
```

---

### 2. `EmbedTree.py` — Build NJ tree from embeddings

Computes a pairwise distance matrix from embeddings (cosine or Euclidean) and constructs a neighbor-joining tree with BioPython.

```
EmbedTree.py --embeddings <embeddings.esm2_650M.npy> --output <basename>
             [--distance cosine] [--save_matrix]
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--embeddings` | required | `.npy` file from EmbedProteins.py |
| `--metadata` | auto-detected | `.json` sibling of the `.npy` file |
| `--output` | required | Output basename |
| `--distance` | `cosine` | Pairwise metric: `cosine` or `euclidean` |
| `--save_matrix` | off | Also write the full distance matrix as TSV |

**Output:**

```
{output}.nw             # Newick tree
{output}.info.json      # run metadata (model, metric, method)
{output}.dist.tsv       # pairwise distance matrix (only with --save_matrix)
```

---

### 3. `AlignTree.py` — Alignment-based phylogenetic tree

Runs a multiple sequence alignment followed by maximum-likelihood or approximate ML tree inference.

```
AlignTree.py --fasta <proteins.fasta> --output <basename>
             [--aligner mafft] [--tree_tool fasttree]
             [--model lg] [--bootstrap 0] [--threads 4]
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--fasta` | required | Input protein FASTA |
| `--output` | required | Output basename |
| `--aligner` | `mafft` | Alignment tool: `mafft` or `muscle` |
| `--tree_tool` | `fasttree` | Tree tool: `fasttree` or `iqtree` |
| `--model` | `lg` / `TEST` | FastTree: `lg`, `wag`, `jtt`. IQ-TREE: any model string |
| `--bootstrap` | `0` | UFBoot replicates (IQ-TREE only; 0 = none) |
| `--threads` | `4` | CPU threads |

**Output:**

```
{output}.aln.fasta      # multiple sequence alignment
{output}.nw             # Newick tree
{output}.info.json      # run metadata (aligner, tree tool, model)
```

---

### 4. `CompareTrees.py` — Quantify tree similarity

Computes Robinson-Foulds and branch-length distances between two Newick trees using `dendropy`.

```
CompareTrees.py --tree1 <embed.nw> --tree2 <align.nw> --output <basename>
                [--label1 tree1] [--label2 tree2]
                [--format tsv] [--prune]
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--tree1` | required | First Newick file |
| `--tree2` | required | Second Newick file |
| `--output` | required | Output basename |
| `--label1` | `tree1` | Label for tree1 in reports |
| `--label2` | `tree2` | Label for tree2 in reports |
| `--format` | `tsv` | Output format(s): `tsv`, `json`, `txt` |
| `--prune` | off | Prune to shared leaf set if taxa differ |

**Metrics reported:**

| Metric | Description |
|---|---|
| `rf` | Robinson-Foulds symmetric difference |
| `rf_max` | Maximum possible RF (= 2*(n−3) for unrooted) |
| `rf_normalized` | RF / rf_max — 0 = identical topology, 1 = maximally different |
| `wrf` | Weighted RF (branch-length sensitive) |
| `euclidean` | Euclidean branch-length distance (Kuhner-Felsenstein) |
| `n_taxa_shared` | Leaves present in both trees |
| `n_taxa_only_t1/t2` | Leaves unique to each tree |

**Output:**

```
{output}.tsv            # metric/value table
{output}.json           # full results with metadata (--format json)
{output}.txt            # human-readable ASCII table report (--format txt)
```

---

### 5. `PlotTrees.py` — Side-by-side tree figure

Renders both trees in a single publication-quality figure with optional leaf coloring and RF annotation.

```
PlotTrees.py --tree1 <embed.nw> --tree2 <align.nw> --output <basename>
             [--label1 tree1] [--label2 tree2]
             [--comparison comparison.json] [--color_file groups.tsv]
             [--format pdf] [--width 14] [--height auto] [--cladogram]
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--tree1` | required | First Newick file |
| `--tree2` | required | Second Newick file |
| `--output` | required | Output basename |
| `--label1/2` | filename stem | Panel titles |
| `--comparison` | none | CompareTrees.py JSON → adds RF footer |
| `--color_file` | none | `seq_id<TAB>group` TSV → colors leaves |
| `--format` | `pdf` | Output format(s): `pdf`, `png`, `svg` |
| `--width` | `14` | Figure width in inches |
| `--height` | auto | Figure height in inches (auto: 0.35 × n_leaves) |
| `--cladogram` | off | Equal branch lengths (topology only) |

**Color file format** (`groups.tsv`):
```
seq1    Clade_A
seq2    Clade_A
seq3    Clade_B
```

**Output:**

```
{output}.pdf / .png / .svg
```

---

## Full pipeline example

```bash
# 1. Embed proteins with ESM-2 650M
EmbedProteins.py \
    --fasta proteins.fasta \
    --output embeddings \
    --model esm2_650M

# 2. Build embedding-based NJ tree (cosine distance)
EmbedTree.py \
    --embeddings embeddings.esm2_650M.npy \
    --output embed_tree \
    --distance cosine

# 3. Build alignment-based ML tree (MAFFT + FastTree, LG model)
AlignTree.py \
    --fasta proteins.fasta \
    --output align_tree \
    --aligner mafft \
    --tree_tool fasttree \
    --threads 8

# 4. Compare the two trees
CompareTrees.py \
    --tree1 embed_tree.nw \
    --tree2 align_tree.nw \
    --label1 "ESM2-650M" \
    --label2 "MAFFT+FastTree" \
    --output comparison \
    --format tsv,json

# 5. Plot side by side
PlotTrees.py \
    --tree1 embed_tree.nw \
    --tree2 align_tree.nw \
    --label1 "ESM2-650M (cosine NJ)" \
    --label2 "MAFFT + FastTree (LG)" \
    --comparison comparison.json \
    --color_file groups.tsv \
    --output trees_plot \
    --format pdf,png
```

---

## Output file summary

| File | Produced by | Description |
|---|---|---|
| `embeddings.esm2_650M.npy` | EmbedProteins | Embedding matrix |
| `embeddings.esm2_650M.json` | EmbedProteins | Sequence IDs + model metadata |
| `embed_tree.nw` | EmbedTree | Embedding-based Newick tree |
| `embed_tree.info.json` | EmbedTree | Distance metric + tree method |
| `embed_tree.dist.tsv` | EmbedTree | Pairwise distance matrix (optional) |
| `align_tree.aln.fasta` | AlignTree | Multiple sequence alignment |
| `align_tree.nw` | AlignTree | Alignment-based Newick tree |
| `align_tree.info.json` | AlignTree | Aligner + tree tool metadata |
| `comparison.tsv` | CompareTrees | Metric table |
| `comparison.json` | CompareTrees | Full results with metadata |
| `trees_plot.pdf` | PlotTrees | Side-by-side figure |

---

## Notes

- Sequences longer than 1 022 residues are truncated when using ESM-2 (hard limit of the model architecture); a warning is printed for each truncated sequence.
- For IQ-TREE, `iqtree2` is tried first in PATH; the script falls back to `iqtree` if not found.
- MUSCLE version is auto-detected (v3 and v5 have different CLI flags).
- `CompareTrees.py` requires at least 4 shared leaves to compute a meaningful RF distance.
- GPU memory is cleared between ESM-2 model runs when using `--model all`.
