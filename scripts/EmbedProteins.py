#!/usr/bin/env python3
"""
EmbedProteins.py — Generate per-sequence embeddings from protein sequences
using ESM-2 protein language models.

Embeddings are computed by mean-pooling the per-residue representations
from the last transformer layer. Sequences longer than the model's maximum
input length (1022 residues) are truncated with a warning.

Supported models (--model):
    esm2_8M    —  ESM-2   8M parameters  (fast, lower accuracy)
    esm2_35M   —  ESM-2  35M parameters
    esm2_150M  —  ESM-2 150M parameters
    esm2_650M  —  ESM-2 650M parameters  (recommended balance)
    esm2_3B    —  ESM-2   3B parameters
    esm2_15B   —  ESM-2  15B parameters  (slow, highest accuracy)

Use 'all' to run every supported model in one go.

Output per model:
    {output}.{model}.npy   — numpy array, shape (n_sequences, embedding_dim)
    {output}.{model}.json  — metadata (sequence IDs, model info, run parameters)

Requirements:
    pip install fair-esm torch numpy

Usage
-----
    EmbedProteins.py --fasta proteins.fasta --output embeddings --model esm2_650M
    EmbedProteins.py --fasta proteins.fasta --output embeddings --model esm2_8M,esm2_650M
    EmbedProteins.py --fasta proteins.fasta --output embeddings --model all
    EmbedProteins.py --fasta proteins.fasta --output embeddings --model esm2_650M --batch_size 8
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

VERSION = "v0.1.0"

ESM_MAX_LEN = 1022  # ESM-2 maximum input length (excluding special tokens)

SUPPORTED_MODELS = {
    "esm2_8M":   "esm2_t6_8M_UR50D",
    "esm2_35M":  "esm2_t12_35M_UR50D",
    "esm2_150M": "esm2_t30_150M_UR50D",
    "esm2_650M": "esm2_t33_650M_UR50D",
    "esm2_3B":   "esm2_t36_3B_UR50D",
    "esm2_15B":  "esm2_t48_15B_UR50D",
}

MODEL_LAYERS = {
    "esm2_t6_8M_UR50D":    6,
    "esm2_t12_35M_UR50D":  12,
    "esm2_t30_150M_UR50D": 30,
    "esm2_t33_650M_UR50D": 33,
    "esm2_t36_3B_UR50D":   36,
    "esm2_t48_15B_UR50D":  48,
}

MODEL_EMBED_DIM = {
    "esm2_t6_8M_UR50D":     320,
    "esm2_t12_35M_UR50D":   480,
    "esm2_t30_150M_UR50D":  640,
    "esm2_t33_650M_UR50D": 1280,
    "esm2_t36_3B_UR50D":   2560,
    "esm2_t48_15B_UR50D":  5120,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr)


def parse_models(model_arg: str) -> list:
    if model_arg.strip().lower() == "all":
        return list(SUPPORTED_MODELS.keys())
    models = []
    for m in model_arg.split(","):
        m = m.strip()
        if m not in SUPPORTED_MODELS:
            print(f"ERROR: unknown model '{m}'. Supported: {', '.join(SUPPORTED_MODELS)}",
                  file=sys.stderr)
            sys.exit(1)
        if m not in models:
            models.append(m)
    return models


# ── FASTA loading ──────────────────────────────────────────────────────────────

def load_fasta(fasta_path: Path) -> list:
    """Return list of (seq_id, sequence) tuples."""
    sequences = []
    seq_id    = None
    seq_buf   = []

    with open(fasta_path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if seq_id is not None:
                    sequences.append((seq_id, "".join(seq_buf)))
                seq_id  = line[1:].split()[0]
                seq_buf = []
            else:
                seq_buf.append(line)
    if seq_id is not None:
        sequences.append((seq_id, "".join(seq_buf)))

    return sequences


# ── Device detection ───────────────────────────────────────────────────────────

def get_device():
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch not found. Install with: pip install torch", file=sys.stderr)
        sys.exit(1)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


# ── Embedding ──────────────────────────────────────────────────────────────────

def load_esm2(model_full_name: str, device):
    try:
        import esm as esmlib
    except ImportError:
        print("ERROR: fair-esm not found. Install with: pip install fair-esm",
              file=sys.stderr)
        sys.exit(1)

    _log(f"Loading {model_full_name} ...")
    loader = getattr(esmlib.pretrained, model_full_name)
    model, alphabet = loader()
    model = model.to(device).eval()
    _log(f"Model loaded on {device}.")
    return model, alphabet


def embed_sequences(model, alphabet, sequences: list, model_full_name: str,
                    device, batch_size: int) -> tuple:
    """
    Generate mean-pooled embeddings for all sequences.
    Returns (embeddings_array, n_truncated, seq_ids).
    """
    import numpy as np
    import torch

    repr_layer  = MODEL_LAYERS[model_full_name]
    batch_conv  = alphabet.get_batch_converter()

    # Truncate sequences exceeding ESM_MAX_LEN
    n_truncated = 0
    prepared    = []
    for seq_id, seq in sequences:
        if len(seq) > ESM_MAX_LEN:
            n_truncated += 1
            _log(f"WARNING: '{seq_id}' truncated from {len(seq)} to {ESM_MAX_LEN} residues")
            seq = seq[:ESM_MAX_LEN]
        prepared.append((seq_id, seq))

    n_batches   = (len(prepared) + batch_size - 1) // batch_size
    embeddings  = []

    for b_idx in range(0, len(prepared), batch_size):
        batch = prepared[b_idx: b_idx + batch_size]
        batch_num = b_idx // batch_size + 1
        _log(f"  Batch {batch_num}/{n_batches}  ({len(batch)} sequences)")

        _, _, tokens = batch_conv(batch)
        tokens = tokens.to(device)

        with torch.no_grad():
            results = model(tokens, repr_layers=[repr_layer], return_contacts=False)

        reps = results["representations"][repr_layer]
        for j, (_, seq) in enumerate(batch):
            emb = reps[j, 1: len(seq) + 1].mean(0).cpu().numpy()
            embeddings.append(emb)

    seq_ids = [sid for sid, _ in prepared]
    return np.array(embeddings), n_truncated, seq_ids


# ── Output ─────────────────────────────────────────────────────────────────────

def save_outputs(embeddings, seq_ids: list, model_short: str, model_full: str,
                 n_truncated: int, device, batch_size: int,
                 fasta_path: Path, output_base: str) -> None:
    import numpy as np

    npy_path  = Path(f"{output_base}.{model_short}.npy")
    json_path = Path(f"{output_base}.{model_short}.json")

    np.save(npy_path, embeddings)
    _log(f"Embeddings saved: {npy_path}  shape={embeddings.shape}")

    metadata = {
        "model_short":     model_short,
        "model_full":      model_full,
        "embedding_dim":   MODEL_EMBED_DIM[model_full],
        "repr_layer":      MODEL_LAYERS[model_full],
        "pooling":         "mean",
        "n_sequences":     len(seq_ids),
        "n_truncated":     n_truncated,
        "max_len":         ESM_MAX_LEN,
        "sequence_ids":    seq_ids,
        "device":          str(device),
        "batch_size":      batch_size,
        "input_fasta":     str(fasta_path),
        "date":            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(json_path, "w") as fh:
        json.dump(metadata, fh, indent=2)
        fh.write("\n")
    _log(f"Metadata saved:   {json_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="EmbedProteins",
        description="Generate ESM-2 protein embeddings from a FASTA file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--fasta",      required=True, type=Path,
                    help="Input protein FASTA file")
    ap.add_argument("--output",     required=True,
                    help="Output basename (e.g. 'embeddings' -> embeddings.esm2_650M.npy)")
    ap.add_argument("--model",      default="esm2_650M",
                    help="Model(s) to use: comma-separated or 'all' (default: esm2_650M). "
                         f"Supported: {', '.join(SUPPORTED_MODELS)}")
    ap.add_argument("--batch_size", type=int, default=4,
                    help="Sequences per batch (reduce if OOM; default: 4)")
    ap.add_argument("--version",    action="version", version=f"%(prog)s {VERSION}")
    args = ap.parse_args(argv)

    if not args.fasta.exists():
        print(f"ERROR: --fasta file not found: {args.fasta}", file=sys.stderr)
        sys.exit(1)

    models  = parse_models(args.model)
    device  = get_device()

    _log(f"EmbedProteins.py  {VERSION}")
    _log(f"Device: {device}")
    _log(f"Models: {', '.join(models)}")

    sequences = load_fasta(args.fasta)
    _log(f"Sequences loaded: {len(sequences)}")

    for model_short in models:
        model_full = SUPPORTED_MODELS[model_short]
        _log(f"── {model_short} ({model_full}) ──")

        model, alphabet = load_esm2(model_full, device)
        embeddings, n_trunc, seq_ids = embed_sequences(
            model, alphabet, sequences, model_full, device, args.batch_size
        )
        save_outputs(embeddings, seq_ids, model_short, model_full,
                     n_trunc, device, args.batch_size, args.fasta, args.output)

        # Free GPU memory before loading next model
        del model
        try:
            import torch
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

    _log("Done.")


if __name__ == "__main__":
    main()
