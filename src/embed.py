"""Build train/eval float32 embedding caches once per pinned encoder."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np

from experiment import (cache_dir, digest, embedding_spec, file_hash, load_params,
                        reviews, validate_splits, write_json)


def encode_to_file(encoder, texts, output, batch_size):
    """Bound resident embeddings to one batch while writing a .npy memmap."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    matrix = None
    for start in range(0, len(texts), batch_size):
        block = np.asarray(encoder.encode(texts[start:start + batch_size], batch_size=batch_size,
                           normalize_embeddings=True, convert_to_numpy=True,
                           show_progress_bar=False), dtype=np.float32)
        if block.ndim != 2 or len(block) != len(texts[start:start + batch_size]) or not np.isfinite(block).all():
            raise ValueError("Encoder returned invalid embeddings.")
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError("Zero embedding returned by encoder.")
        block /= norms
        if matrix is None:
            matrix = np.lib.format.open_memmap(output, mode="w+", dtype="float32", shape=(len(texts), block.shape[1]))
        matrix[start:start + len(block)] = block
    matrix.flush()
    shape = list(matrix.shape)
    del matrix
    return shape


def build(params, alias):
    validate_splits(params)
    spec = params["encoders"][alias]
    output = cache_dir(alias)
    output.mkdir(parents=True, exist_ok=False)
    # Hugging Face's shared, content-addressed cache holds one snapshot per revision.
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(spec["id"], revision=spec["revision"],
                                  device=params["embedding"]["device"], trust_remote_code=False,
                                  model_kwargs={"use_safetensors": True})
    encoder.max_seq_length = spec["max_seq_length"]
    try:
        for split in ("train", "eval"):
            texts, ids = reviews(params, split)
            path = output / f"{split}.npy"
            shape = encode_to_file(encoder, texts, path, params["embedding"]["batch_size"])
            contract = embedding_spec(params, alias, split)
            write_json(output / f"{split}.json", {"cache_id": digest(contract), "spec": contract,
                       "shape": shape, "sha256": file_hash(path), "observation_ids": ids})
            del texts, ids
    finally:
        del encoder
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml", type=Path)
    parser.add_argument("--encoder", required=True)
    args = parser.parse_args(argv)
    build(load_params(args.params), args.encoder)


if __name__ == "__main__":
    main()
