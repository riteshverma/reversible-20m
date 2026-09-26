"""Prepare data: stream a FineWeb-Edu parquet shard, train an 8k BPE tokenizer,
write ~57M tokens as uint16 binaries (train.bin / val.bin)."""

import io
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

ROOT = Path(__file__).parent
DATA = ROOT / "data"
PARQUET = DATA / "fineweb-edu-00000.parquet"
VOCAB = 8192
TRAIN_TOKENS = 52_000_000
VAL_TOKENS = 5_000_000
TOTAL_NEEDED = TRAIN_TOKENS + VAL_TOKENS


def main():
    t0 = time.time()
    pf = pq.ParquetFile(PARQUET)
    print(f"parquet: {pf.metadata.num_rows} rows, {pf.metadata.num_row_groups} row groups")

    # ---- collect a text sample for tokenizer training (~25MB)
    sample, sample_chars = [], 0
    texts_iter = (batch.column("text").to_pylist()
                  for batch in pf.iter_batches(batch_size=2048, columns=["text"]))
    for batch_texts in texts_iter:
        for t in batch_texts:
            if len(t) >= 300:
                sample.append(t)
                sample_chars += len(t)
        if sample_chars > 25_000_000:
            break
    print(f"tokenizer sample: {len(sample)} docs, {sample_chars/1e6:.1f} MB")

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB,
        special_tokens=["<|endoftext|>"],
        show_progress=False,
    )
    tok.train_from_iterator(sample, trainer)
    eot = tok.token_to_id("<|endoftext|>")
    assert eot == 0, f"EOT id is {eot}, expected 0"
    tok.save(str(DATA / "bpe_8192.json"))
    print(f"tokenizer trained: vocab {tok.get_vocab_size()} ({time.time()-t0:.0f}s)")

    del sample

    # ---- tokenize the corpus into a uint16 buffer
    buf = np.empty(TOTAL_NEEDED + 1_000_000, dtype=np.uint16)
    n = 0
    n_docs = 0
    for batch_texts in texts_iter:
        texts = [t for t in batch_texts if len(t) >= 300]
        if not texts:
            continue
        encs = tok.encode_batch(texts, add_special_tokens=False)
        for e in encs:
            ids = e.ids
            if n + len(ids) + 1 > len(buf):
                break
            buf[n : n + len(ids)] = ids
            n += len(ids)
            buf[n] = eot
            n += 1
            n_docs += 1
        if n % 10_000_000 < 40_000:
            print(f"  {n/1e6:.1f}M tokens ({time.time()-t0:.0f}s)", flush=True)
        if n >= TOTAL_NEEDED:
            break

    print(f"tokenized {n/1e6:.1f}M tokens from {n_docs} docs ({time.time()-t0:.0f}s)")
    train = buf[:TRAIN_TOKENS]
    val = buf[TRAIN_TOKENS : TRAIN_TOKENS + VAL_TOKENS]
    train.tofile(DATA / "train.bin")
    val.tofile(DATA / "val.bin")
    print(f"train.bin {TRAIN_TOKENS/1e6:.0f}M tokens, val.bin {VAL_TOKENS/1e6:.0f}M tokens")


if __name__ == "__main__":
    main()
