"""Tokenize raw conversation data and compute prefix overlap statistics.

Usage:
  python prepare_data.py
  python prepare_data.py prepare._target_=prepare_data.tokenize
  python prepare_data.py prepare._target_=prepare_data.compute_overlap
"""
import json
import logging
from pathlib import Path

import numpy as np
import polars as pl
import hydra
from omegaconf import DictConfig
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from spase_cache.utils import setup_output_dir, interleave

log = logging.getLogger(__name__)


_HASH_PRIME = 2654435761
_HASH_MASK = (1 << 63) - 1


def _filter(raw_path, max_rounds, max_rows):
    log.info(f"Loading {raw_path}...")
    df = pl.read_csv(raw_path)
    df = df.with_columns(
        pl.col("message_create_time").str.replace("ts:", "").cast(pl.Float64).alias("ts")
    )

    # drop rows without timestamp
    df = df.filter(pl.col("ts").is_not_null())

    # keep only first max_rounds messages per conversation
    df = df.sort(["url", "message_index"])
    df = df.with_columns(
        pl.col("message_index").rank("ordinal").over("url").alias("_rank")
    ).filter(pl.col("_rank") <= max_rounds).drop("_rank")

    # limit total rows
    df = df.head(max_rows)

    log.info(f"Filtered: {df.n_unique('url')} conversations, {len(df)} rows")
    return df


def tokenize(out_dir, full_cfg, **_kw):
    """Filter raw data, tokenize, and save parquet."""
    df = _filter(full_cfg.data.raw_path, full_cfg.data.max_rounds, full_cfg.data.max_rows)

    log.info(f"Loading tokenizer {full_cfg.model.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(full_cfg.model.tokenizer, use_fast=True)

    # format text as '<|role|> text' with \n separator between messages
    df = df.with_columns(
        (pl.col("message_index") == pl.col("message_index").min().over("url")).alias("_is_first")
    )
    df = df.with_columns(
        (pl.when(pl.col("_is_first"))
         .then(pl.lit("<|") + pl.col("role") + pl.lit("|> ") + pl.col("plain_text").fill_null(""))
         .otherwise(pl.lit("\n<|") + pl.col("role") + pl.lit("|> ") + pl.col("plain_text").fill_null(""))
        ).alias("_formatted")
    ).drop("_is_first")

    # tokenize in chunks, converting to numpy immediately to avoid Python int overhead
    # (Python int = 28 bytes each vs numpy int32 = 4 bytes)
    log.info("Tokenizing messages...")
    chunk_size = full_cfg.prepare_data.tokenizer_chunk_size
    formatted = df["_formatted"]
    all_tokens = []
    for i in tqdm(range(0, len(formatted), chunk_size), desc="tokenizing chunks"):
        chunk_texts = formatted[i:i + chunk_size].to_list()
        for ids in tokenizer(chunk_texts, add_special_tokens=False)["input_ids"]:
            all_tokens.append(np.array(ids, dtype=np.int32))
        del chunk_texts
    df = df.drop("_formatted", "plain_text").with_columns(pl.Series("tokens", all_tokens))
    del all_tokens

    # cumulative token count per conversation for truncation
    df = df.with_columns(
        pl.col("tokens").list.len().cum_sum().over("url").alias("cum_tokens")
    )
    n_before = len(df)
    df = df.filter(pl.col("cum_tokens") <= full_cfg.data.max_seq_len)
    log.info(f"Tokenized {len(df)} messages ({n_before - len(df)} truncated)")

    # skip conversations whose total length is below min_seq_len
    min_seq_len = full_cfg.data.get("min_seq_len", 0)
    if min_seq_len > 0:
        conv_total = df.group_by("url").agg(pl.col("tokens").list.len().sum().alias("total_tokens"))
        short_convs = conv_total.filter(pl.col("total_tokens") < min_seq_len)["url"]
        n_convs_before = df.n_unique("url")
        df = df.filter(~pl.col("url").is_in(short_convs))
        log.info(f"Dropped {n_convs_before - df.n_unique('url')} conversations shorter than {min_seq_len} tokens")

    prepared_path = Path(full_cfg.data.processed)
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(prepared_path)
    log.info(f"Saved {len(df)} messages to {prepared_path}")


def train_test_split(requests, train_frac):
    """Split requests into train (first train_frac) and test (the rest)."""
    n_train = int(len(requests) * train_frac)
    return requests[:n_train], requests[n_train:]


def compute_overlap(out_dir, full_cfg, **_kw):
    """Load tokenized parquet and compute prefix overlap distribution."""
    out_dir = Path(out_dir)
    prepared_path = Path(full_cfg.data.processed)
    log.info(f"Loading {prepared_path}...")
    df = pl.read_parquet(prepared_path)

    log.info("Computing prefix overlap distribution...")
    user_rows = df.filter(pl.col("role") == "user").sort("ts")
    n_conversations = user_rows.n_unique("url")
    log.info(f"{len(user_rows)} requests from {n_conversations} conversations")

    # build compact lookup: url -> (msg_indices, token_arrays)
    conv_tokens = {}
    for url in df["url"].unique(maintain_order=True).to_list():
        conv = df.filter(pl.col("url") == url).sort("message_index")
        conv_tokens[url] = (
            conv["message_index"].to_list(),
            [np.array(t, dtype=np.int32) for t in conv["tokens"].to_list()],
        )

    iter_urls = user_rows["url"].to_list()
    iter_msg_indices = user_rows["message_index"].to_list()
    del user_rows, df

    requests = list(zip(iter_urls, iter_msg_indices))
    if full_cfg.data.get("interleave", False):
        log.info("Interleaving requests with Poisson arrivals (seed=%d)...", full_cfg.seed)
        requests = interleave(requests, full_cfg.seed)

    train_requests, test_requests = train_test_split(
        requests, full_cfg.data.get("train_frac", 0.5))
    log.info("Train: %d requests, Test: %d requests", len(train_requests), len(test_requests))

    seen = set()

    # warm cache with train split
    log.info("Warming prefix cache with train split...")
    for url, msg_idx in tqdm(train_requests, desc="train"):
        msg_indices, token_lists = conv_tokens[url]
        n_msgs = 0
        for mi in msg_indices:
            n_msgs += 1
            if mi == msg_idx:
                break
        h = 0
        for tokens in token_lists[:n_msgs]:
            for t in tokens:
                h = (h * _HASH_PRIME + int(t) + 1) & _HASH_MASK
                seen.add(h)

    # evaluate on test split
    lcp_lengths = []
    log.info("Simulating non-deleting prefix cache on test split...")
    for url, msg_idx in tqdm(test_requests, desc="test"):
        msg_indices, token_lists = conv_tokens[url]
        n_msgs = 0
        for mi in msg_indices:
            n_msgs += 1
            if mi == msg_idx:
                break

        h = 0
        lcp = 0
        matched = True

        for tokens in token_lists[:n_msgs]:
            for t in tokens:
                h = (h * _HASH_PRIME + int(t) + 1) & _HASH_MASK
                if matched and h in seen:
                    lcp += 1
                else:
                    matched = False
                seen.add(h)

        lcp_lengths.append(lcp)

    overlap_path = out_dir / "overlap_lcp.json"
    overlap_path.write_text(json.dumps({
        "lcp_lengths": lcp_lengths,
        "n_requests": len(test_requests),
        "n_train_requests": len(train_requests),
        "n_conversations": n_conversations,
    }))

    n_hits = sum(1 for l in lcp_lengths if l > 0)
    log.info(f"{n_hits}/{len(lcp_lengths)} test requests had cache hits "
             f"({n_hits/len(lcp_lengths)*100:.1f}%)")
    log.info(f"Overlap results saved to {overlap_path}")


def compute_prefix_sharing(out_dir, full_cfg, n_sample=200, **_kw):
    """Sample conversations and compute pairwise LCP matrix for trie diagnostics."""
    out_dir = Path(out_dir)
    df = pl.read_parquet(full_cfg.data.processed)

    urls = df["url"].unique(maintain_order=True).to_list()
    rng = np.random.RandomState(full_cfg.seed)
    if len(urls) > n_sample:
        urls = rng.choice(urls, n_sample, replace=False).tolist()

    conv_seqs = []
    conv_lens = []
    for url in urls:
        conv = df.filter(pl.col("url") == url).sort("message_index")
        tokens = np.concatenate([np.array(t, dtype=np.int32) for t in conv["tokens"].to_list()])
        conv_seqs.append(tokens)
        conv_lens.append(len(tokens))

    n = len(conv_seqs)
    lcp_matrix = np.zeros((n, n), dtype=np.int32)

    for i in tqdm(range(n), desc="prefix sharing"):
        lcp_matrix[i, i] = conv_lens[i]
        for j in range(i + 1, n):
            min_len = min(conv_lens[i], conv_lens[j])
            if min_len == 0:
                continue
            matches = conv_seqs[i][:min_len] == conv_seqs[j][:min_len]
            lcp = int(np.argmin(matches)) if not matches.all() else min_len
            lcp_matrix[i, j] = lcp
            lcp_matrix[j, i] = lcp

    sharing_path = out_dir / "prefix_sharing.json"
    sharing_path.write_text(json.dumps({
        "lcp_matrix": lcp_matrix.tolist(),
        "conv_lens": conv_lens,
        "n_conversations": n,
    }))
    log.info("Prefix sharing matrix (%d×%d) saved to %s", n, n, sharing_path)


def prepare_all(out_dir, full_cfg, **_kw):
    """Run tokenize then compute_overlap."""
    tokenize(out_dir, full_cfg)
    compute_overlap(out_dir, full_cfg)
    compute_prefix_sharing(out_dir, full_cfg)


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    out_dir = setup_output_dir(cfg, "prepare_data")
    fn = hydra.utils.get_method(cfg.prepare_data._target_)
    fn(out_dir=out_dir, full_cfg=cfg)


if __name__ == "__main__":
    main()
