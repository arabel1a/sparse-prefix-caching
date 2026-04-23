"""ShareGPT conversation dataset.

Each conversation has multiple messages (user/assistant turns).
A request corresponds to a user turn: we concatenate all messages up to
that turn into a single token sequence (simulating growing prefix).
"""
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from omegaconf import DictConfig
from tqdm.auto import tqdm

from sparse_prefix_caching.datasets.base import Dataset

log = logging.getLogger(__name__)


class ShareGPTDataset(Dataset):
    """Chat dataset where each request is a prefix up to a user turn."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.conv_tokens = {}   # url -> list[list[int]]
        self._requests = []     # list of (url, turn, n_msgs)

    # -- prepare -------------------------------------------------------------

    def prepare(self, tokenizer) -> None:
        cfg = self.cfg
        log.info("Loading %s...", cfg.raw_path)
        df = pl.read_csv(cfg.raw_path)
        df = df.with_columns(
            pl.col("message_create_time").str.replace("ts:", "").cast(pl.Float64).alias("ts")
        )
        df = df.filter(pl.col("ts").is_not_null())
        df = df.sort(["url", "message_index"])

        # Cap rows early
        df = df.head(cfg.max_rows)

        # Limit conversations
        urls = df["url"].unique(maintain_order=True).to_list()[:cfg.max_convs]
        df = df.filter(pl.col("url").is_in(urls))

        # Limit rounds per conversation
        df = df.with_columns(
            pl.col("message_index").rank("ordinal").over("url").alias("_rank")
        ).filter(pl.col("_rank") <= cfg.max_rounds).drop("_rank")
        log.info("Filtered: %d conversations, %d rows", df.n_unique("url"), len(df))

        # format: '<|role|> text' with \n separator
        df = df.with_columns(
            (pl.col("message_index") == pl.col("message_index").min().over("url")).alias("_is_first")
        )
        df = df.with_columns(
            (pl.when(pl.col("_is_first"))
             .then(pl.lit("<|") + pl.col("role") + pl.lit("|> ") + pl.col("plain_text").fill_null(""))
             .otherwise(pl.lit("\n<|") + pl.col("role") + pl.lit("|> ") + pl.col("plain_text").fill_null(""))
            ).alias("_formatted")
        ).drop("_is_first")

        # tokenize in chunks
        log.info("Tokenizing messages...")
        formatted = df["_formatted"]
        all_tokens = []
        for i in tqdm(range(0, len(formatted), cfg.tokenizer_chunk_size), desc="tokenizing chunks"):
            chunk_texts = formatted[i:i + cfg.tokenizer_chunk_size].to_list()
            for ids in tokenizer(chunk_texts, add_special_tokens=False)["input_ids"]:
                all_tokens.append(np.array(ids, dtype=np.int32))
            del chunk_texts
        df = df.drop("_formatted", "plain_text").with_columns(pl.Series("tokens", all_tokens))
        del all_tokens

        # truncate by cumulative token count
        df = df.with_columns(
            pl.col("tokens").list.len().cum_sum().over("url").alias("cum_tokens")
        )
        n_before = len(df)
        df = df.filter(pl.col("cum_tokens") <= cfg.max_seq_len)
        log.info("Tokenized %d messages (%d truncated)", len(df), n_before - len(df))

        conv_total = df.group_by("url").agg(pl.col("tokens").list.len().sum().alias("total_tokens"))
        short_convs = conv_total.filter(pl.col("total_tokens") < cfg.min_seq_len)["url"]
        n_before = df.n_unique("url")
        df = df.filter(~pl.col("url").is_in(short_convs))
        log.info("Dropped %d conversations shorter than %d tokens", n_before - df.n_unique("url"), cfg.min_seq_len)

        out_path = Path(cfg.processed)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_path)
        log.info("Saved %d messages to %s", len(df), out_path)

    # -- load / access -------------------------------------------------------

    def _load(self) -> None:
        df = pl.read_parquet(self.cfg.processed)

        self.conv_tokens = {}
        conv_msg_indices = {}
        for url in df["url"].unique(maintain_order=True).to_list():
            conv = df.filter(pl.col("url") == url).sort("message_index")
            self.conv_tokens[url] = conv["tokens"].to_list()
            conv_msg_indices[url] = conv["message_index"].to_list()

        user_rows = df.filter(pl.col("role") == "user").sort("ts")
        self._requests = []
        turn_counter = defaultdict(int)
        for url, msg_idx in zip(
            user_rows["url"].to_list(), user_rows["message_index"].to_list()
        ):
            n_msgs = conv_msg_indices[url].index(msg_idx) + 1
            turn = turn_counter[url]
            turn_counter[url] += 1
            self._requests.append((url, turn, n_msgs))

        log.info("Loaded %d requests from %s", len(self._requests), self.cfg.processed)

    def conv_id(self, request) -> str:
        return request[0]

    def turn(self, request) -> int:
        return request[1]

    def get_tokens(self, request) -> list[int]:
        url, turn, n_msgs = request
        return [t for toks in self.conv_tokens[url][:n_msgs] for t in toks]
