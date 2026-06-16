#!/usr/bin/env python
"""
Paper 3, Arm S data — FinBERT-score the FNSPID Benzinga headlines (~13M, 2009-2020)
into a leak-safe daily (date, symbol) sentiment panel.

We score UNIQUE titles only (Benzinga is heavily templated) then join back, so FinBERT
runs on far fewer than 13M rows. Sentiment = P(positive) - P(negative) in [-1, 1].
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/raw_fnspid/All_external.csv"
OUT = ROOT / "data/raw_fnspid"


def main():
    print("scanning news (date, title, symbol)...")
    df = (pl.scan_csv(SRC, ignore_errors=True)
          .select([pl.col("Date").str.slice(0, 10).alias("date"),
                   pl.col("Article_title").alias("title"),
                   pl.col("Stock_symbol").alias("sym")])
          .filter(pl.col("title").is_not_null() & pl.col("sym").is_not_null()
                  & (pl.col("date") >= "2009-01-01") & (pl.col("date") <= "2020-12-31"))
          .collect(engine="streaming"))
    print(f"  rows={df.height:,}  symbols={df['sym'].n_unique():,}  dates={df['date'].min()}..{df['date'].max()}")

    titles = df["title"].unique().to_list()
    print(f"  unique titles to score: {len(titles):,}  (vs {df.height:,} rows)")

    tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to("cuda").eval()
    id2 = model.config.id2label
    pos_i = [i for i, l in id2.items() if l.lower().startswith("pos")][0]
    neg_i = [i for i, l in id2.items() if l.lower().startswith("neg")][0]
    print(f"  finbert labels={id2}  (pos={pos_i}, neg={neg_i})")

    scores = np.empty(len(titles), np.float32)
    B = 512; t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(titles), B):
            batch = titles[i:i + B]
            enc = tok(batch, padding=True, truncation=True, max_length=48, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = torch.softmax(model(**enc).logits.float(), dim=-1)
            scores[i:i + B] = (p[:, pos_i] - p[:, neg_i]).cpu().numpy()
            if i % (B * 200) == 0:
                done = i + len(batch); rate = done / (time.time() - t0)
                print(f"  scored {done:,}/{len(titles):,}  ({rate:.0f}/s, eta {(len(titles)-done)/max(rate,1)/60:.0f}m)", flush=True)

    scored = pl.DataFrame({"title": titles, "sent": scores})
    df = df.join(scored, on="title", how="left")
    panel = (df.group_by(["date", "sym"])
             .agg([pl.col("sent").mean().alias("sent"), pl.len().alias("n_articles")])
             .sort(["date", "sym"]))
    panel.write_parquet(OUT / "sentiment_panel.parquet")
    print(f"\nsaved panel: {panel.height:,} (date,symbol) cells -> data/raw_fnspid/sentiment_panel.parquet")
    print(f"  date range {panel['date'].min()}..{panel['date'].max()}  symbols {panel['sym'].n_unique():,}")
    print(f"  sentiment mean {panel['sent'].mean():.3f}  std {panel['sent'].std():.3f}")


if __name__ == "__main__":
    main()
