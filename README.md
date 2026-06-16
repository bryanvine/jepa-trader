# JEPA-Trader

Self-supervised **Joint-Embedding Predictive Architecture (JEPA)** representation
learning for financial market microstructure, evaluated as a *feasibility study*
for profitable trading signals.

> **Research question.** Do JEPA representations — pretrained to predict *future
> market state in latent space* — yield more predictive, more sample-efficient,
> and more economically profitable trading signals than supervised and classical
> baselines, on real limit-order-book data?

This is a greenfield research project. The primary deliverable is a rigorous,
honestly-reported study (see [`paper/RESEARCH_LOG.md`](paper/RESEARCH_LOG.md))
suitable for a graduate-level paper.

## Data
- **Primary:** 10-level limit-order-book (LOB) snapshots for **SPY + QQQ**
  (and 140 other US equities), Nov–Dec 2025, sourced from an IBKR paper-trading
  system (`/apps/trading-system`).
- **Out-of-time test:** fresh LOB (Mar–Jun 2026) from the live TimescaleDB.
- **Lower-frequency arm:** 1 year of 15m/1h/1d OHLCV bars (≈450 symbols).
- **Transfer arm:** crypto 15m/1h bars, 70–82 symbols, 2024–2026.

See the research log for exact paths, schemas, row counts, and the leakage-control
methodology.

## Environment
- GPU: RTX 5060 (8 GB) for prototyping; scalable to RTX 4080 Super / Arc B70 /
  borrowed RTX 6000 Pro / vast.ai for the large pretraining runs.
- PyTorch 2.11 + CUDA 13.0 (bf16, ~42 TFLOP/s sustained on the 5060).

## Setup
```bash
python3 -m venv .venv --system-site-packages   # inherit working CUDA torch
.venv/bin/pip install -r requirements.txt
```

## Layout
```
configs/        # YAML experiment configs (data / model / train)
src/jepa_trader/
  data/         # DB export, LOB feature extraction, windowed datasets, splits
  models/       # PatchTST-style encoder, EMA target, predictor, JEPA module
  train/        # self-supervised pretraining loop
  eval/         # probes, baselines, metrics (IC/acc/R2), backtest
  utils/        # config, logging, seeding
scripts/        # CLI entrypoints (export -> build -> pretrain -> probe -> backtest)
paper/          # RESEARCH_LOG.md (live), figures, eventual paper draft
experiments/    # checkpoints + results (gitignored)
```

## Status
Phase 0 (data inventory) complete. See research log for the running plan & results.
