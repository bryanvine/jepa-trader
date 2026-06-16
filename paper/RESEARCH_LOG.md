# JEPA-Trader — Research Log

**A feasibility study of Joint-Embedding Predictive Architectures for
profitable trading on limit-order-book microstructure.**

- **Started:** 2026-06-15
- **Hardware:** RTX 5060 (8 GB) primary; RTX 4080 Super (16 GB), Intel Arc B70 Pro
  (32 GB), borrowable RTX 6000 Pro (weekends), vast.ai rental available for scale-up.
- **Stack:** PyTorch 2.11.0 + CUDA 13.0 (bf16, ~42 TFLOP/s sustained on the 5060).
- **Goal:** Test whether JEPA-style self-supervised pretraining produces
  representations that beat supervised + classical baselines on (a) predictive
  metrics, (b) sample efficiency, and (c) net-of-cost backtested profitability —
  and to write this up as a rigorous, honestly-reported graduate-level paper.

This log is the living backbone of the paper. It is append-mostly: design
decisions are dated; results tables are filled as experiments complete.

---

## 1. Motivation & Research Questions

Generative/forecasting models of markets predict in *input space* (next price,
next return), where the irreducible noise of efficient markets dominates the
signal and forces the model to "waste capacity" modelling unpredictable detail.
**JEPA** (LeCun, 2022; I-JEPA, Assran et al. 2023; V-JEPA, Bardes et al. 2024)
instead predicts in *latent space*: a context encoder and an EMA "target" encoder
embed two views of the data, and a predictor maps context embeddings to target
embeddings. The loss lives entirely in representation space, so the model is free
to discard unpredictable nuisance detail and keep only what is predictable —
exactly the property we want for noisy financial series.

**Primary RQ.** Do JEPA representations of LOB microstructure yield better trading
signals than supervised-from-scratch and classical baselines?

Sub-questions / hypotheses:
- **H1 (predictive).** A frozen JEPA encoder + linear probe achieves higher
  Information Coefficient (IC) and directional accuracy on forward returns than
  an order-flow-imbalance (OFI) linear model and matches/beats a supervised
  encoder trained end-to-end on labels.
- **H2 (sample efficiency).** JEPA's advantage grows as labelled data shrinks
  (1% / 10% / 100% label regimes) — the canonical SSL selling point.
- **H3 (economic).** A simple threshold policy on the JEPA signal produces a
  positive net-of-cost Sharpe at some horizon, after subtracting half-spread +
  fees; and decays gracefully (not catastrophically) with horizon.
- **H4 (generalization).** Representations pretrained on Nov–Dec 2025 transfer to
  a *held-out later period* (Mar–Jun 2026) and to *held-out symbols* without
  collapse — evidence the model learned microstructure, not a period artifact.
- **H0 (honest null).** It is entirely possible JEPA does *not* beat baselines
  net of cost. A well-powered negative result, clearly reported, is a valid and
  publishable outcome of a feasibility study.

---

## 2. Data Inventory

All sources below were verified directly (file shapes, DB queries) on 2026-06-15.

### 2.1 Primary — raw 10-level LOB parquet (Nov–Dec 2025)
- **Path:** `/apps/trading-system/data/training/lob_YYYYMMDD.parquet` (29 files,
  one per trading day), `manifest.json` alongside.
- **Coverage:** 142 US equities (incl. **SPY, QQQ**), 2025-11-10 → 2025-12-21
  (29 trading days), **56,541,625 rows**, zstd-compressed (~4.6 GB).
- **Schema:** `time, symbol, bid_price_1..10, bid_size_1..10, ask_price_1..10,
  ask_size_1..10` — full 10-level book per snapshot.
- This is the **canonical source** for the core study (most complete, contiguous,
  multi-symbol LOB window). We build our own features from it for full provenance.

### 2.2 Pre-derived feature matrix (convenience; not canonical)
- **Path:** `/apps/trading-system/data/training_ready/`
- `features.npz` → key `X`, shape **(46,135,422, 29)** float32 (~5.3 GB in RAM);
  29 named microstructure features (spread_bps, imbalance_1..5, microprice_offset,
  weighted_imbalance, book_pressure, bid/ask_dist_1..5, bid/ask_size_log_1..5).
- `returns/horizon_XXXX.npz` → key `returns_bps`, shape (46.1M,) float32, for
  **496 horizons (5..500 snapshots ahead)**, in **basis points**, **NaN at segment
  boundaries** (forward window crosses a symbol/day edge). ~41 GB total.
- ⚠️ **No row→(symbol,day) index is stored.** Opaque boundaries are a leakage
  risk, so we do **not** treat this as canonical; we rebuild from §2.1 parquet
  with explicit indexing. (We may still use it for fast sanity baselines.)

### 2.3 Live TimescaleDB — `trading` (container `trading-timescaledb`)
Postgres 15 + TimescaleDB, **internal port 16432** (socket
`/run/postgresql/.s.PGSQL.16432`; not the default 5432). Access via
`docker exec ... psql` using the container's `POSTGRES_*` env (see
`src/jepa_trader/data/db.py`). Relevant hypertables (verified 2026-06-15):

| table | coverage | notes |
|---|---|---|
| `lob_snapshots` | **2026-03-16 → 2026-06-11** | 10-level + precomputed `mid_price`,`spread`; recently **SPY-focused**. → **out-of-time test set** (H4). |
| `bars` (OHLCV) | **2025-06-12 → 2026-06-12** | multi-timeframe, full year → **lower-freq JEPA arm** & regime diversity. |
| `quotes` (L1) | 2026-05-14 → 2026-06-15 | 21M rows / 4.4 GB, ~1 month best bid/ask. |
| `ticks` | 2025-12-10 → 2025-12-12 | tiny. |
| `order_book` | empty | — |
| `options_snapshots` | 2026-02-23 → 2026-05-08 | optional later. |

`bars` timeframe breakdown: `15m` 2.23M rows/449 sym; `1h` 592k/448; `1d` 399k/1636;
`5m` 5.07M (→Dec 2025); `1m` 3.59M (Nov–Dec 2025); `2h` 339k; `4h` 205k.

### 2.4 Live TimescaleDB — `crypto` (container `crypto-timescaledb`)
`bars` hypertable, fresh to today — **transfer arm**:
`15m` 4.43M rows / 70 sym / 2024-02-25→2026-06-15; `1h` 1.03M / 82 sym;
`1d` 37k / 82 sym; `4h` 161k (→Feb 2026, stale). Far richer than the on-disk
`/apps/crypto-trader/data/historical/*.csv` (which had only 1h/4h/1d).

### 2.5 Methodological implications (these shape every experiment)
1. **Effective N ≪ nominal N.** 46M LOB rows span only ~6 weeks and are heavily
   autocorrelated within each day. We report *both* raw counts and a notion of
   effective sample size, and never quote 46M as if it were i.i.d.
2. **Splits are by time and by symbol — never random row shuffles.** Default core
   split: train = first ~19 days, val = next ~4, test = last ~6 (by calendar
   date), plus a **symbol-held-out** variant. Out-of-time test = §2.3 LOB
   (Mar–Jun 2026).
3. **Normalization is fit on train only** (per-feature robust scaling), applied to
   val/test. No statistic may see the future.
4. **Label NaNs at segment boundaries are masked** in every loss/metric.
5. **Windows never cross (symbol, day) boundaries.** Enforced by the explicit
   index we build in Phase 1.

---

## 3. Compute Environment
- RTX 5060, 8.08 GB VRAM, bf16 supported, ~42 TFLOP/s sustained (4096³ matmul
  measured at 3.25 ms). Host: 32 cores, 62 GB RAM (~29 GB free).
- Scale-up path for the large pretraining run: RTX 4080 Super (16 GB) → Arc B70
  Pro (32 GB, needs IPEX/XPU validation) → borrowed RTX 6000 Pro (weekends) →
  vast.ai. All code is device-agnostic and config-driven (batch/model sizes in
  YAML) so a weekend run can scale up without code changes.

---

## 4. Method — Time-Series JEPA (TS-JEPA)

### 4.1 Input representation
A **window** of `L` consecutive LOB-feature vectors from a single (symbol, day)
segment, shape `(L, F)` with `F=29`. We **patchify** along time (PatchTST-style):
non-overlapping patches of length `P`, each linearly embedded to dimension `D`,
plus learned positional encodings → a sequence of `L/P` patch tokens.

### 4.2 Architecture
- **Context encoder** `f_θ`: Transformer encoder over patch tokens (default
  `D=192`, depth 6, heads 6, MLP ratio 4) — sized for 8 GB at bf16.
- **Target encoder** `f_ξ`: same architecture; weights are an **EMA** of `f_θ`
  (momentum schedule 0.996→1.0), **stop-gradient**. This is the I-JEPA/BYOL
  anti-collapse mechanism (no negatives needed).
- **Predictor** `g_φ`: narrow Transformer (D≈96, depth 4) that takes the context
  tokens + mask/position tokens for the target locations and predicts target
  embeddings.

### 4.3 Objective
Smooth-L1 (or normalized L2) between predicted target embeddings and the EMA
target encoder's embeddings of the target patches (the targets are
layer-normalized, as in I-JEPA). Loss masked to valid (non-boundary) patches.

### 4.4 Anti-collapse & diagnostics
EMA target + stop-grad is the primary guard. We additionally **monitor** (and can
add as a VICReg-style regularizer in ablation): per-dimension embedding variance,
effective rank of the embedding covariance, and pairwise prediction variance. A
collapse (variance→0 / rank→1) invalidates a run and is reported, not hidden.

### 4.5 Variants (studied)
- **(A) Causal latent-forecasting (lead).** Context = first part of the window;
  targets = *future* patches. Predict the future in latent space. Cleanest trading
  story; at inference the encoder only ever sees the past. **This is the headline
  method.**
- **(B) Masked-block (I-JEPA-style).** Mask several contiguous blocks anywhere in
  the window, predict them from the visible context. Stronger general
  representation; used for representation-quality comparison.

---

## 5. Experimental Design

### 5.1 Downstream tasks
For target horizons `h ∈ {10, 30, 100, 300}` snapshots (a sweep, to show how
predictability and net-of-cost profit decay with horizon):
- **Regression** of forward return (bps).
- **3-class direction** (down / flat / up) with a dead-band around 0.

### 5.2 Baselines (must beat these to claim anything)
1. **Naïve:** predict 0 / predict last-sign / persistence.
2. **OFI linear:** order-flow-imbalance → return, the canonical microstructure
   alpha; logistic/linear.
3. **XGBoost** on the 29 features (strong tabular baseline).
4. **Supervised-from-scratch:** identical encoder trained end-to-end on labels
   (isolates the *pretraining* contribution).
5. (If reusable) the existing **DeepLOB** CNN+LSTM from the trading system.

### 5.3 Metrics
- Predictive: **IC** (Spearman & Pearson of prediction vs realized return),
  directional accuracy / balanced accuracy, R².
- Representation: linear-probe accuracy, embedding variance / effective rank,
  kNN-retrieval coherence.
- Economic (§5.4).

### 5.4 Backtest protocol (the trading test)
Event-driven over the test period, per symbol. Signal → position in {−1,0,+1}
via a threshold on the predicted return. **Costs:** cross the half-spread on
entry/exit (we *have* the book, so this is realistic) + a configurable per-share
/ bps fee. Report **net** cumulative PnL, annualized Sharpe, hit-rate, turnover,
average holding time, and PnL-per-trade vs cost-per-trade. We also report the
**break-even cost** (the fee at which Sharpe→0) as an honest robustness number.

### 5.5 Ablations
Masking ratio; EMA momentum; patch length `P`; embed dim `D`; predictor depth;
causal (A) vs masked (B); +/− VICReg term; window length `L`; pretrain compute.

### 5.6 Generalization tests (H4)
- **Symbol-held-out:** pretrain on SPY, probe on QQQ (and vice-versa); later, the
  142-symbol universe.
- **Out-of-time:** pretrain on Nov–Dec 2025, evaluate on Mar–Jun 2026 LOB (§2.3).
- **Cross-asset (exploratory):** equities→crypto bars.

---

## 6. Decisions Log

- **2026-06-15.** Project bootstrapped. Verified data + compute (this doc §2–3).
  Decisions: (i) **SPY+QQQ core, then expand** to 142 universe; (ii) optimize for
  a **rigorous feasibility study** (clean methodology, strong baselines,
  ablations, honest reporting) over raw profit-chasing; (iii) **build our own
  leak-free pipeline from raw parquet** rather than trust the opaque
  `training_ready.npz`; (iv) lead with the **causal latent-forecasting** JEPA
  variant; (v) treat DB `lob_snapshots` (Mar–Jun 2026) as an **out-of-time test
  set** and DB `bars` (1 yr) as a **lower-frequency arm**; (vi) device-agnostic,
  config-driven code for GPU scale-up.
- **2026-06-15 (Phase 1).** Built the SPY+QQQ dataset from raw parquet
  (`src/jepa_trader/data/`). Discovered & fixed two data issues: (a) **QQQ
  2025-12-17** was captured at event/sub-100ms rate (15.6M rows/day vs ~340k
  normal) → adopted **uniform 100 ms grid resampling** (last snapshot per bucket)
  for *all* segments, which both collapses anomalies (QQQ 12-17 → 229k) and makes
  "horizon = h × 100 ms" exactly true; (b) data actually spans **Nov 10 – Dec 19**
  with a Dec 6–10 gap (weekend + excluded 12-08/09), so the split is
  train ≤ Dec 2 / val Dec 3–5 / test Dec 11–19, with the gap conveniently between
  val and test. Also found the parquet contains **89–91 symbols on later days**
  (full 142-universe data is present for the Phase-5 expansion).
- **2026-06-15 (Phase 2).** Implemented TS-JEPA (PatchTST patch-embed + Transformer
  context encoder, EMA target, Transformer predictor; latent smooth-L1). Verified it
  trains and does **not collapse** (tgt_std ~0.8-0.97); ~20k windows/s, **GPU peak
  < 1.3 GB** (huge headroom on 8 GB). Two findings that shape the method:
  - **(F1) Pooling dominates probe quality.** Alpha lives in the *most recent* book
    state, so **last-token pooling** gives IC~0.162 @0.1s vs ~0.03 for mean-pooling.
    -> use last/concat pooling; mean-pooling is a weak baseline.
  - **(F2) Causal latent extrapolation is unstable.** Predicting the whole future half
    makes the EMA target variance saturate while the predictor lags and val-loss rises
    (far-future mid ~ random walk -> target largely unpredictable). **Decision:** make
    the canonical **block/masked (interpolation)** variant the primary run; keep
    causal-vs-block as a documented ablation (masked interpolation >> causal
    extrapolation on near-random-walk price data is itself a useful negative result).

---

## 7. Results

### 7.0 Built dataset — SPY+QQQ v1 (`data/spy_qqq_lob`)
100 ms grid, RTH (14:30–21:00 UTC), 29 features, horizons {1,5,10,30,60,100,300,600}
steps (= 0.1 s … 60 s). **10,092,894 rows / 50 (symbol,day) segments.** Split:
train 6.95M (32 seg) / val 1.29M (6) / test 1.86M (12). Windowed (L=128, stride=8):
**868k / 161k / 232k** windows. DataLoader ≈ 80k windows/s (4 workers). Label std
grows diffusively: 0.20 bps @0.1 s → 0.64 @1 s → 2.07 @10 s → 5.04 @60 s. At 0.1 s
only 27% of returns are > 0 (large "flat"/zero mass) → 3-class probes need a
dead-band. Half-spread ≈ 0.073 bps (SPY penny-wide).

### 7.1 Baseline — order-flow imbalance (Spearman IC, test set)
The signal JEPA must beat. Single-snapshot features; pooled over SPY+QQQ test rows
(200k sample). Predictability is real but **concentrated at sub-second horizons and
decays fast** (efficient-market microstructure):

| feature \ horizon | 0.1 s | 1 s | 3 s | 10 s | 30 s | 60 s |
|---|---|---|---|---|---|---|
| `imbalance_1` (L1 OFI) | **+0.171** | +0.062 | +0.037 | +0.019 | +0.011 | +0.007 |
| `microprice_offset` | +0.160 | +0.058 | +0.035 | +0.017 | +0.010 | +0.006 |
| `book_pressure` | +0.081 | +0.027 | +0.017 | +0.004 | −0.005 | −0.002 |
| `weighted_imbalance` | +0.061 | +0.019 | +0.013 | +0.001 | −0.006 | −0.004 |

*Hypothesis for JEPA's edge:* a single snapshot's imbalance is memoryless; encoding
the **temporal evolution** of the book over a 128-step (12.8 s) window should lift
IC at the 1–10 s horizons where instantaneous OFI has already decayed.

### 7.2 Downstream — predictive IC, learned models (test set, last-pooling)
Run `jepa_block_v3` (block/masked JEPA, L=128, P=8, D=192, depth 6, 3.56M params,
12k steps). Frozen-encoder ridge probe, alpha selected on val. Spearman IC:

| method | 0.1 s | 0.5 s | 1 s | 3 s | 6 s | 10 s | 30 s | 60 s |
|---|---|---|---|---|---|---|---|---|
| **jepa_emb (frozen)** | +0.158 | +0.074 | +0.048 | +0.027 | +0.009 | +0.002 | −0.013 | −0.012 |
| raw_last (ridge, 29 feat) | +0.172 | +0.083 | +0.055 | +0.037 | +0.023 | +0.020 | +0.004 | +0.007 |
| ofi_imb1 (single feature) | +0.168 | +0.083 | +0.057 | +0.036 | +0.021 | +0.020 | +0.010 | +0.009 |
| raw_flat (ridge, whole window) | +0.140 | +0.057 | +0.032 | +0.011 | +0.005 | +0.008 | +0.005 | +0.001 |
| supervised (end-to-end, same enc) | +0.162 | +0.078 | +0.049 | +0.029 | +0.016 | +0.009 | −0.007 | −0.016 |

**Reading:** (1) **Frozen JEPA ≈ supervised end-to-end** (+0.158 vs +0.162 @0.1 s,
near-identical across all horizons) — the *label-free* representation matches a model
trained directly on the labels (a clean positive for the SSL premise). (2) But **both
deep models slightly trail the linear OFI/raw baseline** (+0.17), and `raw_flat`
(whole-window ridge) is *worse* than the last snapshot. ⇒ at 0.1–60 s the SPY/QQQ LOB
signal is **near-Markovian and essentially linear**: the current book state dominates,
history and nonlinearity add no exploitable edge at full data.

### 7.3 Economic backtest — net of spread + fees (test set)
Non-overlapping holds; round-trip = cross full spread (~0.15 bps) + 2×0.1 bps fee.
**Mean net bps/trade** (negative = loses money), top-5%-selectivity column shown:

| method | 1 s | 10 s | 30 s | 60 s |
|---|---|---|---|---|
| jepa_emb | −0.40 | −0.40 | −0.50 | −0.81 |
| ofi_imb1 | −0.34 | −0.33 | −0.21 | **+0.22** |
| raw_last | −0.38 | −0.35 | −0.30 | **+0.12** |

**Headline economic result (honest negative):** the microstructure edge is real
(IC up to 0.17) but **fully consumed by the bid–ask spread** for a liquidity-*taking*
strategy. Net P&L/trade is negative at essentially all horizons for all methods;
only the simple OFI/raw baselines reach marginal break-even at 60 s under extreme
selectivity, and **JEPA does not**. (Per-trade annualized Sharpe magnitudes are an
artifact of HFT trade counts — sign only; we lead with net bps/trade + break-even.)

### 7.4 Interpretation & what would change the verdict
A liquidity-taking JEPA strategy at sub-minute horizons on liquid ETFs is
**structurally unprofitable** — consistent with market efficiency and how real HFT
actually earns (providing liquidity / latency / rebates, not crossing the spread).
The feasibility of JEPA for *profitable* trading therefore hinges on regimes our
HFT slice can't express, and on SSL's known strengths — the next experiments:
1. **Sample efficiency** (H2): JEPA-frozen vs supervised at 1%/10%/100% labels —
   the canonical SSL win, independent of the cost wall.
2. **Lower-frequency / larger-move arm**: the 1-year 15 m/1 h **bars** (moves ≫ spread)
   — the regime where net-of-cost profit is actually plausible.
3. **Liquidity provision framing**: earn the spread instead of paying it.
4. **Scale** the encoder/pretraining on a larger GPU (4080S / borrowed RTX 6000 Pro).

### 7.5 Sample efficiency (H2) — test IC @0.1 s vs # labeled training windows
| # labels | jepa_emb (frozen) | raw_last (linear) | supervised (deep) |
|---|---|---|---|
| 868 | 0.027 | **0.118** | — |
| 8.7k (1%) | 0.067 | **0.161** | 0.038 |
| 87k (10%) | 0.142 | **0.167** | 0.143 |
| 868k (100%) | 0.160 | **0.173** | 0.162 |

**Finding:** JEPA **beats supervised-deep in the low-label regime** (0.067 vs 0.038 @1%)
and reaches near-full IC with ~10% of labels — the SSL label-efficiency win (H2) is real.
**But** the 29-feature linear ridge dominates at *every* label count. ⇒ For a
near-linear+Markovian signal, deep SSL's advantage over supervised is **moot** — the
correct model is linear. (H2 holds vs supervised; yields no practical edge here.)

### 7.6 Lower-frequency arm — 15 m bars (`data/bars_15m`)
Built from live DB `bars` (25 stationary OHLCV features; **448 symbols, dense Jun 2025–
Feb 2026**; after Feb the scan universe collapses to ~82 sym, so capped). Split by month:
train 1.59M (Jun–Dec) / val 235k (Jan) / test 234k (Feb). Single-feature baseline IC is
weak (|IC| ≤ 0.035 — mild RSI/SMA mean-reversion, vol→return at long h) **but label std is
50–400 bps vs a ~2 bps spread** — the cost/move ratio is ~100× friendlier than the HFT
slice, so even a weak *combined* signal could be net-positive. JEPA (window 64 = 16 h,
block, 10k steps).

**Predictive (test = Feb 2026, IC):** weak, as expected — `raw_last` ridge best at
longer horizons (h32=8 h: +0.047, h64=16 h: +0.038); `jepa_emb` weaker (+0.025–0.029);
`raw_flat` best at 15 m (+0.055). JEPA again does not beat the linear baseline.

**Economic (net of 2 bps spread + 0.5 bps/side fee), trade-EVERY-signal (reliable, ~3k–25k trades):**
modestly **positive net bps/trade at longer horizons** — `raw_last`: +1.2@8 h, +4.6@8 h,
+4.6@16 h; `jepa_emb` +7.2@16 h (hit ~0.51–0.53). The favorable cost/move ratio *does*
flip the economics vs the HFT slice.

⚠️ **Reliability caveat (critical, honest):** the test is **a single month (Feb 2026)**.
The eye-popping *selective* cells (e.g. raw_last +102 bps/trade @16 h, trade_frac 0.1)
come from only **~300 trades at a 16 h horizon within one regime** — heavily
time/cross-section-correlated, ≈a handful of independent periods. **Not trustworthy.**
The broad-trading positives are more believable but still one month. **A profitability
claim requires walk-forward validation across the 9-month dense window** (multiple
independent test folds) — the clear next step for the bars arm.

**Net so far:** the bars arm is *more promising* than HFT (broad-trade net is positive,
not negative) but **profitability is not yet demonstrated**; JEPA ≤ linear here too.

### 7.7 Bars **walk-forward** — 4 independent test months (Nov 2025–Feb 2026)
*Rigorous re-test.* Encoder pretrained ONCE on Jun–Sep (frozen, leak-free); ridge probe
refit per fold on pre-test data only; 4 expanding folds pooled. Net bps/trade
(trade-every; 2 bps spread + 0.5 bps/side fee), horizons in wall-clock:

| method | 15 m | 1 h | 4 h | 8 h | 16 h |
|---|---|---|---|---|---|
| jepa_emb | −1.8 | −1.9 | −3.1 | +4.5 | +9.0 |
| raw_last | −1.0 | −0.6 | +0.6 | +1.7 | +1.0 |

Pooled IC ≤ 0.06 (raw_last ≥ jepa_emb). Per-month @4 h: raw_last +0.3/+1.2/+1.0/−0.2
(3/4 positive, ~+1 bp); jepa_emb mostly negative.

**Verdict:** the first-pass **+102 bps was single-period overfitting — it does NOT survive
walk-forward.** `raw_last` shows at best a marginal, fragile ~+1 bp/trade at multi-hour
horizons (within cost-assumption noise, not deployable); `jepa_emb`'s lone +9 bps @16 h
rides a noisy IC (0.06). **No robust net-of-cost edge; JEPA ≤ linear.**

### 7.8 Cross-regime conclusion (working thesis)
Across HFT-LOB **and** lower-frequency bars, JEPA produces **valid label-free
representations that match supervised learning but do not beat simple linear models**,
and **no tradeable net-of-cost alpha is demonstrated** — HFT is killed by the spread;
bars are within-noise under walk-forward. A clean, honest feasibility result that
separates "the SSL works" from "there is exploitable alpha here." Remaining levers
(scale, L3/market-making framing, futures/crypto regimes) are scoped in §7.4/§8.

### 7.9 Scaling negative control (RTX 4080 Super)
Trained a **6× larger** encoder (dim 384, depth 10, **20.7M params**, 20k steps, batch
2048) on the same SPY+QQQ LOB. Frozen-probe test IC@0.1 s = **+0.156** vs the 3.5M-param
model's +0.158 and the linear baseline's +0.172 — **identical; scaling does not help**
(marginally worse at 3–10 s). ⇒ The ceiling is the **signal** (near-linear, Markovian),
not model capacity — a clean negative control that strengthens the thesis. (Multi-GPU
infra now live: 5060 + 4080 Super over SSH; uv-managed remote env, rsync deploy.)

---

### 7.10 Multi-modal arm — does news sentiment predict returns? (premise check)
Tested against the trading DB news feed (**346k articles, 197 symbols, Jan–Jun 2026,
100% LLM-sentiment-scored**, with event taxonomy earnings/guidance/analyst/macro).
Leak-safe daily panel (`scripts/07_sentiment_baseline.py`): aggregate sentiment per
(symbol, day) → align each news day to the FIRST trading day strictly after it → label =
close-to-close forward return. **News sentiment is a *contrarian* predictor — IC is
negative every month Jan–Jun** (directionally persistent), strongest as the
symbol-demeaned *sentiment surprise* `s_surp`.

⚠️ **But month-by-month is essential** (lesson from the bars +102 bps): a single test
month showed `s_surp`@2–3 d IC ≈ **−0.13**, yet **pooled across 6 months it is only
≈ −0.04** — the strong months were small-sample (Mar–Apr, ~130 rows) or a specific
regime (May–Jun). Honest read: a **weak but directionally-persistent** contrarian signal
(pooled |IC| ≈ 0.04 @2–3 d).

**Why it still matters:** this is the **first non-price signal with a persistent sign** in
the whole study, and it lives at a **daily horizon where costs (~2–4 bps) are negligible
vs 300–480 bps moves** — so even a weak edge could be economically real, unlike the
cost-dominated microstructure. Partially validates the multi-modal premise (information
> microstructure for *capturable* alpha). **Next:** (1) walk-forward backtest of the
contrarian sentiment strategy (is it tradeable?); (2) **price + sentiment JEPA fusion**
(does fusing beat either modality alone?); (3) crypto funding-carry arm. Code + first
result pushed to `github.com/bryanvine/jepa-trader`.

### 7.11 Sentiment — tradeability + fusion test (both negative-ish)
- **Tradeable standalone?** Causal cross-sectional long-short that *fades* sentiment
  surprise (`scripts/08_sentiment_backtest.py`). Pooled daily-cohort net ≈ +13 bps/cohort
  @4 bps (k=2) *looks* positive, but the honest **non-overlapping Sharpe is only +0.16
  @4 bps** (negative @8 bps), unstable across k, and dominated by ~1-cohort months
  (Mar/Apr) — only ~28 independent periods. **Not robustly tradeable** on this 5-month window.
- **Does fusion add value?** Feature-level ridge IC (test May–Jun, `scripts/09_fusion_features.py`):
  price-only **−0.02**, sentiment-only **+0.080**, price+sentiment **+0.006**. **Fusion
  HURTS** — daily price is efficient, so it only dilutes the sentiment signal. ⇒ the
  signal is entirely in the sentiment modality; a **price+sentiment JEPA fusion is low-EV**.
- **Multi-modal verdict:** sentiment is the lone non-price signal (weak, contrarian, daily),
  not standalone-tradeable on 5 months, and there is nothing in price to fuse. The remaining
  real shot at tradeable alpha is **crypto funding carry** (§7.12, in progress).

### 7.12 Crypto arm — funding doesn't predict price; carry too small to harvest
66–73 coins, 2024–2026, 8-hourly funding as-of-aligned to 1h bars
(`scripts/10_crypto_funding.py`, `11_crypto_carry.py`). (1) **Funding does NOT predict
forward price** (cross-sectional IC ≈ −0.003 at 8/24/72 h) — crypto price is efficient
like equities, so a crypto-price JEPA won't beat baselines. (2) **Carry backtest**
(short high-funding / long low-funding tercile, harvest + price, net): harvested spread
is only **+0.3 bps/8 h (~3%/yr gross)**, while the price term is **−4.7 bps/8 h** (high-
funding coins keep *rising* — funding tracks **momentum**, not reversal), so the fade
**loses** (Sharpe −1.6 at zero cost, worse with fees). Delta-neutral would isolate the
+0.3 bps but that's tiny and cost-eaten. ⇒ **No tradeable funding edge** on this universe.

### 7.13 Final research synthesis (research complete)
Tested every freely-available signal with leak-safe, walk-forward / multi-period rigor:
| arm | signal? | beats linear? | tradeable net of cost? |
|---|---|---|---|
| HFT LOB (0.1–60 s) | yes, strong but micro | no (JEPA ≈ supervised ≤ linear OFI) | **no** — spread-dominated |
| Bars 15 m (walk-forward) | weak | no | **no** — within-noise |
| Model scale (6×, 4080) | — | no change | — |
| News sentiment (daily) | **weak contrarian, sign-persistent** | (best non-price) | **no** — ~0.16 Sharpe, thin data |
| Price+sentiment fusion | — | fusion *hurts* (price is noise) | **no** |
| Crypto funding carry | ~none | — | **no** — carry tiny, fade loses to momentum |

**Thesis:** JEPA reliably learns valid label-free representations (≈ supervised), but
across frequency, asset class, modality, and scale it **never beats a simple linear
model**, and **no robust net-of-cost alpha** exists in these freely-available signals —
markets are efficient w.r.t. the data we can access for free; costs and signal-linearity
are the binding constraints, not model capacity. The lone bright spot (weak contrarian
news sentiment) is not standalone-tradeable on 5 months. A rigorous, honest,
publishable feasibility result. Untested levers requiring paid data / new framing:
L3 market-making (queue/fill), longer sentiment history, options/IV, futures MBO.

## 8. References (to expand)
- LeCun (2022), *A Path Towards Autonomous Machine Intelligence* (JEPA).
- Assran et al. (2023), *Self-Supervised Learning from Images with a
  Joint-Embedding Predictive Architecture* (I-JEPA).
- Bardes et al. (2024), *V-JEPA: Revisiting Feature Prediction for Video*.
- Nie et al. (2023), *A Time Series is Worth 64 Words* (PatchTST).
- Zhang et al. (2019), *DeepLOB: Deep Convolutional Neural Networks for LOB*.
- Bardes et al. (2022), *VICReg*.
