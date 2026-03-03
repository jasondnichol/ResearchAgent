# TradeSavvy Trading System — Complete Strategy Guide

A complete, plain-English explanation of how this trading system works, its current configuration, and projected returns.

**Last updated:** March 1, 2026
**Owner:** Jason Nichol

---

## System Overview

The system runs two independent bots on AWS EC2 (184.72.84.30), both paper trading on Coinbase:

1. **Donchian Bot** — Daily candles, tri-mode (Futures Long + Spot Long + Futures Short)
2. **Intraday Bot** — Hourly candles, MTF Momentum (experimental, proving itself)

All long trading is routed through **Coinbase CFM perpetual futures** (0.06% taker fee) as the primary execution venue, with Coinbase spot (0.50% fee) as a fallback for coins without perps. This all-futures approach is the single biggest return lever — 8x lower fees roughly double net returns.

---

## The Evolution: Why We're Here

### Phase 1: Hourly Regime-Based Trading (Deprecated)

The original system classified BTC into three regimes and ran a different strategy for each:

| Regime | Strategy | Result |
|--------|----------|--------|
| Ranging | Williams %R Mean Reversion | 54.9% WR, 2.09 PF |
| Trending | ADX Momentum Thrust | 50% WR, 1.92 PF |
| Volatile | BB Mean Reversion | 72.7% WR, 1.65 PF |

Looked great in backtests — **failed in practice** because Coinbase fees (~0.60% round-trip) were 12x larger than the average hourly trade gain of ~0.05%. A +57% backtest turned into -99% with realistic fees.

### Phase 2: Daily Donchian Breakout (Current)

Pivoted to daily timeframe. Average winning trade: +25.8% gain over 21-day hold. The 0.90% round-trip fee is negligible. The strategy waits for confirmed breakouts on strong volume and rides trends with a trailing stop.

### Phase 3: Optimization (Feb-Mar 2026)

Added pyramiding (+15% trigger), 4x ATR trailing stops, partial profit-taking, futures execution, and short-side trading. Optimized risk to 4% per trade. Combined result: +385% over 4 years on backtest.

---

## Current Strategy: Donchian Channel Breakout

### The Core Idea

The Donchian Channel tracks the highest high and lowest low over a lookback period. When price breaks above the 20-day high on strong volume, it signals institutional buying and the start of a potential trend. We ride that trend with a trailing stop that locks in gains.

### Three Trading Modes

| Mode | Priority | Exchange | Fees | Risk | When Active |
|------|----------|----------|------|------|-------------|
| **Futures Long** | 1st (primary) | Coinbase CFM Perps | 0.06%/side | 4% per trade | BTC > SMA(200) |
| **Spot Long** | 2nd (fallback) | Coinbase Spot | 0.50%/side | 4% per trade | BTC > SMA(200), coins without perps |
| **Futures Short** | 3rd | Coinbase CFM Perps | 0.06%/side | 2% per trade | Death cross (SMA50 < SMA200 + BTC < SMA200) |

Futures long is checked **first** in the daily loop. It claims position slots at 0.06% fees. Spot long only fires for coins not already entered via futures (currently just NEAR, which has no perp). Shorts only activate during confirmed bear markets.

---

## Entry Rules

### Long Entry (Futures Long + Spot Long)

ALL conditions must be true:

1. **Bull filter**: BTC daily close > 200-day SMA
2. **Breakout**: Close > previous 20-day Donchian high
3. **Volume**: Volume > 1.5x 20-day average
4. **Trend**: Price > 21-day EMA

### Short Entry (Futures Short)

ALL conditions must be true:

1. **Bear filter (death cross)**: BTC SMA(50) < SMA(200) AND BTC close < SMA(200)
2. **Breakdown**: Close < previous 10-day Donchian low
3. **Volume**: Volume > 2.0x 20-day average (higher threshold than longs)
4. **Trend**: Price < 21-day EMA

---

## Exit Rules

### Long Exits (any one triggers)

| Exit | Rule | Purpose |
|------|------|---------|
| **Trailing stop** | Price < high watermark - 4x ATR(14) | Primary exit, locks in gains |
| **Donchian exit** | Close < 10-day low | Catches trend reversals |
| **Emergency stop** | Price < entry - 15% | Hard floor for catastrophic moves |
| **Blow-off detection** | Volume > 3x avg AND RSI > 80 → tighten stop to 1.5x ATR | Captures parabolic tops |

### Short Exits (any one triggers)

| Exit | Rule | Purpose |
|------|------|---------|
| **Trailing stop** | Price > low watermark + 2x ATR(14) | Primary exit (inverted) |
| **Donchian exit** | Close > 15-day high | Catches trend reversals |
| **Emergency stop** | Price > entry + 15% | Hard floor |
| **Max hold** | 30 days | Prevents stale positions |

### Partial Profit Taking (Both Sides)

| Level | Trigger | Action |
|-------|---------|--------|
| TP1 | +10% gain (or -10% for shorts) | Sell 25% of position |
| TP2 | +20% gain (or -20% for shorts) | Sell 25% of position |
| Runner | Trailing stop hit | Sell remaining 50% |

---

## Pyramiding (Adding to Winners)

When a long position gains +15% AND price makes a new 20-day high:

- Add a **1% risk tranche** (smaller than the initial 4% entry)
- Maximum 1 pyramid add per position
- Only during bull market (bull filter active)
- Enabled for both futures longs and spot longs
- **Not enabled for shorts** (insufficient live validation)

Impact: Pyramiding boosted returns from +47.9% to +80.9% and improved PF from 1.72 to 2.73 in full-period backtest.

---

## Position Sizing & Capital

| Parameter | Current Value |
|-----------|---------------|
| Starting capital | $10,000 |
| Max concurrent positions | 4 (shared across ALL modes) |
| Long risk per trade | 4% of equity |
| Short risk per trade | 2% of equity |
| Pyramid risk per add | 1% of equity |
| Futures leverage | 1x (no amplification) |

**How sizing works**: If equity is $10,000 and risk is 4%, you're risking $400 per trade. If the stop is 5% below entry, position size = $400 / 0.05 = $8,000. Wider stops = smaller positions, tighter stops = larger positions. This normalizes risk across different volatility levels.

---

## Position Conflicts & Priorities

### The Shared Pool

All 3 modes compete for the same 4 position slots. This prevents over-leveraging.

### Conflict Rule

A coin can only have ONE position across all modes. BTC-PERP-INTX (futures) blocks BTC-USD (spot) and vice versa.

### Daily Check Order (00:15 UTC)

1. **Exits first** — Long exits → Futures long exits → Short exits (free up slots)
2. **Regime filters** — Compute bull/bear status from BTC indicators
3. **Futures long pyramiding** (checked first for lower fees)
4. **Spot long pyramiding** (fallback for spot positions)
5. **Futures long entries** (checked first — claims slots at 0.06% fees)
6. **Spot long entries** (fallback — only for coins without perps, e.g. NEAR)
7. **Short entries** (last — only during confirmed bear market)

### Trailing Stop Checks

Every 30 minutes, the bot fetches current prices and checks all trailing stops, partial TPs, and emergency stops for all 3 modes. This runs independently of the daily signal check.

---

## Regime Filters

### Bull Filter (Gates Long Entries)

- **Rule**: BTC daily close > 200-day SMA
- **Current status**: BEAR (longs blocked)
- **When inactive**: No new long entries. Existing positions still managed (stops, TPs work normally)
- **Impact**: Reduced drawdown from 20% to 15%, improved PF from 1.45 to 1.72

### Bear Filter (Gates Short Entries)

- **Rule**: BTC SMA(50) < SMA(200) AND BTC close < SMA(200) (death cross)
- **Current status**: ACTIVE (shorts enabled)
- **More restrictive** than bull filter — requires both MA cross AND price confirmation
- **Impact**: +15% with death cross vs +6.7% with SMA(200)-only, DD 9% vs 22%

---

## Coins

| Mode | Coins (8 each) |
|------|----------------|
| Spot Long | BTC, ETH, SOL, XRP, SUI, LINK, ADA, **NEAR** |
| Futures Long | BTC, ETH, SOL, XRP, SUI, LINK, ADA, **DOGE** |
| Futures Short | BTC, ETH, SOL, XRP, SUI, LINK, ADA, DOGE |

NEAR trades spot only (no Coinbase perp). DOGE trades futures only (replaced NEAR in perp lists). 7 coins overlap — with futures-first priority, they route through perps at lower fees.

---

## Timing

| Event | Time |
|-------|------|
| Daily signal check | 00:15 UTC (15 min after daily candle close) |
| Trailing stop checks | Every 30 minutes |
| Daily summary (Telegram) | 20:00 UTC (noon PST) |
| Config reload from Supabase | Every daily check + every stop check loop |

---

## Return Projections ($10,000 Starting Capital)

### 4-Year Backtest (2022-2025)

| Metric | Value |
|--------|-------|
| Long side (futures, 4% risk) | +370% |
| Short side (futures, 2% risk) | +15% |
| **Combined** | **+385%** ($10K → $48.5K) |
| CAGR | 48.4% |
| Max drawdown | ~26% |
| Win rate | ~65-79% |
| Profit factor | ~2.5 |

### 10-Year Compound Projection

| Year | Start | Net Gain | End | Monthly |
|------|-------|----------|-----|---------|
| 1 | $10,000 | $4,840 | $14,840 | $403/mo |
| 2 | $14,840 | $6,105 | $20,945 | $509/mo |
| 3 | $20,945 | $8,617 | $29,562 | $718/mo |
| 4 | $29,562 | $12,162 | $41,724 | $1,014/mo |
| 5 | $41,724 | $17,166 | $58,890 | $1,430/mo |
| 6 | $58,890 | $24,228 | $83,118 | $2,019/mo |
| 7 | $83,118 | $34,195 | $117,313 | $2,850/mo |
| 8 | $117,313 | $48,263 | $165,575 | $4,022/mo |
| 9 | $165,575 | $68,118 | $233,694 | $5,677/mo |
| 10 | $233,694 | $96,143 | $329,837 | $8,012/mo |

*Assumes 15% tax on gains over $5K. Bear years will be flat/slightly positive; bull years drive the majority of returns.*

---

## The Intraday Bot (Separate, Experimental)

Runs alongside the Donchian bot in a separate screen session (`screen -S intraday`). Completely independent capital pool.

| Parameter | Value |
|-----------|-------|
| Strategy | MTF Momentum — daily trend + hourly RSI dip + volume |
| ADX gate | Only trades when BTC daily ADX >= 25 (trending) |
| Coins | 8 altcoins (excludes BTC), max 3 positions |
| Risk | 1.5% per trade, 2x leverage |
| Fees | Futures (0.06%/side) |
| Backtest | +69% over 12 months, PF 1.65, $108/day avg |
| Status | Paper trading, not yet in TradeSavvy dashboard |

---

## Backtesting Summary

### Strategy Approval Criteria

| Path | Style | Requirements |
|------|-------|-------------|
| Path A | Mean reversion | 55%+ WR AND 1.5+ PF |
| Path B | Trend following | 1.8+ PF AND avg_win/avg_loss >= 1.5 AND 10+ trades |

### Key Backtest Results

| Test | Trades | WR | PF | Return | MaxDD |
|------|--------|-----|------|--------|-------|
| Baseline (bull filter only) | 66 | 43.9% | 1.72 | +47.9% | 14.9% |
| 4x ATR + pyramid | 90 | 64.4% | 2.73 | +80.9% | 13.8% |
| Walk-forward OOS | 20 | 70.0% | 1.60 | +4.9% | 8.2% |
| Short (death cross) | 16 | 62.5% | 2.73 | +15.0% | 9.0% |
| Combined long+short | — | — | — | +110.4% | 13.2% |

### What Was Tested and Rejected

| Test | Result | Why |
|------|--------|-----|
| Weekly MTF entry filter | Reduced OOS returns | Over-filtered, starved opportunities |
| ADX conviction filter | Negative OOS | Breakouts start when ADX is low |
| 18-coin expansion | -2.7% OOS vs +3.5% | Position slot competition |
| Ranging overlay (hourly) | All 10 variants PF < 1.0 | No viable mean reversion on crypto hourly |
| Multi-market (NQ, ES, forex) | All negative OOS | Crypto momentum is unique |
| 15-minute timeframe | All 216 configs negative | Too noisy |
| Hybrid core/satellite | +92% vs bot +111% | Bull filter protects better than buy-and-hold |

---

## Fee Structure

| Venue | Taker Fee | Used For |
|-------|-----------|----------|
| Coinbase CFM Perps | 0.06%/side | Futures longs + shorts (primary) |
| Coinbase Spot ($1K-$10K tier) | 0.50%/side | Spot longs (fallback, NEAR only) |

The switch from spot to futures for long entries is the single biggest optimization lever: ~8x lower fees roughly doubles net returns over 4 years.

---

## Infrastructure

| Component | Details |
|-----------|---------|
| Language | Python 3.14, venv |
| Hosting | AWS EC2 t3.small, Ubuntu 24.04 |
| Dashboard | TradeSavvy (tradesavvy.io) — React + FastAPI + Supabase |
| Notifications | Telegram Bot API (per-user config) |
| Data | Coinbase Public API (market data), Coinbase CFM API (futures) |
| Source | GitHub: jasondnichol/ResearchAgent + jasondnichol/tradesavvy |

---

## Important Rules

- **Paper trading only** until validation is complete
- **Never switch to live** without explicit owner approval
- **Never commit API keys** to git (use .env + load_dotenv)
- **Always test locally** before deploying to EC2
