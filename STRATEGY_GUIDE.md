# ResearchAgent Strategy Guide

A complete, plain-English explanation of how this trading system works.

---

## What This System Does

ResearchAgent is an automated crypto trading bot that paper-trades on Coinbase. It watches 8 coins on daily candles, waits for breakout signals, enters positions with strict risk management, and exits via trailing stops or partial profit-taking. Everything runs 24/7 on an AWS EC2 server with Telegram alerts for every trade.

---

## The Evolution: Why We Switched Strategies

### Phase 1: Hourly Regime-Based Trading (Deprecated)

The original system classified the market into three regimes (Trending, Volatile, Ranging) and ran a different strategy for each:

| Regime | Strategy | Logic |
|--------|----------|-------|
| Ranging | Williams %R Mean Reversion | Buy oversold bounces in sideways markets |
| Trending | ADX Momentum Thrust | Ride momentum when trend is strong |
| Volatile | Bollinger Band Mean Reversion | Buy at lower band when volatility spikes |

This worked in backtests but **failed in practice** because of Coinbase fees. Hourly trades averaged only +0.05% gain, but each round-trip cost ~0.60% in fees. The fees were 12x larger than the average trade. A strategy that showed +57% in backtest turned into -99% with realistic fees.

### Phase 2: Daily Donchian Breakout (Current)

We pivoted to a daily-timeframe breakout strategy that holds for weeks, not hours. With average gains of +25.8% per winning trade, the 0.45% fee per side is negligible. This strategy is **regime-agnostic** -- it doesn't need to classify the market. It simply waits for confirmed breakouts with volume.

---

## Current Strategy: Donchian Channel Breakout

### The Core Idea

The Donchian Channel tracks the highest high and lowest low over a lookback period. When price breaks above the 20-day high on strong volume, it signals institutional buying and the start of a potential trend. We ride that trend with a trailing stop that locks in gains as price moves up.

### Entry Conditions (ALL must be true)

1. **Breakout**: Today's close is above the previous 20-day Donchian high
2. **Volume confirmation**: Today's volume is > 1.5x the 20-day average volume
3. **Trend filter**: Price is above the 21-day EMA (we only buy in uptrends)

The volume filter is critical. Without it, returns drop from +46.4% to +18.7%. Volume confirms that the breakout has real institutional participation behind it, not just a random spike.

### Exit Conditions (any one triggers a sell)

1. **Trailing stop (primary exit)**: Price drops below the high watermark minus 3x ATR(14). As price rises, the stop ratchets up but never moves down. This lets winners run while protecting gains.

2. **Donchian exit channel**: Price closes below the 10-day low. This catches trend reversals that the trailing stop might miss.

3. **Emergency stop**: Price drops 15% below entry. A hard floor to limit catastrophic losses.

### Blow-Off Top Detection

If volume spikes above 3x average AND RSI exceeds 80, the stop tightens from 3x ATR to 1.5x ATR. This captures the bulk of a parabolic move while exiting before the inevitable crash. Blow-off tops in crypto are common and can give back 50%+ of gains in days.

### Partial Profit Taking

Instead of all-or-nothing exits, we scale out in three stages:

| Level | Trigger | Action |
|-------|---------|--------|
| TP1 | +10% gain | Sell 25% of position |
| TP2 | +20% gain | Sell 25% of position |
| Runner | Trailing stop hit | Sell remaining 50% |

This locks in some profit early while keeping exposure to larger moves.

### Position Sizing and Risk Management

- **Risk per trade**: 2% of portfolio equity
- **Position size**: Calculated from stop distance. If the stop is 3x ATR below entry, the position is sized so that getting stopped out loses exactly 2% of equity.
- **Max concurrent positions**: 4 out of 8 coins
- **Cash reserve**: Always keeps 5% cash available

### Coins Traded

| Coin | 4-Year Backtest | Status |
|------|----------------|--------|
| SOL | +190%, 50% WR | Keep |
| SUI | +173%, 50% WR | Keep |
| BTC | +78%, 44% WR | Keep |
| ETH | +38%, 53% WR | Keep |
| LINK | +30%, 36% WR | Keep |
| ADA | +14%, 43% WR | Keep |
| XRP | -7%, 50% WR | Keep |
| NEAR | -15%, 22% WR | Keep |
| AVAX | -67%, 11% WR | Dropped |
| HBAR | -87%, 0% WR | Dropped |

HBAR and AVAX were removed due to consistently poor performance across the full 4-year period.

---

## Bot Architecture

### Dual-Loop Design

```
00:15 UTC   Daily signal check (fetch 60 candles per coin, compute all indicators, check entries/exits)
00:45 UTC   Trailing stop check (fetch current prices, update high watermarks, check stops)
01:15 UTC   Trailing stop check
...         (every 30 minutes)
23:45 UTC   Trailing stop check
00:15 UTC   Next daily signal check
```

The **daily check** (once per day at 00:15 UTC, after the daily candle closes) runs the full strategy: fetches candle history, calculates indicators, checks exit conditions for open positions, then scans for new entry signals.

The **trailing stop check** (every 30 minutes) is lightweight: fetches only the current price from Coinbase's ticker endpoint, updates the high watermark if price has risen, and checks if any stop or take-profit level has been hit.

### State Persistence

The bot saves its state (positions, cash, P&L) to `bot_state.json` after every trade. If the bot crashes or the server reboots, it reloads this file on startup and continues where it left off. No positions are lost.

### Telegram Notifications

Every trade action sends a formatted message to Telegram:
- **BUY**: coin, price, position size, stop level
- **SELL / PARTIAL SELL**: entry, exit, P&L percentage and dollar amount, hold duration
- **Daily summary**: sent at 20:00 UTC (noon PST) with portfolio value and open positions

---

## Market Regime Detection (Legacy System)

The regime classifier still exists in `market_regime.py` but is not used by the Donchian bot. It's kept for reference and potential future use.

### How It Works

Uses Wilder's Exponential Moving Average (EWM with alpha=1/period) to compute ADX(14), ATR(14), SMA(20), and SMA(50) on price data.

### Classification Rules

| Regime | Condition |
|--------|-----------|
| **Trending** | ADX > 25 AND price shows clear direction (close > SMA20 > SMA50 for uptrend, or close < SMA20 < SMA50 for downtrend) |
| **Volatile** | ATR/close > 3% (daily) or > 0.5% (hourly) |
| **Ranging** | Everything else |

### 4-Year BTC Distribution

- Trending: 37.1% of days
- Volatile: 44.6% of days
- Ranging: 18.3% of days

---

## Backtesting

### How We Validate Strategies

Every strategy must pass a 4-year backtest on historical data (Feb 2022 -- Feb 2026) before being approved for paper trading. The backtest includes realistic transaction costs (0.45% per side for Coinbase).

### Approval Criteria (Two Paths)

| Path | Target Style | Requirements |
|------|-------------|--------------|
| **Path A** | Mean reversion | 55%+ win rate AND 1.5+ profit factor |
| **Path B** | Trend following | 1.8+ profit factor AND avg win/avg loss >= 1.5 AND 10+ trades |

The Donchian strategy was validated across 10 coins simultaneously: 103 trades, 38.8% win rate, 1.45 profit factor, +46.4% total return with fees included. It qualifies as a trend-following strategy with a 2.38 avg-win/avg-loss ratio.

### Key Backtest Files

| File | Purpose |
|------|---------|
| `backtest_donchian_daily.py` | 4-year multi-coin Donchian backtest |
| `regime_backtester.py` | Regime-specific backtesting (hourly, legacy) |
| `backtest_adx_trending.py` | ADX Momentum backtest (legacy) |
| `backtest_bb_volatile.py` | BB Mean Reversion backtest (legacy) |
| `cache_daily/` | Cached daily candles per coin |

---

## Coinbase Fee Structure

Fees were the reason we abandoned hourly trading. Here's the full picture:

| Monthly Volume | Taker Fee | Maker Fee |
|---------------|-----------|-----------|
| < $1K | 1.20% | 0.60% |
| $1K -- $10K | 0.75% | 0.35% |
| $10K -- $50K | 0.40% | 0.25% |

We use 0.45% per side in backtests (conservative estimate for $1K-$10K tier). A round-trip trade costs ~0.90%, which is negligible on trades averaging +25.8% gain but devastating on trades averaging +0.05% gain.

---

## Infrastructure

### Tech Stack
- **Language**: Python 3.14
- **Libraries**: pandas, numpy, requests, python-dotenv, anthropic
- **Data**: Coinbase Public API (no authentication needed for market data)
- **AI Research**: Claude API (for strategy research, not used in trading)
- **Notifications**: Telegram Bot API
- **Hosting**: AWS EC2 t3.small, Ubuntu 24.04

### Deployment
- **Local dev**: `C:\ResearchAgent` on Windows 11
- **Production**: `/home/ubuntu/ResearchAgent` on EC2
- **Process**: SCP files to EC2, restart bot in `screen -S donchian`
- **Source control**: GitHub (`jasondnichol/ResearchAgent`)

### Key Files

| File | Purpose |
|------|---------|
| `donchian_multicoin_bot.py` | Production bot (runs 24/7 on EC2) |
| `donchian_breakout_strategy.py` | Donchian signal generation |
| `market_regime.py` | Regime classifier (legacy, kept for reference) |
| `notify.py` | Telegram notifications + file logging |
| `strategy_library.json` | All approved strategies (current + legacy) |
| `integrated_switcher.py` | Old hourly regime-switching bot (deprecated) |
| `.env` | API keys (never committed to git) |

---

## What's Next

1. Monitor Donchian paper trading over 30 days
2. Evaluate dropping NEAR and XRP if performance stays negative
3. Consider live trading with small capital ($500--$1,000) after validation
4. Integrate into TradeSavvy dashboard

---

## Rules

- **Paper trading only** until 30-day validation is complete
- **Never switch to live** without explicit owner approval
- **Never commit API keys** to git
- **Always test locally** before deploying to EC2
