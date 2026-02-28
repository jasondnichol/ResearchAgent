# ResearchAgent Strategy Guide

A complete, plain-English explanation of how this trading system works.

---

## What This System Does

ResearchAgent is an automated crypto trading bot that paper-trades on Coinbase. It runs three modes simultaneously: **Spot Long** (buy breakouts on spot), **Futures Long** (buy breakouts via leveraged perpetual futures), and **Futures Short** (sell breakdowns via perpetual futures). It watches 8 coins on daily candles, waits for breakout/breakdown signals, enters positions with strict risk management, and exits via trailing stops or partial profit-taking. Everything runs 24/7 on an AWS EC2 server with Telegram alerts for every trade.

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

1. **Bull market filter**: BTC must be above its 200-day SMA AND the 50-day SMA must be above the 200-day SMA (golden cross). This is a macro gate -- if BTC isn't in a bull market, no entries are taken on any coin.
2. **Breakout**: Today's close is above the previous 20-day Donchian high
3. **Volume confirmation**: Today's volume is > 1.5x the 20-day average volume
4. **Trend filter**: Price is above the 21-day EMA (we only buy in uptrends)

The bull filter is the most important improvement. Walk-forward testing showed the strategy lost -9.4% in out-of-sample 2025-2026 without it, but only -3.2% with it (and the remaining losses came from the few bull windows that quickly reversed). Over the full 4-year period, the filter improved profit factor from 1.45 to 1.72 and cut max drawdown from 20% to 15% while maintaining returns.

The volume filter is also critical. Without it, returns drop from +46.4% to +18.7%. Volume confirms that the breakout has real institutional participation behind it, not just a random spike.

### Exit Conditions (any one triggers a sell)

1. **Trailing stop (primary exit)**: Price drops below the high watermark minus 4x ATR(14). As price rises, the stop ratchets up but never moves down. The wider 4x multiplier (upgraded from 3x in Phase 3) lets winners run longer while still protecting gains.

2. **Donchian exit channel**: Price closes below the 10-day low. This catches trend reversals that the trailing stop might miss.

3. **Emergency stop**: Price drops 15% below entry. A hard floor to limit catastrophic losses.

### Blow-Off Top Detection

If volume spikes above 3x average AND RSI exceeds 80, the stop tightens from 4x ATR to 1.5x ATR. This captures the bulk of a parabolic move while exiting before the inevitable crash. Blow-off tops in crypto are common and can give back 50%+ of gains in days.

### Partial Profit Taking

Instead of all-or-nothing exits, we scale out in three stages:

| Level | Trigger | Action |
|-------|---------|--------|
| TP1 | +10% gain | Sell 25% of position |
| TP2 | +20% gain | Sell 25% of position |
| Runner | Trailing stop hit | Sell remaining 50% |

This locks in some profit early while keeping exposure to larger moves.

### Pyramiding (Adding to Winners)

When an existing position is up +15% AND making a new 20-day Donchian high, the bot adds a second tranche. This only happens once per position, and only when the bull filter is active.

| Parameter | Value |
|-----------|-------|
| Trigger | Position up +15% AND new 20-day high |
| Add-on risk | 1% of equity |
| Size | Calculated from 4x ATR stop distance, capped at 50% of remaining cash |
| Max adds | 1 per position |

Pyramiding was the single biggest improvement in Phase 3 backtesting. Over the full 4-year period, it boosted returns from +47.9% to +80.9% (with 4x ATR) and improved PF from 1.72 to 2.73. In out-of-sample testing, it turned a -3.2% loss into a +4.9% gain.

### Position Sizing and Risk Management

- **Risk per trade**: 2% of portfolio equity
- **Position size**: Calculated from stop distance. If the stop is 4x ATR below entry, the position is sized so that getting stopped out loses exactly 2% of equity.
- **Max concurrent positions**: 4 out of 8 coins
- **Cash reserve**: Always keeps 5% cash available

### Bull Market Filter (BTC Macro Gate)

The bot checks two conditions on BTC before allowing any new entries across all coins:

1. **BTC close > 200-day SMA** -- BTC is in a long-term uptrend
2. **BTC 50-day SMA > 200-day SMA** -- The golden cross confirms sustained bullish momentum

If either condition is false, the bot sits on the sidelines. No new positions are opened. Existing positions are unaffected -- trailing stops and exits still work normally.

This filter exists because breakout strategies work in bull markets and get chopped up in bear markets. Over our 4-year BTC dataset, about 52% of days qualified as "bull" by this definition. The filter blocked 145 of 248 potential entries, keeping only the 66 highest-conviction trades.

**Impact on backtest results:**

| Metric | Without Filter | With Filter |
|--------|---------------|-------------|
| Trades | 103 | 66 |
| Win Rate | 38.8% | 43.9% |
| Profit Factor | 1.45 | 1.72 |
| Total Return | +46.4% | +47.9% |
| Max Drawdown | 20.1% | 14.9% |
| Sharpe Ratio | 0.31 | 0.37 |

Fewer trades, better win rate, higher profit factor, lower drawdown, and slightly higher returns. The filter removes bad trades without removing good ones.

### Futures Long Mode

The Futures Long mode uses the **exact same entry and exit logic** as Spot Long, but executes trades via Coinbase CFM perpetual futures instead of spot orders. This allows applying leverage (1-3x, configurable via the TradeSavvy dashboard) to amplify returns.

| Feature | Spot Long | Futures Long |
|---------|-----------|-------------|
| Entry signal | Same | Same |
| Exit logic | Same | Same |
| Pyramiding | 1% risk add | 1% risk add * leverage |
| Position sizing | 2% risk | 2% risk * leverage |
| Exchange | Coinbase Spot | Coinbase CFM Perps |
| Leverage | 1x (fixed) | 1-3x (configurable) |
| Bull filter | Required | Required + `futures_long_enabled` |
| Coins | BTC, ETH, SOL, XRP, SUI, LINK, ADA, NEAR | BTC, ETH, SOL, XRP, SUI, LINK, ADA, DOGE |

Futures long is off by default. Users opt in via the TradeSavvy dashboard. Leverage starts at 1x and is hard-capped at 3x in the bot code. The leverage setting is read from Supabase before each daily check.

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

### Tri-Mode Design

The bot operates three trading modes simultaneously:

| Mode | Market | Filter | Direction | Exchange |
|------|--------|--------|-----------|----------|
| **Spot Long** | Bull (golden cross) | BTC > SMA(200), SMA(50) > SMA(200) | Buy breakouts | Coinbase Spot |
| **Futures Long** | Bull (golden cross) | Same as spot + `futures_long_enabled` | Buy breakouts via perps | Coinbase CFM (1-3x leverage) |
| **Futures Short** | Bear (death cross) | SMA(50) < SMA(200), BTC < SMA(200) | Sell breakdowns via perps | Coinbase CFM |

All three modes share a pool of max 4 concurrent positions. A coin can only have one position across all modes.

### Dual-Loop Design

```
00:15 UTC   Daily signal check (fetch 60 candles per coin, compute all indicators, check entries/exits)
00:45 UTC   Trailing stop check (fetch current prices, update high/low watermarks, check stops)
01:15 UTC   Trailing stop check
...         (every 30 minutes)
23:45 UTC   Trailing stop check
00:15 UTC   Next daily signal check
```

The **daily check** flow:
1. Spot long exits → Futures long exits → Short exits
2. Regime filters (bull + bear)
3. Spot long pyramiding → Futures long pyramiding
4. Spot long entries → Futures long entries → Short entries

The **trailing stop check** (every 30 minutes) is lightweight: fetches only the current price from Coinbase's ticker endpoint, updates the high/low watermark if price has moved, and checks if any stop or take-profit level has been hit for all three modes.

### State Persistence

The bot saves its state (positions, cash, P&L) to `bot_state.json` after every trade. If the bot crashes or the server reboots, it reloads this file on startup and continues where it left off. No positions are lost.

### Telegram Notifications

Every trade action sends a formatted message to Telegram:
- **Startup**: strategy info, equity, bull filter status
- **BUY**: coin, price, position size, stop level
- **SELL / PARTIAL SELL**: entry, exit, P&L percentage and dollar amount, hold duration
- **Daily summary**: sent at 20:00 UTC (noon PST) with portfolio value, open positions, and bull filter status

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

The Donchian strategy was validated across 10 coins simultaneously: 103 trades, 38.8% win rate, 1.45 profit factor, +46.4% total return with fees included. With the bull filter: 66 trades, 43.9% win rate, 1.72 profit factor, +47.9% return.

### Walk-Forward Validation

To check for overfitting, we split the data into train (2022-2024) and test (2025-2026) periods. The strategy parameters were frozen from the training period -- no re-optimization on out-of-sample data.

| Period | Trades | WR | PF | Return | MaxDD | Sharpe |
|--------|--------|-----|------|--------|-------|--------|
| Train 2022-2024 (no filter) | 70 | 42.9% | 2.26 | +67.0% | 16.0% | 0.72 |
| Test 2025-2026 (no filter) | 39 | 30.8% | 0.74 | -9.4% | 20.1% | -0.60 |
| Train 2022-2024 (bull filter) | 44 | 50.0% | 2.98 | +63.5% | 12.8% | 0.80 |
| Test 2025-2026 (bull filter) | 16 | 43.8% | 0.78 | -3.2% | 12.5% | -0.60 |

The out-of-sample period (2025-2026) was a challenging bear/sideways market for crypto. The bull filter correctly kept us mostly sidelined (only 16 entries vs 39 without it) and reduced losses from -9.4% to -3.2%. The strategy isn't overfit to noise -- it's genuinely a bull market strategy, and the filter ensures it only trades in its ideal conditions.

### Slippage Stress Test

We tested the strategy with increasing slippage to find the breaking point:

| Slippage | Total Cost/Side | Return | PF |
|----------|----------------|--------|-----|
| 0.00% | 0.40% | +47.7% | 1.46 |
| 0.10% | 0.50% | +44.5% | 1.43 |
| 0.20% | 0.60% | +40.4% | 1.39 |
| 0.30% | 0.70% | +36.7% | 1.35 |
| 0.50% | 0.90% | +29.5% | 1.28 |

The strategy remains profitable at all tested slippage levels, even at 0.90% total cost per side. With realistic 0.20% slippage, returns are +40.4%. Transaction costs are not a risk for this strategy.

### Phase 3: Pyramiding & Exit Tuning

Tested 7 variants combining pyramiding (add to winners at +15%) and exit tuning (4x ATR trailing stop) on top of the bull filter baseline.

**Full Period (2022-2026):**

| Variant | Trades | WR | PF | Return | MaxDD |
|---------|--------|-----|------|--------|-------|
| Baseline (bull filter) | 66 | 43.9% | 1.72 | +47.9% | 14.9% |
| 4x ATR trailing | 60 | 46.7% | 1.95 | +36.4% | 11.8% |
| 4x ATR + pyramid | 90 | 64.4% | 2.73 | +80.9% | 13.8% |

**Out-of-Sample (2025-2026):**

| Variant | Trades | WR | PF | Return | MaxDD |
|---------|--------|-----|------|--------|-------|
| Baseline (bull filter) | 16 | 43.8% | 0.78 | -3.2% | 12.5% |
| 4x ATR trailing | 13 | 53.8% | 1.15 | +1.2% | 6.4% |
| **4x ATR + pyramid** | **20** | **70.0%** | **1.60** | **+4.9%** | **8.2%** |

The 4x ATR + pyramid variant was selected for production. It flipped the OOS from -3.2% loss to +4.9% gain while cutting drawdown from 12.5% to 8.2%.

### Entry Filter Tests (Weekly MTF + ADX Conviction)

Tested adding weekly multi-timeframe confirmation (weekly close > 21-week EMA) and ADX conviction filters (ADX > 22 or > 25) on top of the Phase 3 winner. **Neither improved OOS performance.**

| Variant | Full Period | OOS Return | OOS PF |
|---------|------------|------------|--------|
| Baseline (4x ATR + pyramid) | +80.9% | +4.9% | 1.60 |
| + Weekly MTF | +55.7% | +3.8% | 1.47 |
| + ADX > 22 | +48.5% | -2.4% | 0.74 |
| + Weekly MTF + ADX > 22 | +35.6% | -2.4% | 0.74 |

The existing entry filters (volume confirmation + EMA trend + bull filter) already capture signal quality effectively. ADX hurt performance because breakouts often start when ADX is low (the trend is just beginning). Adding more filters on a strategy with 66-90 trades just starved it of opportunities.

### Coin Expansion Test (Phase 4)

Screened 16 candidate coins on Coinbase. Several showed strong individual results (SEI +231%, FET +145%, SHIB +141%, DOGE +103%), but expanding the portfolio from 8 to 18 coins actually **hurt performance**:

| Universe | Full Period | OOS Return | OOS PF | OOS MaxDD |
|----------|------------|------------|--------|-----------|
| Current 8 coins | +82.1% | +3.5% | 1.38 | 9.2% |
| Expanded 18 coins | +77.3% | -2.7% | 0.76 | 9.1% |

With max 4 concurrent positions, more coins create competition for position slots rather than more opportunities. New coins sometimes displaced better-performing incumbents. Every OOS trade from a new coin was a loser. **The current 8-coin universe was confirmed as optimal.**

Coins tested but not added: DOGE, DOT, LTC, UNI, ATOM, AAVE, FIL, SHIB, FET, OP, INJ, APT, ARB, SEI, TIA, RENDER. Coins with too little Coinbase history to test: BNB, TON, HYPE (all listed in late 2025/2026).

### Key Backtest Files

| File | Purpose |
|------|---------|
| `backtest_donchian_daily.py` | 4-year multi-coin Donchian backtest |
| `backtest_walkforward.py` | Walk-forward validation + slippage stress test |
| `backtest_bull_filter.py` | Bull filter backtest + walk-forward revalidation |
| `backtest_phase3.py` | Phase 3 pyramiding + exit tuning variants |
| `backtest_filters.py` | Entry filter tests (weekly MTF + ADX conviction) |
| `backtest_coin_expansion.py` | Phase 4 coin expansion screening |
| `regime_backtester.py` | Regime-specific backtesting (hourly, legacy) |
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
| `donchian_multicoin_bot.py` | Production bot — tri-mode (runs 24/7 on EC2) |
| `donchian_breakout_strategy.py` | Donchian signal generation (long + short) |
| `coinbase_futures.py` | Coinbase CFM perpetual futures client |
| `supabase_sync.py` | Supabase sync for TradeSavvy dashboard |
| `notify.py` | Telegram notifications + file logging |
| `market_regime.py` | Regime classifier (legacy, kept for reference) |
| `strategy_library.json` | All approved strategies (current + legacy) |
| `integrated_switcher.py` | Old hourly regime-switching bot (deprecated) |
| `.env` | API keys (never committed to git) |

---

## What's Next

1. Monitor tri-mode paper trading (Spot Long + Futures Long + Futures Short)
2. ~~Phase 3: Pyramiding + exit tuning~~ **DONE** — 4x ATR + pyramid deployed (Feb 26, 2026)
3. ~~Phase 4: Coin expansion~~ **TESTED** — 16 candidates screened, current 8 coins confirmed optimal
4. ~~Entry filters (weekly MTF, ADX)~~ **TESTED** — Neither improved OOS performance
5. ~~Futures integration (F1-F4)~~ **DONE** — Tri-mode bot deployed with TradeSavvy UI (Feb 28, 2026)
6. Evaluate selective coin swaps (e.g., DOGE for NEAR) after paper trading validation
7. Consider live trading with $1,000-$2,000 after validation

---

## Rules

- **Paper trading only** until 30-day validation is complete
- **Never switch to live** without explicit owner approval
- **Never commit API keys** to git
- **Always test locally** before deploying to EC2
