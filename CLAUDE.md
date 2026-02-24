# ResearchAgent - Project Context for Claude Code

## Quick Summary

This is an **AI-powered crypto trading bot** that uses regime detection to automatically switch between trading strategies. Built by Jason Nichol. The system researches strategies via Claude API, backtests them on 4 years of BTC data, and deploys approved strategies for automated paper trading.

## Current Status (Feb 24, 2026)

- **Production:** Running 24/7 on AWS EC2 (184.72.84.30) in paper trading mode
- **RANGING strategy:** Williams %R Mean Reversion (54.9% win rate, 2.09 PF) — Path A approved
- **TRENDING strategy:** ADX Momentum Thrust (50% win rate, 1.92 PF) — Path B approved
- **VOLATILE strategy:** Bollinger Band Mean Reversion (72.7% win rate, 1.65 PF) — Path A approved
- **Market Regime:** TRENDING (ADX Momentum is active, holding — bearish DI)
- **Regime classifier:** Unified `RegimeClassifier` in market_regime.py (Wilder's EWM ADX, 3 regimes)
- **All 3 regimes covered** — bot has a strategy for every market condition
- **Force-sell:** Bot auto-exits positions after 3 hours if regime changes and no strategy matches
- **Telegram notifications:** Integrated and working (BUY/SELL/force-sell alerts + daily summary)
- **Logging:** File logging to logs/trading.log and logs/trades.log

## Architecture

```
Claude API → Research Agent → Backtest → Strategy Library → Strategy Switcher → Coinbase Public API
```

- **RegimeClassifier** (market_regime.py) — single source of truth for regime labels using Wilder's EWM ADX
- **Strategy Switcher** (integrated_switcher.py) — dual-loop architecture:
  - **Hourly full check:** Fetches 100 candles, detects regime, computes indicators, generates signals
  - **5-min lightweight check:** Fetches current price from Coinbase ticker, monitors stop loss & take-profit
- **Williams %R** approved for RANGING markets (Path A: high win rate)
- **ADX Momentum Thrust** approved for TRENDING markets (Path B: high profit factor, trailing stop)
- **BB Mean Reversion** approved for VOLATILE markets (Path A: high win rate, enters at lower BB)

## Key Files

| File | Purpose |
|------|---------|
| `integrated_switcher.py` | Main trading bot (runs 24/7 on EC2) ⭐ |
| `market_regime.py` | `RegimeClassifier` (canonical) + `MarketRegimeDetector` |
| `williams_r_strategy.py` | Williams %R signal generation (RANGING) |
| `adx_momentum_strategy.py` | ADX Momentum signal generation (TRENDING) |
| `bb_reversion_strategy.py` | BB Mean Reversion signal generation (VOLATILE) |
| `notify.py` | Telegram notifications + file logging |
| `daily_summary.py` | Sends daily trade summary via Telegram |
| `strategy_library.py` | Strategy library management |
| `strategy_library.json` | Approved strategies database |
| `regime_backtester.py` | Regime-specific backtesting (90-day hourly) |
| `backtest_adx_trending.py` | ADX Momentum 4-year daily backtest |
| `backtest_bb_volatile.py` | BB Mean Reversion 4-year daily backtest |
| `cycle_backtester_cached.py` | 4-year BTC data cache + regime classification |
| `research_agent_v2.py` | Claude API strategy research |
| `btc_4year_cache.json` | Cached 4-year historical data |
| `.env` | API keys (CLAUDE_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID) |

## Deployment

- **Local dev:** C:\ResearchAgent (Windows, PowerShell)
- **Production:** /home/ubuntu/ResearchAgent on EC2
- **Deploy method:** SCP files to EC2, restart bot in screen session
- **EC2 SSH:** `ssh -i "C:\Users\TradingBot\tradingbot-key.pem" ubuntu@184.72.84.30`
- **Bot runs in:** `screen -S tradingbot`

### Deploy workflow:
1. Edit files locally
2. SCP changed files to EC2
3. SSH in, kill old bot, restart: `screen -dmS tradingbot venv/bin/python3 integrated_switcher.py`

## Tech Stack

- **Python 3.14** with venv
- **Libraries:** anthropic, coinbase-advanced-py, pandas, numpy, ta, requests, python-dotenv
- **APIs:** Coinbase Public API (no auth for market data), Claude API (for research)
- **Notifications:** Telegram Bot API
- **Hosting:** AWS EC2 t3.small, Ubuntu 24.04

## Configuration

- API keys stored in `.env` (loaded via python-dotenv)
- Strategy approval thresholds (two paths):
  - Path A (mean-reversion): 55%+ win rate AND 1.5+ profit factor
  - Path B (trend-following): 1.8+ profit factor AND avg_win/avg_loss >= 1.5 AND 10+ trades
- Paper trading mode ON (no real money)
- Bot checks regime every 1 hour (full signal generation)
- Price monitoring every 5 minutes (stop loss + take-profit)
- Emergency stop: 5% below entry price (all strategies)

## Trading Strategy Details

### Williams %R Mean Reversion (APPROVED - RANGING markets, Path A)
- Williams %R(14) crosses above -80 + price below SMA(21) → BUY
- Williams %R crosses below -20 OR price ≥ SMA + 1.5% → SELL
- **5-min monitoring:** Stop = emergency 5% only | Take-profit = SMA(21) * 1.015
- Timeframe: 1H candles
- Backtested: 51 trades, 54.9% win rate, 2.09 profit factor, +7.15% total P&L

### ADX Momentum Thrust (APPROVED - TRENDING markets, Path B)
- Entry: ADX > 20 + ADX rising (5-bar) + +DI > -DI + DI crossover (15 bars) + RSI 35-78 + price > SMA(50)
- Exit: ADX < 20 OR RSI > 75 OR trailing stop (1.5x ATR from high watermark) OR bearish DI reversal
- **5-min monitoring:** Stop = trailing (high_watermark - 1.5x ATR, updated every 5 min) | No fixed take-profit
- Timeframe: 1H candles
- Backtested: 10 trades, 50% win rate, 1.92 profit factor, +12.17% total P&L
- Avg win: +5.07% | Avg loss: -2.64% | Trailing stop locks in gains

### Bollinger Band Mean Reversion (APPROVED - VOLATILE markets, Path A)
- Entry: Close below lower BB (SMA(20) - 2σ) + RSI(14) < 35 (oversold)
- Exit: Price reaches SMA(20) middle band OR RSI > 70 OR 1.5x ATR stop loss
- **5-min monitoring:** Stop = entry - 1.5x ATR | Take-profit = SMA(20) middle band
- Regime exit: Force-close after 3 consecutive non-VOLATILE bars
- Timeframe: 1H candles
- Backtested: 11 trades, 72.7% win rate, 1.65 profit factor, +21.10% total P&L
- Avg win: +6.69% | Avg loss: -10.80%

### 5-Minute Price Monitoring (Between Hourly Signals)

The bot uses a dual-loop architecture to avoid being blind between hourly checks:

```
:00  Full signal check (100 candles, all indicators, regime detection)
:05  Lightweight price check (Coinbase ticker → stop/TP check)
:10  Lightweight price check
...
:55  Lightweight price check
:00  Full signal check (next hour)
```

- **Lightweight check** uses `Coinbase /products/BTC-USD/ticker` — just current price, no candle history
- Stop and take-profit levels are computed during the hourly full check, then stored on the switcher
- Every 5 min: check price against stored levels, trigger SELL if hit
- **Emergency stop** (all strategies): auto-sell if price drops > 5% below entry
- ADX trailing stop updates high watermark every 5 min (stop ratchets up as price rises)

### Market Regime Detection (Unified RegimeClassifier)
- Uses Wilder's EWM (alpha=1/period) for ATR, DI, and ADX smoothing
- Calculates ADX(14), ATR(14), SMA(20), SMA(50)
- TRENDING: ADX > 25 AND clear direction (UPTREND or DOWNTREND)
- VOLATILE: Volatility (ATR/close) > 3%
- RANGING: Everything else

### 4-Year Historical Data
- Period: 2022-02-23 to 2026-02-21 (1460 days)
- TRENDING: 37.1% | VOLATILE: 44.6% | RANGING: 18.3%
- Cached in btc_4year_cache.json

## Related Projects

- **TradeSavvy:** SaaS platform at tradesavvy.io (React + FastAPI + Supabase)
- **TradingBot:** Original bot at C:\TradingBot (superseded by this project)

## Priorities / What's Next

1. Monitor paper trading performance over 30 days (all 3 regimes now covered)
2. Add more trading pairs (ETH, etc.)
3. Consider live trading with small capital ($500-1000) after validation
4. Integrate findings into TradeSavvy dashboard

## Important Rules

- **NEVER switch to live trading** without explicit owner approval
- **NEVER commit API keys** to git
- **Always test locally** before deploying to EC2
- **Paper trading only** until 30-day validation complete
- Strategy must pass backtesting via two-path approval before deployment:
  - Path A (mean-reversion): 55%+ win rate AND 1.5+ PF
  - Path B (trend-following): 1.8+ PF AND avg_win/avg_loss >= 1.5 AND 10+ trades

## Owner

Jason Nichol — jasonnichol@gmail.com — Hidden Hills, CA (PST)
