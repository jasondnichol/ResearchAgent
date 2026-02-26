# ResearchAgent - Project Context for Claude Code

## Quick Summary

This is an **automated crypto trading bot** that paper-trades 8 coins on daily candles using a Donchian Channel Breakout strategy. Built by Jason Nichol. The system was originally regime-based (hourly), but pivoted to daily breakouts after Coinbase fees destroyed hourly profitability. Runs 24/7 on AWS EC2.

## Current Status (Feb 26, 2026)

- **Production:** Donchian bot running on EC2 (184.72.84.30) in `screen -S donchian`
- **Strategy:** Donchian Channel Breakout (daily candles, regime-agnostic)
- **Coins:** BTC, ETH, SOL, XRP, SUI, LINK, ADA, NEAR (8 coins, dropped HBAR/AVAX)
- **Backtest:** 103 trades, 38.8% WR, 1.45 PF, +46.4% return (with 0.45% fees/side)
- **Portfolio:** $10K paper, max 4 concurrent positions, 2% risk per trade
- **Monitoring:** Daily signal check at 00:15 UTC, trailing stops every 30 min
- **Telegram:** All trades + daily summary at 20:00 UTC
- **GitHub:** `jasondnichol/ResearchAgent` (pushed, API keys scrubbed from history)
- **Legacy hourly bot:** Stopped on EC2, code still in repo for reference

## Architecture

```
Coinbase Public API → Daily Candles → Donchian Strategy → Signal → Trade Execution → Telegram
                                                                          ↓
                                                                   bot_state.json (crash recovery)
```

**Dual-loop:**
- **Daily check (00:15 UTC):** Fetch 60 candles per coin, compute indicators, check exits then entries
- **Trailing stop check (every 30 min):** Fetch current prices, update high watermarks, check stops/TP

## Donchian Strategy

- **Entry:** Close > 20-day Donchian high + volume > 1.5x avg + price > EMA(21)
- **Exit:** 3x ATR(14) trailing stop OR close < 10-day low OR 15% emergency stop
- **Blow-off:** Tighten stop to 1.5x ATR if volume > 3x avg AND RSI > 80
- **Partial TP:** 25% at +10%, 25% at +20%, 50% runner on trailing stop
- **Position sizing:** 2% risk per trade, sized by stop distance
- **Avg win:** +25.8% | **Avg loss:** -10.8% | **Avg hold:** 21 days

## Key Files

| File | Purpose |
|------|---------|
| `donchian_multicoin_bot.py` | Production bot (runs 24/7 on EC2) |
| `donchian_breakout_strategy.py` | Donchian signal generation |
| `backtest_donchian_daily.py` | 4-year multi-coin backtest |
| `market_regime.py` | `RegimeClassifier` (legacy, kept for reference) |
| `integrated_switcher.py` | Old hourly regime-switching bot (deprecated) |
| `williams_r_strategy.py` | Williams %R signal generation (legacy) |
| `adx_momentum_strategy.py` | ADX Momentum signal generation (legacy) |
| `bb_reversion_strategy.py` | BB Mean Reversion signal generation (legacy) |
| `notify.py` | Telegram notifications + file logging |
| `strategy_library.json` | All approved strategies (current + legacy) |
| `research_agent_v2.py` | Claude API strategy research |
| `cache_daily/` | Cached daily candles per coin |
| `.env` | API keys (CLAUDE_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID) |
| `STRATEGY_GUIDE.md` | Complete strategy explanation (plain English) |

## Deployment

- **Local dev:** C:\ResearchAgent (Windows 11, bash shell)
- **Production:** /home/ubuntu/ResearchAgent on EC2
- **Deploy:** SCP files to EC2, kill old screen, restart in `screen -S donchian`
- **EC2 SSH:** `ssh -i "C:\Users\TradingBot\tradingbot-key.pem" ubuntu@184.72.84.30`
- **Bot screen:** `screen -S donchian` (attach: `screen -r donchian`, detach: Ctrl+A D)

### Deploy workflow:
1. Edit/test files locally (`venv/Scripts/python` on Windows)
2. `scp -i "C:\Users\TradingBot\tradingbot-key.pem" <files> ubuntu@184.72.84.30:/home/ubuntu/ResearchAgent/`
3. SSH in, `screen -S donchian -X quit`, then `cd /home/ubuntu/ResearchAgent && screen -dmS donchian venv/bin/python3 donchian_multicoin_bot.py`

## Tech Stack

- **Python 3.14** with venv
- **Libraries:** pandas, numpy, requests, python-dotenv, anthropic
- **APIs:** Coinbase Public API (no auth for market data), Claude API (research only)
- **Notifications:** Telegram Bot API
- **Hosting:** AWS EC2 t3.small, Ubuntu 24.04
- **Source control:** GitHub (jasondnichol/ResearchAgent)

## Why We Pivoted from Hourly to Daily

Hourly trading was structurally unprofitable after Coinbase fees:
- ~1,100 trades with avg gain ~0.05% vs ~0.60% round-trip fee cost
- Fees were 12x larger than average trade gain
- Adding realistic costs turned +57% backtest into -99%
- Daily holds (avg 21 days) make the 0.90% round-trip fee negligible vs +25.8% avg win

## Legacy: Regime-Based Hourly System (Deprecated)

The old system classified BTC into 3 regimes and ran a different strategy for each:
- **RANGING:** Williams %R Mean Reversion (54.9% WR, 2.09 PF)
- **TRENDING:** ADX Momentum Thrust (50% WR, 1.92 PF)
- **VOLATILE:** BB Mean Reversion (72.7% WR, 1.65 PF)
- **Regime classifier:** ADX > 25 = Trending, ATR/close > 3% = Volatile, else Ranging
- Code still in repo (`integrated_switcher.py`, individual strategy files)

## Strategy Approval Thresholds

- Path A (mean-reversion): 55%+ win rate AND 1.5+ profit factor
- Path B (trend-following): 1.8+ profit factor AND avg_win/avg_loss >= 1.5 AND 10+ trades

## Coinbase Fee Tiers

| Volume | Taker | Maker |
|--------|-------|-------|
| < $1K | 1.20% | 0.60% |
| $1K-$10K | 0.75% | 0.35% |
| $10K-$50K | 0.40% | 0.25% |

Backtests use 0.45% per side (conservative for $1K-$10K tier).

## Priorities / What's Next

1. Monitor Donchian paper trading over 30 days
2. Evaluate dropping NEAR/XRP if performance stays negative
3. Consider live trading with $500-$1,000 after validation
4. Integrate into TradeSavvy dashboard

## Important Rules

- **NEVER switch to live trading** without explicit owner approval
- **NEVER commit API keys** to git (use .env + load_dotenv)
- **Always test locally** before deploying to EC2
- **Paper trading only** until 30-day validation complete

## Related Projects

- **TradeSavvy:** SaaS platform at tradesavvy.io (React + FastAPI + Supabase)
- **TradingBot:** Original bot at C:\TradingBot (superseded by this project)

## Owner

Jason Nichol — jasonnichol@gmail.com — Hidden Hills, CA (PST)
