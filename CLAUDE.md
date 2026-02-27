# ResearchAgent - Project Context for Claude Code

## Quick Summary

This is an **automated crypto trading bot** that paper-trades 8 coins on daily candles using a Donchian Channel Breakout strategy. Built by Jason Nichol. The system was originally regime-based (hourly), but pivoted to daily breakouts after Coinbase fees destroyed hourly profitability. Runs 24/7 on AWS EC2.

## Current Status (Feb 27, 2026)

- **Production:** Donchian bot running on EC2 (184.72.84.30) in `screen -S donchian`
- **Strategy:** Donchian Channel Breakout (daily) + bull filter + 4x ATR trailing + pyramiding
- **Phase 3 deployed (Feb 26):** 4x ATR trailing stop + pyramid at +15% on new 20-day high
- **Supabase sync deployed (Feb 27):** Bot syncs trades/positions to TradeSavvy dashboard
- **Bull filter:** Entries only when BTC > SMA(200) AND SMA(50) > SMA(200). Currently BEAR.
- **Coins:** BTC, ETH, SOL, XRP, SUI, LINK, ADA, NEAR (8 coins, dropped HBAR/AVAX)
- **Backtest (4x ATR + pyramid):** 90 trades, 64.4% WR, 2.73 PF, +80.9% return, 13.8% max DD
- **Walk-forward OOS:** +4.9% return, 1.60 PF, 70% WR, 8.2% DD (vs baseline -3.2%)
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

- **Bull filter:** BTC close > SMA(200) AND SMA(50) > SMA(200) — gates all entries
- **Entry:** Close > 20-day Donchian high + volume > 1.5x avg + price > EMA(21)
- **Exit:** 4x ATR(14) trailing stop OR close < 10-day low OR 15% emergency stop
- **Blow-off:** Tighten stop to 1.5x ATR if volume > 3x avg AND RSI > 80
- **Pyramiding:** Add 1% equity risk tranche at +15% gain on new 20-day high (max 1 add per position)
- **Partial TP:** 25% at +10%, 25% at +20%, 50% runner on trailing stop
- **Position sizing:** 2% risk per trade, sized by stop distance

### Phase 3 Impact (4x ATR + pyramid, 4-year backtest)
- Baseline (bull filter only): 66 trades, 43.9% WR, 1.72 PF, +47.9%, 14.9% DD
- **4x ATR + pyramid: 90 trades, 64.4% WR, 2.73 PF, +80.9%, 13.8% DD**
- Walk-forward OOS (2025-2026): +4.9% with Phase 3 vs -3.2% baseline

## Key Files

| File | Purpose |
|------|---------|
| `donchian_multicoin_bot.py` | Production bot (runs 24/7 on EC2) |
| `donchian_breakout_strategy.py` | Donchian signal generation |
| `backtest_donchian_daily.py` | 4-year multi-coin backtest |
| `backtest_walkforward.py` | Walk-forward validation + slippage stress test |
| `backtest_bull_filter.py` | Bull filter backtest + walk-forward revalidation |
| `backtest_phase3.py` | Phase 3 pyramiding + exit tuning variants |
| `backtest_filters.py` | Batch 1: weekly MTF + ADX conviction filter tests |
| `backtest_coin_expansion.py` | Phase 4: coin expansion screening + portfolio test |
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

1. Monitor Donchian paper trading over 60-90 days (through correction and any rebound)
2. ~~Phase 3: Pyramiding + exit tuning~~ **DONE** — 4x ATR + pyramid deployed (Feb 26, 2026)
3. ~~Phase 4: Expand coin universe~~ **TESTED** — 16 candidates screened, expansion hurt OOS due to position slot competition. Current 8 coins confirmed optimal for max 4 positions.
4. ~~Entry filters (weekly MTF, ADX)~~ **TESTED** — Neither improved OOS. Current signal quality is already high.
5. ~~Integrate into TradeSavvy dashboard~~ **DONE** — Full SaaS platform deployed (Feb 26, 2026)
6. ~~Strategy page configurable~~ **DONE** — Exit rules, position sizing, pyramiding all configurable with validation + per-tab Reset Defaults (Feb 27, 2026)
7. Evaluate selective coin swaps (e.g., DOGE for NEAR) after paper trading validation
8. Consider live trading with $1,000-$2,000 after validation

## Important Rules

- **NEVER switch to live trading** without explicit owner approval
- **NEVER commit API keys** to git (use .env + load_dotenv)
- **Always test locally** before deploying to EC2
- **Paper trading only** until 30-day validation complete

## Related Projects

- **TradeSavvy:** SaaS platform at tradesavvy.io (`jasondnichol/tradesavvy`)
  - React + FastAPI + Supabase, hosted on Vercel + Railway
  - Pages: Dashboard, Trades, Performance, Portfolio, Watchlist, Strategy, Backtest, Bot Control, Settings
  - Demo account: demo@tradesavvy.io / demo123 (settings locked, daily reset via GitHub Actions)
  - Portfolio page with Coinbase + CoinGecko live prices
  - Strategy page: fully configurable exit rules, position sizing, pyramiding with server-side validation + per-tab Reset Defaults
  - Settings: notification toggles, email/password update, Coinbase/Telegram setup guides
  - Local dev: `C:\tradesavvy` (frontend :3000, backend :8000)
- **TradingBot:** Original bot at C:\TradingBot (superseded by this project)

## Owner

Jason Nichol — jasonnichol@gmail.com — Hidden Hills, CA (PST)
