# ResearchAgent - Project Context for Claude Code

## Quick Summary

This is an **automated crypto trading bot** that paper-trades 8 coins on daily candles using a Donchian Channel Breakout strategy. Built by Jason Nichol. The system was originally regime-based (hourly), but pivoted to daily breakouts after Coinbase fees destroyed hourly profitability. Runs 24/7 on AWS EC2.

## Current Status (Feb 27, 2026)

- **Production:** Donchian bot running on EC2 (184.72.84.30) in `screen -S donchian`
- **Strategy:** Dual-mode Donchian Breakout (daily) — longs (spot) + shorts (perpetual futures)
- **Phase 3 deployed (Feb 26):** 4x ATR trailing stop + pyramid at +15% on new 20-day high
- **Phase F3 built (Feb 27):** Dual-mode bot — short-side via Coinbase CFM perpetual futures
- **Supabase sync deployed (Feb 27):** Bot syncs trades/positions to TradeSavvy dashboard
- **Bull filter:** Entries only when BTC > SMA(200) AND SMA(50) > SMA(200). Currently BEAR.
- **Bear filter:** Short entries when SMA(50) < SMA(200) AND BTC < SMA(200) (death cross). Currently ACTIVE.
- **Long coins:** BTC, ETH, SOL, XRP, SUI, LINK, ADA, NEAR (8 coins)
- **Short coins:** BTC, ETH, SOL, XRP, SUI, LINK, ADA, DOGE (8 coins, NEAR replaced by DOGE for perps)
- **Backtest (4x ATR + pyramid):** 90 trades, 64.4% WR, 2.73 PF, +80.9% return, 13.8% max DD
- **Short backtest (death cross):** 16 trades, 62.5% WR, 2.73 PF, +15.0%, 9.0% DD
- **Combined long+short:** +110.4% vs +95.3% long-only, DD 13.2% vs 13.8%
- **Walk-forward OOS:** +4.9% return, 1.60 PF, 70% WR, 8.2% DD (vs baseline -3.2%)
- **Portfolio:** $10K paper, max 4 concurrent positions (shared pool), 2% risk per trade
- **Monitoring:** Daily signal check at 00:15 UTC, trailing stops every 30 min
- **Telegram:** All trades + daily summary at 20:00 UTC
- **GitHub:** `jasondnichol/ResearchAgent` (pushed, API keys scrubbed from history)
- **Legacy hourly bot:** Stopped on EC2, code still in repo for reference

## Architecture

```
Coinbase Public API → Daily Candles → Donchian Strategy → Signal → Trade Execution → Telegram
                                                                          ↓
Coinbase CFM API → Perp Futures ──────────────────────── Short Execution ─┘
                                                                          ↓
                                                                   bot_state.json (crash recovery)
```

**Dual-mode:** Longs trade spot (Coinbase), shorts trade perpetual futures (Coinbase CFM, paper mode).

**Dual-loop:**
- **Daily check (00:15 UTC):** Fetch 60 candles per coin, check long exits → short exits → regime filters → long entries → short entries
- **Trailing stop check (every 30 min):** Fetch current prices, update high/low watermarks, check stops/TP for both sides

## Donchian Strategy — Long Side (Spot)

- **Bull filter:** BTC close > SMA(200) AND SMA(50) > SMA(200) — gates long entries
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

## Donchian Strategy — Short Side (Perpetual Futures)

- **Bear filter:** SMA(50) < SMA(200) AND BTC close < SMA(200) (death cross) — gates short entries
- **Entry:** Close < 10-day Donchian low + volume > 2.0x avg + price < EMA(21)
- **Exit:** 2x ATR(14) inverted trailing stop (low watermark + 2x ATR) OR close > 15-day high OR 15% emergency stop OR 30-day max hold
- **Partial TP:** 25% at -10% (price drop), 25% at -20%, 50% runner on trailing stop
- **Position sizing:** 2% risk per trade, sized by short stop distance
- **Pyramiding:** Disabled initially (enable after validation)
- **Exchange:** Coinbase CFM perpetual futures (paper mode), 0.06% taker fee, isolated margin, USDC collateral
- **Backtest (death cross):** 16 trades, 62.5% WR, 2.73 PF, +15.0%, 9.0% DD
- **Walk-forward OOS:** +5.8-12.4%, PF 2.37-2.46 (all positive)
- **Combined long+short:** +110.4% vs +95.3% long-only

## Key Files

| File | Purpose |
|------|---------|
| `donchian_multicoin_bot.py` | Production bot — dual-mode long+short (runs 24/7 on EC2) |
| `donchian_breakout_strategy.py` | Donchian signal generation (long + short signals) |
| `coinbase_futures.py` | Coinbase CFM perpetual futures client (paper + live) |
| `supabase_sync.py` | Supabase sync for TradeSavvy dashboard |
| `test_cfm_connection.py` | Coinbase CFM integration test (12/12 passing) |
| `backtest_shorts.py` | Short-side Donchian backtest with death cross filter |
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
| `.env` | API keys (CLAUDE_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, API_KEY, API_SECRET_FILE) |
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
- **Libraries:** pandas, numpy, requests, python-dotenv, anthropic, coinbase-advanced-py
- **APIs:** Coinbase Public API (spot market data), Coinbase Advanced API (futures, authenticated), Claude API (research only)
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
7. ~~Phase F1: Futures research~~ **DONE** — Coinbase CFM selected, short backtest validated (Feb 27, 2026)
8. ~~Phase F2: Exchange integration~~ **DONE** — `coinbase_futures.py` built, 12/12 tests passing (Feb 27, 2026)
9. ~~Phase F3: Dual-mode bot~~ **DEPLOYED** — Short-side on EC2, Supabase migrated (Feb 28, 2026)
10. **TradeSavvy dual-mode UI** — Strategy/Trades/Performance/BotControl pages need spot vs futures differentiation
11. **Monitor dual-mode** — Observe short signals in bear market, validate P&L tracking
12. Evaluate selective coin swaps (e.g., DOGE for NEAR) after paper trading validation
13. Consider live trading with $1,000-$2,000 after validation

## Important Rules

- **NEVER switch to live trading** without explicit owner approval
- **NEVER commit API keys** to git (use .env + load_dotenv)
- **Always test locally** before deploying to EC2
- **Paper trading only** until 30-day validation complete
- **ALWAYS update memory before context reset** — Before clearing context or ending a session, update MEMORY.md and relevant memory files (strategies.md, deployment.md, architecture.md) with: current project state, what was accomplished, what's in progress, and next steps. This ensures continuity across sessions. Also update CLAUDE.md if priorities or status changed.

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
