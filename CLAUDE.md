# ResearchAgent - Project Context for Claude Code

## Quick Summary

This is an **AI-powered crypto trading bot** that uses regime detection to automatically switch between trading strategies. Built by Jason Nichol. The system researches strategies via Claude API, backtests them on 4 years of BTC data, and deploys approved strategies for automated paper trading.

## Current Status (Feb 23, 2026)

- **Production:** Running 24/7 on AWS EC2 (184.72.84.30) in paper trading mode
- **Strategy:** Williams %R Mean Reversion (54.9% win rate, 2.09 profit factor)
- **Market Regime:** RANGING (Williams %R is active)
- **First trade completed:** BUY $68,072 → SELL $68,091 (+0.03%)
- **Telegram notifications:** Integrated and working (BUY/SELL alerts + daily summary)
- **Logging:** File logging to logs/trading.log and logs/trades.log

## Architecture

```
Claude API → Research Agent → Backtest → Strategy Library → Strategy Switcher → Coinbase Public API
```

- **Market Regime Detector** classifies BTC as RANGING/TRENDING/VOLATILE
- **Strategy Switcher** (integrated_switcher.py) runs every 1 hour, picks strategy for current regime
- **Williams %R** is the only approved strategy (for RANGING markets)
- **No approved TRENDING or VOLATILE strategies yet** (EMA Crossover and Breakout both failed backtesting)

## Key Files

| File | Purpose |
|------|---------|
| `integrated_switcher.py` | Main trading bot (runs 24/7 on EC2) ⭐ |
| `notify.py` | Telegram notifications + file logging |
| `daily_summary.py` | Sends daily trade summary via Telegram |
| `williams_r_strategy.py` | Williams %R signal generation |
| `market_regime.py` | Regime detection (RANGING/TRENDING/VOLATILE) |
| `research_agent_v2.py` | Claude API strategy research |
| `strategy_library.py` | Strategy library management |
| `strategy_library.json` | Approved strategies database |
| `regime_backtester.py` | Regime-specific backtesting |
| `cycle_backtester_cached.py` | 4-year BTC data cache |
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
3. SSH in, stop bot (Ctrl+C in screen), restart `python integrated_switcher.py`

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
- Bot checks regime every 1 hour

## Trading Strategy Details

### Williams %R Mean Reversion (APPROVED - RANGING markets)
- Williams %R(14) crosses above -80 + price below SMA(21) → BUY
- Williams %R crosses below -20 OR price ≥ SMA + 1.5% → SELL
- Timeframe: 1H candles
- Backtested: 51 trades, 54.9% win rate, 2.09 profit factor, +7.15% total P&L

### Market Regime Detection
- Uses 30 days of price data
- Calculates ADX, ATR, SMA 20/50/200
- RANGING: ADX < 25, low volatility
- TRENDING: ADX > 25, clear direction
- VOLATILE: High ATR, rapid price swings

### 4-Year Historical Data
- Period: 2022-02-23 to 2026-02-21 (1460 days)
- TRENDING: 48.7% | VOLATILE: 31.5% | RANGING: 19.8%
- Cached in btc_4year_cache.json

## Related Projects

- **TradeSavvy:** SaaS platform at tradesavvy.io (React + FastAPI + Supabase)
- **TradingBot:** Original bot at C:\TradingBot (superseded by this project)

## Priorities / What's Next

1. Monitor paper trading performance over 30 days
2. Research and approve TRENDING market strategies
3. Research VOLATILE market strategies
4. Add more trading pairs (ETH, etc.)
5. Consider live trading with small capital ($500-1000) after validation
6. Integrate findings into TradeSavvy dashboard

## Important Rules

- **NEVER switch to live trading** without explicit owner approval
- **NEVER commit API keys** to git
- **Always test locally** before deploying to EC2
- **Paper trading only** until 30-day validation complete
- Strategy must pass backtesting (55%+ win rate, 1.5+ PF) before deployment

## Owner

Jason Nichol — jasonnichol@gmail.com — Hidden Hills, CA (PST)
