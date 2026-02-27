# F1 Research: Institutional Futures Trading Strategies

**Date:** February 27, 2026
**Purpose:** Deep research into how top institutions (CTAs, hedge funds) trade futures — focusing on trend-following short strategies applicable to our Donchian system. Updated with US regulatory constraints and exchange analysis.

---

## 0. Derivatives, Futures, and Perpetuals — Plain English

### What's a Derivative?
A derivative is a contract whose value is "derived" from something else (like Bitcoin's price). You don't own the actual Bitcoin — you own a contract that pays out based on where Bitcoin's price goes. Think of it like betting on a horse race — you don't own the horse, you just profit or lose based on the result.

### What's a Futures Contract?
A futures contract is a specific type of derivative — an agreement to buy or sell something at a specific price on a specific future date. Traditional futures have **expiration dates** (e.g., "March 2026 Bitcoin Futures" expires in March). When it expires, the contract settles and you get your profit/loss.

### What's a Perpetual Future?
A perpetual future is a futures contract with **no expiration date**. You can hold it forever (as long as you have enough margin). This is crypto's innovation — traditional finance doesn't really have these. To keep the perpetual price close to the actual spot price, there's a **funding rate** — a small periodic payment between longs and shorts.

### Quick Comparison

| | Spot Trading | Traditional Futures | Perpetual Futures |
|---|---|---|---|
| **Own the asset?** | Yes | No | No |
| **Can short?** | No | Yes | Yes |
| **Expiration?** | N/A | Yes (monthly/quarterly) | No |
| **Leverage?** | No | Yes | Yes |
| **Funding rate?** | No | No | Yes |
| **Liquidation risk?** | No | Yes | Yes |
| **Best for** | Buy and hold | Hedging, institutions | Active trading, shorting |

### What Decisions Do We Need to Make?

1. **Perpetuals vs traditional futures?** → **Perpetuals.** No expiration means we don't have to roll contracts monthly. Our Donchian strategy holds for ~21 days on average — perpetuals let us hold without worrying about expiry. Traditional CME futures require rolling every month (selling the expiring contract, buying the next one), which adds complexity and slippage.

2. **Exchange?** → **Coinbase Financial Markets (CFM).** See Section 4 below. Already regulated for US, already our platform, has API access.

3. **Leverage level?** → **Start at 1x** (no amplification). Scale up to 2-3x max after the algo proves itself. At 1x, the position behaves almost like spot trading in terms of P&L — you just also have the ability to go short.

4. **Margin mode?** → **Isolated margin.** Each position's margin is ring-fenced. If one trade blows up, it can only lose the margin allocated to it — not your entire account.

5. **Collateral?** → **USDC.** Coinbase requires USDC for isolated margin. We'll need to hold USDC in the futures wallet.

---

## 1. Turtle Trading — The Foundation

The original Turtle Trading system (Richard Dennis, 1983) is the direct ancestor of our Donchian strategy. Crucially, **it traded both directions from day one** — longs AND shorts with identical rules, just mirrored.

### System 1 (Short-Term)
| Parameter | Long | Short |
|-----------|------|-------|
| **Entry** | Close > 20-day high | Close < 20-day low |
| **Exit** | Close < 10-day low | Close > 10-day high |
| **Stop** | 2N (2x ATR) from entry | 2N (2x ATR) from entry |
| **Position sizing** | 1% equity risk per "unit" | Same |
| **Max units** | 4 per market, 12 correlated | Same |
| **Pyramiding** | Add at +0.5N intervals (up to 4) | Add at -0.5N intervals (up to 4) |

### System 2 (Long-Term)
| Parameter | Long | Short |
|-----------|------|-------|
| **Entry** | Close > 55-day high | Close < 55-day low |
| **Exit** | Close < 20-day low | Close > 20-day high |
| **Stop** | 2N (2x ATR) from entry | Same |

### Key Turtle Rules
- **N = 20-day exponential moving average of True Range** (essentially ATR(20))
- **System 1 filter:** Skip a breakout if the previous breakout was profitable (contrarian filter to catch the "second try")
- **System 2:** Take every signal, no filter
- **Correlated market cap:** Max 6 units in same direction across correlated markets (e.g., all crypto = correlated)
- **Profit target:** None — let winners run, exits are systematic only

### Relevance to Our System
Our current long-only strategy is essentially **Turtle System 1 for longs with modifications**:
- We use 20-day Donchian high (same as Turtles)
- We use 10-day low exit (same as Turtles)
- We use ATR trailing stop (Turtles used fixed 2N stop)
- We add volume and EMA filters (Turtles had none)
- **We don't trade shorts — this is the gap**

---

## 2. Modern CTA / Managed Futures Approaches

### AQR Capital Management (Time-Series Momentum)
AQR's seminal research ("Time Series Momentum," Moskowitz, Ooi, Pedersen, 2012) established that:
- **12-month lookback** is the optimal period for trend signals across 58 instruments
- **Volatility scaling** is critical — position size = target vol / realized vol
- **Long AND short** signals generate roughly equal returns
- Returns persist for ~12 months, then partially reverse
- Works across all asset classes: equities, bonds, currencies, commodities

### How Modern CTAs Trade
1. **Signal generation:** 12-month return > 0 → long; < 0 → short (simplest form)
2. **Multi-timeframe blending:** Many CTAs blend 1-month, 3-month, 6-month, and 12-month signals
3. **Volatility targeting:** Scale position size to target a specific portfolio volatility (usually 10-15% annualized)
4. **Risk parity:** Equal risk allocation across instruments, not equal dollar allocation
5. **Carry overlay:** Some add funding rate / carry as a signal modifier

### Key CTA Statistics
- Managed futures as an industry: ~$400B AUM
- Average annual return: 8-12% with ~15% volatility
- **Sharpe ratio:** 0.5-0.8 typical, best firms 1.0+
- **Max drawdown:** 15-25% typical
- **Correlation to equities:** Near-zero (the whole point — crisis alpha)

---

## 3. Bear Market Asymmetry — Why Shorts Behave Differently

This is the **most critical finding** for our adaptation. Short trades are NOT mirror images of longs:

### Statistical Asymmetry
| Metric | Bull Markets | Bear Markets |
|--------|-------------|-------------|
| **Median monthly return** | +3-5% | -5-10% |
| **Duration** | 12-36 months | 3-12 months |
| **Speed** | Gradual | Violent |
| **Volatility** | Lower | 50-100% higher |
| **Mean reversion** | Slow | Fast snap-backs |

### Implications for Strategy Design
1. **Shorter lookback for shorts:** Bear breakdowns happen faster → 10-15 day lookback may beat 20-day
2. **Tighter stops on shorts:** Bear market rallies are violent → 2-3x ATR stop vs our 4x ATR for longs
3. **Faster profit-taking:** Bear moves are shorter → don't wait as long for trailing to trigger
4. **Higher win rate but smaller wins:** Shorts tend to be right more often but gains are capped by snap-back rallies
5. **Volatility explosion risk:** ATR can double in 48 hours during a crash → position sizing must use CURRENT vol, not historical

### Crypto-Specific Bear Patterns
- BTC bear markets last 12-18 months typically (2018, 2022)
- **Cascading liquidations** amplify moves — unique to crypto futures
- **Altcoin correlation** spikes to 0.95+ during crashes — shorting one is effectively shorting all
- **Funding rate flips negative** during prolonged bears — shorts get PAID

---

## 4. Exchange Analysis — US Regulatory Reality

### THE PROBLEM: Binance is Off Limits

**Binance Global (binance.com) does not allow US residents to trade futures.** Period. The CFTC regulates crypto derivatives in the US, and Binance's global platform explicitly blocks US users. Binance.US (the separate entity) only offers spot trading — no futures, no perpetuals, no margin.

Using a VPN to bypass this is against terms of service, risks account freezes, and has legal implications. **This is not an option.**

### US-Available Exchanges for Crypto Futures

| Exchange | Product | Pairs | Max Leverage | Fees (Maker/Taker) | Margin Modes | US Retail? | API? |
|----------|---------|-------|-------------|-------------------|-------------|-----------|------|
| **Coinbase CFM** | Perpetual-style futures | BTC, ETH + expanding | 10x | 0.00% / 0.03% (promo) | Isolated + Cross | **YES** | **YES** |
| **CME (via IBKR)** | Standard futures (monthly) | BTC, ETH only | ~5x effective | $0.85/contract (Micro BTC) | Exchange margin | YES | YES |
| **Bitnomial** | Perpetual futures | BTC, ETH, APT + others | TBD | TBD | Isolated + Cross | YES | Limited |
| **Kraken** | Perpetual futures | 100+ pairs | 50x | 0.02% / 0.05% | Isolated + Cross | **NO** (non-US only) | YES |

### RECOMMENDATION: Coinbase Financial Markets (CFM)

Coinbase launched **CFTC-regulated perpetual-style futures** for US retail traders on **July 21, 2025**. This is our clear winner:

**Why Coinbase CFM:**
1. **We're already on Coinbase** — same platform, same ecosystem, familiar API
2. **CFTC-regulated** — fully legal for US residents, no gray area
3. **Isolated margin available** — exactly what we want for risk control
4. **1x leverage supported** — perfect for our baby-steps approach
5. **API access** — Advanced Trade API supports perpetual futures (Market + Limit orders)
6. **USDC collateral** — clean, simple
7. **Fees are excellent** — 0.00% maker / 0.03% taker (promotional), normal ~0.06% / 0.09%
8. **No contract rolling** — perpetual-style means no monthly expiration hassle
9. **Min order: 10 USDC** — low barrier for testing

**Current status (confirmed Feb 27, 2026):**
- 15 perp pairs available on Coinbase Advanced under the **"Perps"** tab (not "Futures"):
  BTC, ETH, XRP, SOL, DOGE, DOT, LINK, BCH, ADA, SUI, XLM, HBAR, LTC, 1000SHIB, AVAX
- Account already has access — no application needed, leverage and margin fields visible
- **7 of our 8 coins have perps** (BTC, ETH, SOL, XRP, SUI, LINK, ADA). Only NEAR is missing.
- DOGE available as potential NEAR replacement (was already a candidate)
- Max 10x leverage (not a problem — we want 1x to start)
- Product is ~7 months old — maturing but may have some rough edges

**Fee comparison (round-trip):**
- Coinbase Spot: 0.70-1.50% round-trip (what killed our hourly strategy)
- **Coinbase CFM Perpetuals: 0.06% round-trip** (promo: 0.03%)
- That's **12-25x cheaper** — makes a massive difference on strategy viability

### Why Not CME via Interactive Brokers?

CME futures work but have significant drawbacks for our use case:
- **Monthly expiration** — must roll contracts, adds complexity and slippage
- **Only BTC + ETH** — no altcoins
- **Fixed contract sizes** — Micro BTC = 1/10 BTC (~$8,500), less flexible
- **Different API ecosystem** — would need IBKR integration from scratch
- **Institutional focus** — more paperwork, higher minimums
- **No perpetuals** — standard futures only

CME is a solid fallback if Coinbase CFM doesn't work out, but it would significantly increase development scope.

---

## 5. Leverage & Margin — Our Approach

### Starting at 1x Leverage (No Amplification)

At 1x leverage with isolated margin:

| Scenario | Your Position | Your Margin | Price Move | Your P&L | Liquidation? |
|----------|-------------|------------|-----------|---------|-------------|
| Short BTC at $85K | $10,000 short | $10,000 USDC | BTC drops 20% | +$2,000 (20%) | No |
| Short BTC at $85K | $10,000 short | $10,000 USDC | BTC rises 20% | -$2,000 (20%) | No |
| Short BTC at $85K | $10,000 short | $10,000 USDC | BTC rises 50% | -$5,000 (50%) | No |
| Short BTC at $85K | $10,000 short | $10,000 USDC | BTC doubles (+100%) | -$10,000 (100%) | ~Yes (near full loss) |

**At 1x, you can only be liquidated if the asset roughly doubles in price.** With our ATR trailing stops and emergency stop at 12-15%, we'd be out long before that happens. The emergency stop triggers at worst at -15%, meaning max loss per trade = 15% of position size = 15% of margin allocated.

### Leverage Progression Plan

| Phase | Leverage | When | Risk Profile |
|-------|---------|------|-------------|
| **Phase 1** | 1x | First 30-60 days | Zero amplification, prove the algo works |
| **Phase 2** | 2x | After 30+ trades, PF > 1.3 | Position is 2x margin, liquidation at ~50% adverse move |
| **Phase 3** | 3x | After 60+ trades, PF > 1.5 | Max target, liquidation at ~33% adverse move |

At each phase, position sizing stays the same (2% equity risk per trade). Higher leverage just means you need less margin per position, freeing capital for more concurrent trades or larger positions.

### Isolated Margin — The Safety Net

**How it works:** When you open a short, you allocate a specific amount of USDC as margin for that position. If the trade goes against you and hits liquidation, ONLY that margin is lost. Your other positions and your main balance are untouched.

**Example:**
- Account balance: $10,000 USDC
- Open short on BTC-PERP: allocate $2,000 margin (1x, $2,000 notional)
- Open short on ETH-PERP: allocate $2,000 margin (1x, $2,000 notional)
- Remaining: $6,000 USDC untouched
- If BTC doubles and you get liquidated: you lose $2,000. ETH position and $6,000 balance are safe.

This is exactly how our current spot strategy thinks about risk — 2% equity risk per trade, isolated per position.

---

## 6. Funding Rate — The Hidden Cost (or Income)

### How It Works
Every 8 hours (or hourly on Coinbase CFM, settled twice daily), a small payment is exchanged between longs and shorts to keep the perpetual price near spot price.

- **Positive funding rate (typical in bull markets):** Longs pay shorts
- **Negative funding rate (typical in bear markets):** Shorts pay longs

### Impact on Our Strategy

| Scenario | Our Position | Funding Direction | Impact |
|----------|-------------|------------------|--------|
| Bull market, we're long (spot bot) | N/A — spot has no funding | N/A | N/A |
| Bear market, we're short (futures bot) | Short | Shorts get PAID | **Tailwind** — ~0.03%/day income |
| Bear market rally, we're still short | Short | May flip positive briefly | Small cost |

**Key insight:** When the bull filter detects a bear market (BTC < SMA(200)), that's exactly when we'd activate shorts. During bear markets, funding rates tend to go negative, meaning **we get paid to hold shorts**. This is a natural tailwind that doesn't exist in spot trading.

With avg 21-day holds and ~0.01% per 8-hour interval:
- **Worst case (paying):** ~0.63% cost per trade
- **Best case (receiving):** ~0.63% income per trade
- **Backtests must model this** — use actual historical funding rates, not averages

---

## 7. Parameter Optimization — Short-Side Asymmetry

### Research Consensus on Short Parameters
Based on CTA literature and crypto backtest studies:

| Parameter | Long (Current) | Short (Recommended Start) | Rationale |
|-----------|---------------|--------------------------|-----------|
| **Donchian entry** | 20-day high | 15-day low | Bear breakdowns are faster |
| **Donchian exit** | 10-day low | 10-day high | Keep symmetric initially |
| **ATR trailing** | 4x ATR | 2.5-3x ATR | Bear rallies are violent, need tighter stops |
| **ATR period** | 14-day | 10-day | Faster vol adaptation in bear markets |
| **Volume filter** | 1.5x avg | 2.0x avg | Require stronger confirmation for shorts |
| **EMA trend** | > EMA(21) | < EMA(21) | Mirror for downtrend |
| **Bear filter** | BTC > SMA(200) | BTC < SMA(200) | Mirror — only short in bear regime |
| **Pyramiding** | +15% gain | +10% gain | Bear moves are faster, pyramid sooner |
| **Emergency stop** | 15% | 12% | Tighter risk control for shorts |
| **Max hold time** | None | 30 days | Prevent holding through bear-to-bull transition |
| **Partial TP** | 25% at +10% | 25% at +8% | Take profits faster |

### Why Not Just Mirror Exactly?
- Bear markets are **2-3x faster** than bull markets → parameters need to be tighter
- Bear market **mean reversion** is more violent → wider stops get stopped out by snap-back rallies
- **Volatility regime changes** happen faster in bears → ATR period should be shorter
- **Funding rate** is a tailwind for shorts during bears → can afford slightly tighter stops

---

## 8. Donchian Short Strategy — Backtesting Plan

### Proposed Backtest Matrix

**Base case (mirrored Turtle):**
- Entry: Close < 20-day low
- Exit: Close > 10-day high OR 2x ATR trailing stop
- Filter: BTC < SMA(200)

**Variants to test:**
1. **Lookback period:** 10, 15, 20, 25 day lows
2. **ATR multiplier:** 2x, 2.5x, 3x, 3.5x, 4x
3. **ATR period:** 10, 14, 20
4. **Volume threshold:** 1.0x, 1.5x, 2.0x, 2.5x
5. **Exit period:** 5, 10, 15, 20 day highs
6. **Bear filter:** BTC < SMA(200) only vs SMA(200) + death cross
7. **With/without pyramiding**
8. **Partial TP levels:** 8%/15% vs 10%/20%

### Historical Periods for Testing
- **2018 bear:** Jan 2018 - Dec 2018 (BTC $19K → $3.2K, -83%)
- **2019-2020 chop:** Jan 2019 - Mar 2020 (mixed, includes COVID crash)
- **2021 mid-year dip:** May 2021 - Jul 2021 (BTC $64K → $29K, -55%)
- **2022 bear:** Jan 2022 - Dec 2022 (BTC $47K → $16.5K, -65%)
- **2024 correction:** Mar 2024 - Sep 2024 (if data available)

### Success Criteria
- **Profit factor:** >= 1.5 (lower bar than longs due to shorter holds)
- **Win rate:** >= 40% (trend-following shorts typically lower)
- **Max drawdown:** <= 20%
- **Walk-forward validation:** Positive OOS returns
- **Combined long+short:** Better risk-adjusted returns than long-only

### Fee Assumptions for Backtest
| Cost | Value | Notes |
|------|-------|-------|
| **Trading fee** | 0.06% per side (0.12% round-trip) | Coinbase CFM retail rate |
| **Funding rate** | Use actual historical rates | Model per-position |
| **Slippage** | 0.05% per side | Conservative for daily candles |
| **Total per trade** | ~0.22% + funding | Much lower than spot (0.90%+) |

---

## 9. What Transfers from Current Work (~70%)

### Direct Transfers
1. **Donchian channel framework** — exact same concept, just mirrored
2. **ATR-based position sizing** — 2% risk per trade, sized by stop distance
3. **Volume confirmation** — works for breakdowns too
4. **EMA trend filter** — flip direction
5. **Pyramiding logic** — same concept, add on continuation
6. **Multi-coin portfolio management** — max positions, correlation awareness
7. **Bot architecture** — dual-loop (daily signal + intraday trailing), state persistence
8. **Notification system** — Telegram alerts, daily summaries
9. **Backtesting framework** — `backtest_donchian_daily.py` needs short-side addition
10. **Walk-forward validation** — same methodology
11. **Dashboard/TradeSavvy** — trade display, performance tracking
12. **Coinbase API familiarity** — same platform, similar API patterns

### Needs New Research (~30%)
1. **Short-side parameter optimization** — can't assume longs mirror perfectly
2. **Funding rate modeling** — must be in backtest P&L
3. **Coinbase CFM API integration** — perpetual futures endpoints (new, but same ecosystem)
4. **Margin monitoring** — track margin health, maintenance margin
5. **Combined long/short portfolio** — position limits, correlation, net exposure
6. ~~**Binance API integration**~~ — NOT NEEDED, staying on Coinbase
7. ~~**Liquidation risk management**~~ — At 1x, liquidation requires ~100% adverse move; our stops exit long before

---

## 10. Revised Phase F1 Execution Plan

### Week 1: Data & Infrastructure
- [ ] Verify Coinbase CFM perpetual pairs available (BTC, ETH — confirm SOL, XRP, ADA, etc.)
- [ ] Set up historical data pipeline for backtesting (can use Coinbase spot data — price action is the same)
- [ ] Add short-side signal generation to `donchian_breakout_strategy.py`
- [ ] Research Coinbase CFM historical funding rate data availability
- [ ] Set up Coinbase CFM API access (may need separate futures account/wallet)

### Week 2: Base Backtest
- [ ] Implement mirrored Turtle System 1 shorts (20-day low entry, 10-day high exit, 2x ATR stop)
- [ ] Run on all bear periods (2018, 2022) individually
- [ ] Compare to buy-and-hold and long-only results
- [ ] Add funding rate cost/income to P&L (use estimates if historical data unavailable)
- [ ] Model at 1x leverage with Coinbase CFM fee structure

### Week 3: Optimization & Validation
- [ ] Run parameter sweep (lookback, ATR mult, exit period, volume threshold)
- [ ] Identify top 3-5 parameter sets
- [ ] Walk-forward validation on each
- [ ] Combined long+short portfolio backtest (full 4-year period)
- [ ] Slippage stress test at Coinbase CFM fee levels
- [ ] Document findings and recommend final parameters

### Key Risks
1. **Overfitting:** More parameters to optimize = higher overfit risk → walk-forward is essential
2. **Crypto correlation:** All coins dump together → max short positions may need to be lower (2-3 vs 4)
3. **Short squeezes:** Violent rallies can blow through stops → must model this in backtest
4. **Funding rate variability:** Can swing widely → use actual historical rates where possible
5. **Coinbase CFM maturity:** Product is ~7 months old — API/liquidity may have rough edges
6. **Pair availability:** May be limited to BTC + ETH initially; altcoin shorts may come later

---

## 11. Bottom Line

**The current Donchian system is ~70% ready for bidirectional trading.** The core framework (Donchian channels, ATR stops, volume filters, position sizing) transfers directly.

**Exchange decision: Coinbase Financial Markets (CFM).** It's US-regulated, we're already on the platform, it has API access, supports isolated margin, and allows 1x leverage. Binance is off limits for US residents. CME is a fallback but adds complexity.

**Risk approach: 1x leverage, isolated margin, baby steps.** At 1x with isolated margin, max loss per position = allocated margin. With our ATR stops triggering at 12-15% adverse move, actual max loss per trade is well-defined and capped. No leverage amplification until the algo proves itself over 30-60 days.

**The main new work is:**
1. **Short-side parameter tuning** — bear markets are faster, parameters need adjustment
2. **Funding rate modeling** — must be in backtests (it's a tailwind for shorts in bears)
3. **Coinbase CFM API integration** — same ecosystem, new endpoints
4. **Combined long/short portfolio logic** — when to be long vs short vs flat

**Fees are transformative:** Coinbase CFM perpetuals are **12-25x cheaper** than Coinbase spot. This alone makes strategies viable that were killed by spot fees.

**Estimated timeline for F1 (research + backtest): 2-3 weeks of focused work.**

---

## Sources

- [Coinbase: Perpetual Futures Have Arrived in the US](https://www.coinbase.com/blog/perpetual-futures-have-arrived-in-the-us)
- [Coinbase: US Perpetual-Style Futures Launch](https://www.coinbase.com/blog/coming-july-21-us-perpetual-style-futures)
- [CryptoSlate: Coinbase CFTC-Regulated Perpetuals](https://cryptoslate.com/coinbase-starts-cftc-regulated-perpetuals-for-us-traders-offering-10x-leverage-and-0-02-fees/)
- [Coinbase Advanced Trade Perpetual Futures API](https://docs.cdp.coinbase.com/coinbase-business/advanced-trade-apis/guides/perpetual)
- [Interactive Brokers: CME Micro Bitcoin Futures](https://www.interactivebrokers.com/en/trading/cme-micro-bitcoin.php)
- [Interactive Brokers: Futures Commissions](https://www.interactivebrokers.com/en/pricing/commissions-futures.php)
- [Bitnomial: US Perpetual Futures](https://bitnomial.com/blog/us-perpetual-futures)
- [Kraken: Derivatives Fee Schedule](https://support.kraken.com/articles/360048917612-fee-schedule)
- [Coinbase: Funding Rate Mechanism](https://help.coinbase.com/en/derivatives/perpetual-style-futures/funding-rate)
