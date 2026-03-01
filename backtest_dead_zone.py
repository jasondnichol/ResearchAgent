"""Dead Zone Analysis — Regime Gap Research + A/B Testing + Sub-Daily Timeframe Sweep

The bot uses two independent filters that are NOT complementary:
  - Bull filter: BTC > SMA(200) AND SMA(50) > SMA(200)  →  gates longs
  - Bear filter: BTC < SMA(200) AND SMA(50) < SMA(200)  →  gates shorts

This creates 4 states, 2 of which are a "dead zone" where NO trades are allowed:
  - BULL:          price > SMA200, SMA50 > SMA200  →  longs open
  - DEATH_CROSS:   price < SMA200, SMA50 < SMA200  →  shorts open
  - DEAD_ZONE_1:   price > SMA200, SMA50 < SMA200  →  NOTHING (early recovery)
  - DEAD_ZONE_2:   price < SMA200, SMA50 > SMA200  →  NOTHING (early decline)

Phase 1: Measure dead zone frequency and characteristics
Phase 2: A/B test — relaxing filters on daily timeframe (5 variants)
Phase 3: Sub-daily timeframe sweep during dead zone periods (futures vs spot fees)
Phase 4: Summary and recommendations
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import math
import time as time_module
import requests

from backtest_donchian_daily import (
    fetch_all_coins,
    fetch_and_cache_daily_data,
    calculate_indicators,
    compute_stats,
    DEFAULT_PARAMS,
    COIN_UNIVERSE,
)
from backtest_bull_filter import compute_btc_bull_filter
from backtest_shorts import (
    compute_btc_bear_filter,
    backtest_portfolio_short,
    calculate_short_indicators,
    SHORT_DEFAULT_PARAMS,
    SHORT_COINS,
)
from backtest_phase3 import backtest_portfolio_phase3, PYRAMID_GAIN_PCT, PYRAMID_RISK_PCT
from backtest_walkforward import (
    split_coin_data,
    compute_sharpe,
    compute_max_drawdown,
    print_section,
    print_stats_row,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

LONG_COINS = [
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD',
    'SUI-USD', 'LINK-USD', 'ADA-USD', 'NEAR-USD',
]

SHORT_COINS_DZ = [
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD',
    'SUI-USD', 'LINK-USD', 'ADA-USD', 'DOGE-USD',
]

# Production long params (Phase 3: 4x ATR + pyramiding)
PROD_LONG_PARAMS = {
    **DEFAULT_PARAMS,
    'atr_mult': 4.0,
    'label': 'Prod Long (4x ATR)',
}

# Production short params (best death cross variant)
PROD_SHORT_PARAMS = {
    **SHORT_DEFAULT_PARAMS,
    'donchian_period': 10,
    'atr_mult': 2.0,
    'exit_period': 15,
    'volume_mult': 2.0,
    'label': 'Prod Short (L10/A2.0/E15/V2.0)',
}

# Fee models
FEE_SPOT = {'fee_pct': 0.40, 'slippage_pct': 0.05}
FEE_FUTURES = {'fee_pct': 0.06, 'slippage_pct': 0.05}


# ============================================================================
# PHASE 1: REGIME CLASSIFICATION
# ============================================================================

def classify_btc_regime(btc_df):
    """Classify each BTC day into one of 4 market regime states.

    Returns:
        regime_map: dict[date -> str]  (BULL, DEATH_CROSS, DEAD_ZONE_1, DEAD_ZONE_2)
        regime_df: DataFrame with columns [time, close, sma_50, sma_200, regime]
    """
    df = btc_df.copy()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()

    regime_map = {}
    rows = []
    for _, row in df.iterrows():
        date = row['time'].date()
        if pd.isna(row['sma_200']) or pd.isna(row['sma_50']):
            regime_map[date] = 'WARMUP'
            continue

        price_above_200 = row['close'] > row['sma_200']
        golden_cross = row['sma_50'] > row['sma_200']

        if price_above_200 and golden_cross:
            regime = 'BULL'
        elif not price_above_200 and not golden_cross:
            regime = 'DEATH_CROSS'
        elif price_above_200 and not golden_cross:
            regime = 'DEAD_ZONE_1'  # early recovery: price bounced, SMAs haven't crossed
        else:
            regime = 'DEAD_ZONE_2'  # early decline: price dipped, golden cross still active

        regime_map[date] = regime
        rows.append({
            'date': date,
            'close': row['close'],
            'sma_50': row['sma_50'],
            'sma_200': row['sma_200'],
            'regime': regime,
        })

    regime_df = pd.DataFrame(rows)
    return regime_map, regime_df


def compute_regime_stats(regime_map):
    """Compute statistics about regime durations and transitions."""
    # Filter out warmup
    filtered = {d: r for d, r in regime_map.items() if r != 'WARMUP'}
    dates = sorted(filtered.keys())
    total = len(dates)

    # Counts
    counts = {}
    for r in filtered.values():
        counts[r] = counts.get(r, 0) + 1

    # Streaks
    episodes = {}  # regime -> [(start, end, length), ...]
    if dates:
        current_regime = filtered[dates[0]]
        streak_start = dates[0]
        for i in range(1, len(dates)):
            r = filtered[dates[i]]
            if r != current_regime:
                # End of streak
                length = (dates[i-1] - streak_start).days + 1
                episodes.setdefault(current_regime, []).append((streak_start, dates[i-1], length))
                current_regime = r
                streak_start = dates[i]
        # Final streak
        length = (dates[-1] - streak_start).days + 1
        episodes.setdefault(current_regime, []).append((streak_start, dates[-1], length))

    return {
        'total': total,
        'counts': counts,
        'episodes': episodes,
        'date_range': (dates[0], dates[-1]) if dates else None,
    }


def print_regime_analysis(stats):
    """Print formatted regime analysis."""
    total = stats['total']
    counts = stats['counts']
    episodes = stats['episodes']

    print(f"\n  Regime Distribution ({total} trading days after SMA warmup):")
    print(f"  {'='*70}")

    order = ['BULL', 'DEATH_CROSS', 'DEAD_ZONE_1', 'DEAD_ZONE_2']
    labels = {
        'BULL': 'BULL (longs open)',
        'DEATH_CROSS': 'DEATH_CROSS (shorts open)',
        'DEAD_ZONE_1': 'DEAD ZONE 1 (price>SMA200, no golden cross)',
        'DEAD_ZONE_2': 'DEAD ZONE 2 (price<SMA200, golden cross)',
    }

    dz_total = counts.get('DEAD_ZONE_1', 0) + counts.get('DEAD_ZONE_2', 0)

    for regime in order:
        count = counts.get(regime, 0)
        pct = count / total * 100 if total > 0 else 0
        eps = episodes.get(regime, [])
        avg_streak = np.mean([e[2] for e in eps]) if eps else 0
        max_streak = max([e[2] for e in eps]) if eps else 0
        print(f"    {labels[regime]:<50s}  {count:>4d} days ({pct:>5.1f}%)  "
              f"avg streak: {avg_streak:>4.0f}d  max: {max_streak:>4d}d  ({len(eps)} episodes)")

    print(f"    {'─'*70}")
    dz_pct = dz_total / total * 100 if total > 0 else 0
    print(f"    {'TOTAL DEAD ZONE':<50s}  {dz_total:>4d} days ({dz_pct:>5.1f}%)")
    active_total = counts.get('BULL', 0) + counts.get('DEATH_CROSS', 0)
    active_pct = active_total / total * 100 if total > 0 else 0
    print(f"    {'TOTAL ACTIVE (trading allowed)':<50s}  {active_total:>4d} days ({active_pct:>5.1f}%)")

    # List dead zone episodes
    dz_episodes = []
    for regime in ['DEAD_ZONE_1', 'DEAD_ZONE_2']:
        for start, end, length in episodes.get(regime, []):
            dz_episodes.append((start, end, length, regime))
    dz_episodes.sort(key=lambda x: x[0])

    if dz_episodes:
        print(f"\n  Dead Zone Episodes:")
        print(f"  {'─'*70}")
        for start, end, length, regime in dz_episodes:
            tag = 'DZ1 (early recovery)' if regime == 'DEAD_ZONE_1' else 'DZ2 (early decline)'
            print(f"    {start} to {end}  ({length:>3d} days)  {tag}")


# ============================================================================
# PHASE 2: FILTER CONSTRUCTORS
# ============================================================================

def compute_relaxed_bull_filter(btc_df):
    """Bull filter relaxed: only require price > SMA200 (no golden cross needed).
    Allows longs during BULL + DEAD_ZONE_1 periods.
    """
    df = btc_df.copy()
    df['sma_200'] = df['close'].rolling(window=200).mean()

    filt = {}
    for _, row in df.iterrows():
        if pd.isna(row['sma_200']):
            filt[row['time'].date()] = False
            continue
        filt[row['time'].date()] = bool(row['close'] > row['sma_200'])

    bull_days = sum(1 for v in filt.values() if v)
    print(f"  Relaxed bull (>SMA200 only): {bull_days} bull days ({bull_days/len(filt)*100:.1f}%)")
    return filt


def compute_always_true_filter(btc_df):
    """Filter that allows entries on every day (completely unfiltered)."""
    filt = {}
    for _, row in btc_df.iterrows():
        filt[row['time'].date()] = True
    print(f"  Unfiltered: {len(filt)} days (100%)")
    return filt


def compute_dead_zone_only_filter(btc_df, regime_map):
    """Filter that only allows entries during dead zone periods."""
    filt = {}
    for _, row in btc_df.iterrows():
        date = row['time'].date()
        regime = regime_map.get(date, 'WARMUP')
        filt[date] = regime in ('DEAD_ZONE_1', 'DEAD_ZONE_2')

    dz_days = sum(1 for v in filt.values() if v)
    print(f"  Dead zone only: {dz_days} days ({dz_days/len(filt)*100:.1f}%)")
    return filt


# ============================================================================
# PHASE 2: A/B TEST RUNNER
# ============================================================================

def run_daily_ab_test(coin_data_long, coin_data_short, btc_df, regime_map):
    """Run 5 A/B variants on daily timeframe, full period + walk-forward."""
    print_section("PHASE 2: A/B TEST — DAILY TIMEFRAME (5 VARIANTS)")

    # Build all filter variants
    print("\n  Building filter variants...")
    bull_filter = compute_btc_bull_filter(btc_df)
    bear_filter_dc = compute_btc_bear_filter(btc_df, require_death_cross=True)
    bear_filter_simple = compute_btc_bear_filter(btc_df, require_death_cross=False)
    relaxed_bull = compute_relaxed_bull_filter(btc_df)
    always_true = compute_always_true_filter(btc_df)
    dead_zone_only = compute_dead_zone_only_filter(btc_df, regime_map)

    # Define variants: (name, long_filter, short_filter)
    variants = [
        ('A: Baseline (prod filters)',    bull_filter,    bear_filter_dc),
        ('B: Relaxed longs (>SMA200)',    relaxed_bull,   bear_filter_dc),
        ('C: Relaxed shorts (<SMA200)',   bull_filter,    bear_filter_simple),
        ('D: Fully unfiltered',           always_true,    always_true),
        ('E: Dead zone only',            dead_zone_only,  dead_zone_only),
    ]

    # ── Full period ──
    print(f"\n\n  {'='*140}")
    print(f"  FULL PERIOD RESULTS (2022-2026)")
    print(f"  {'='*140}")

    full_results = []
    for name, long_filt, short_filt in variants:
        # Run longs
        l_trades, l_eq, l_cap, l_pyrs = backtest_portfolio_phase3(
            coin_data_long, PROD_LONG_PARAMS, long_filt, pyramiding=True)
        l_stats = compute_stats(l_trades, f'{name} [LONG]')

        # Run shorts
        s_trades, s_eq, s_cap, s_pyrs = backtest_portfolio_short(
            coin_data_short, PROD_SHORT_PARAMS, short_filt, pyramiding=False)
        s_stats = compute_stats(s_trades, f'{name} [SHORT]')

        # Combined equity curve (additive PnL)
        starting = PROD_LONG_PARAMS['starting_capital']
        l_pnl = {e['date']: e['equity'] - starting for e in l_eq} if l_eq else {}
        s_pnl = {e['date']: e['equity'] - starting for e in s_eq} if s_eq else {}
        all_dates = sorted(set(list(l_pnl.keys()) + list(s_pnl.keys())))
        combined_eq = []
        for d in all_dates:
            eq = starting + l_pnl.get(d, 0) + s_pnl.get(d, 0)
            combined_eq.append({'date': d, 'equity': eq})

        all_trades = l_trades + s_trades
        c_stats = compute_stats(all_trades, f'{name} [COMBINED]')

        full_results.append({
            'name': name,
            'long': (l_stats, l_eq, l_pyrs),
            'short': (s_stats, s_eq, s_pyrs),
            'combined': (c_stats, combined_eq),
        })

    # Print combined results table
    print(f"\n  {'COMBINED (Long + Short)':<35s} {'Trades':>7s} {'WR':>7s} {'PF':>7s} "
          f"{'Return':>9s} {'MaxDD':>7s} {'Sharpe':>8s} {'Final':>12s}")
    print(f"  {'─'*100}")
    for r in full_results:
        c_stats, c_eq = r['combined']
        print_stats_row(r['name'], c_stats, c_eq)

    # Print long-only table
    print(f"\n  {'LONGS ONLY':<35s} {'Trades':>7s} {'WR':>7s} {'PF':>7s} "
          f"{'Return':>9s} {'MaxDD':>7s} {'Sharpe':>8s} {'Final':>12s}")
    print(f"  {'─'*100}")
    for r in full_results:
        l_stats, l_eq, _ = r['long']
        print_stats_row(r['name'], l_stats, l_eq)

    # Print short-only table
    print(f"\n  {'SHORTS ONLY':<35s} {'Trades':>7s} {'WR':>7s} {'PF':>7s} "
          f"{'Return':>9s} {'MaxDD':>7s} {'Sharpe':>8s} {'Final':>12s}")
    print(f"  {'─'*100}")
    for r in full_results:
        s_stats, s_eq, _ = r['short']
        print_stats_row(r['name'], s_stats, s_eq)

    # ── Walk-forward ──
    print_section("PHASE 2 WALK-FORWARD (Train 2022-2024, Test 2025-2026)")

    cutoff = pd.Timestamp('2025-01-01')
    _, test_long = split_coin_data(coin_data_long, cutoff)
    _, test_short = split_coin_data(coin_data_short, cutoff)

    btc_test = test_long.get('BTC-USD')
    if btc_test is None:
        btc_test = test_short.get('BTC-USD')
    if btc_test is None:
        print("  ERROR: No BTC test data")
        return full_results

    # Rebuild filters on test data
    bull_test = compute_btc_bull_filter(btc_test)
    bear_dc_test = compute_btc_bear_filter(btc_test, require_death_cross=True)
    bear_simple_test = compute_btc_bear_filter(btc_test, require_death_cross=False)
    relaxed_bull_test = compute_relaxed_bull_filter(btc_test)
    always_true_test = compute_always_true_filter(btc_test)

    regime_map_test, _ = classify_btc_regime(btc_test)
    dz_only_test = compute_dead_zone_only_filter(btc_test, regime_map_test)

    oos_variants = [
        ('A: Baseline (prod filters)',   bull_test,          bear_dc_test),
        ('B: Relaxed longs (>SMA200)',   relaxed_bull_test,  bear_dc_test),
        ('C: Relaxed shorts (<SMA200)',  bull_test,          bear_simple_test),
        ('D: Fully unfiltered',          always_true_test,   always_true_test),
        ('E: Dead zone only',           dz_only_test,        dz_only_test),
    ]

    print(f"\n  {'OOS COMBINED (2025-2026)':<35s} {'Trades':>7s} {'WR':>7s} {'PF':>7s} "
          f"{'Return':>9s} {'MaxDD':>7s} {'Sharpe':>8s} {'Final':>12s}")
    print(f"  {'─'*100}")

    oos_results = []
    for name, long_filt, short_filt in oos_variants:
        l_trades, l_eq, l_cap, _ = backtest_portfolio_phase3(
            test_long, PROD_LONG_PARAMS, long_filt, pyramiding=True)
        s_trades, s_eq, s_cap, _ = backtest_portfolio_short(
            test_short, PROD_SHORT_PARAMS, short_filt, pyramiding=False)

        starting = PROD_LONG_PARAMS['starting_capital']
        l_pnl = {e['date']: e['equity'] - starting for e in l_eq} if l_eq else {}
        s_pnl = {e['date']: e['equity'] - starting for e in s_eq} if s_eq else {}
        all_dates = sorted(set(list(l_pnl.keys()) + list(s_pnl.keys())))
        combined_eq = [{'date': d, 'equity': starting + l_pnl.get(d, 0) + s_pnl.get(d, 0)}
                       for d in all_dates]

        c_stats = compute_stats(l_trades + s_trades, f'{name} OOS')
        print_stats_row(name, c_stats, combined_eq)
        oos_results.append({'name': name, 'stats': c_stats, 'eq': combined_eq})

    return full_results


# ============================================================================
# PHASE 3: HOURLY DATA FETCH & CACHING
# ============================================================================

def fetch_and_cache_hourly_data(symbol='BTC-USD', cache_dir='cache_hourly', years=4, force_refresh=False):
    """Fetch and cache 4 years of hourly candles for a single coin."""
    os.makedirs(cache_dir, exist_ok=True)
    safe_symbol = symbol.replace('-', '_').lower()
    cache_file = os.path.join(cache_dir, f'{safe_symbol}_{years}yr_hourly.json')

    if not force_refresh and os.path.exists(cache_file):
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        age_days = (datetime.now() - file_time).days
        if age_days <= 7:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            df = pd.DataFrame(cache_data['data'])
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time').reset_index(drop=True)
            print(f"  {symbol}: loaded {len(df):,} hourly candles from cache")
            return df

    # Fetch from Coinbase
    coinbase_api = "https://api.exchange.coinbase.com"
    all_data = []
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=years * 365)
    chunk_hours = 300  # Coinbase max 300 candles per request

    total_hours = int((end_time - start_time).total_seconds() / 3600)
    num_chunks = math.ceil(total_hours / chunk_hours)
    print(f"  {symbol}: fetching {num_chunks} chunks of hourly data (~{total_hours:,} hours)...")

    chunk_start = start_time
    for i in range(num_chunks):
        chunk_end = min(chunk_start + timedelta(hours=chunk_hours), end_time)

        try:
            response = requests.get(
                f"{coinbase_api}/products/{symbol}/candles",
                params={
                    'start': chunk_start.isoformat(),
                    'end': chunk_end.isoformat(),
                    'granularity': 3600,
                },
                timeout=15,
            )
            if response.status_code == 200:
                data = response.json()
                if data:
                    all_data.extend(data)
            else:
                print(f"    chunk {i+1}: HTTP {response.status_code}")
        except Exception as e:
            print(f"    chunk {i+1}: error {e}")

        chunk_start = chunk_end
        if i % 20 == 0 and i > 0:
            print(f"    progress: {i}/{num_chunks} chunks ({len(all_data):,} candles)")
        time_module.sleep(0.3)  # rate limit safety

    if not all_data:
        print(f"  {symbol}: NO DATA fetched")
        return None

    # Deduplicate and sort
    seen = set()
    unique = []
    for row in all_data:
        ts = row[0] if isinstance(row, list) else row.get('time', row.get('start'))
        if ts not in seen:
            seen.add(ts)
            unique.append(row)

    df = pd.DataFrame(unique, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.sort_values('time').reset_index(drop=True)

    # Save cache
    cache_data = {
        'metadata': {
            'symbol': symbol,
            'granularity': '1h',
            'start': str(df['time'].iloc[0]),
            'end': str(df['time'].iloc[-1]),
            'total_candles': len(df),
            'fetched_at': str(datetime.utcnow()),
        },
        'data': df.to_dict(orient='records'),
    }
    # Convert timestamps to strings for JSON
    for row in cache_data['data']:
        row['time'] = str(row['time'])

    with open(cache_file, 'w') as f:
        json.dump(cache_data, f)

    print(f"  {symbol}: cached {len(df):,} hourly candles ({df['time'].iloc[0].date()} to {df['time'].iloc[-1].date()})")
    return df


def fetch_all_coins_hourly(coins=None, years=4):
    """Fetch hourly data for all coins."""
    coins = coins or LONG_COINS
    print(f"\n{'='*80}")
    print(f"FETCHING HOURLY DATA FOR {len(coins)} COINS ({years} years)")
    print(f"{'='*80}")

    coin_data = {}
    for symbol in coins:
        df = fetch_and_cache_hourly_data(symbol, years=years)
        if df is not None and len(df) > 500:
            coin_data[symbol] = df
        else:
            print(f"  {symbol}: SKIPPED (insufficient hourly data)")

    print(f"\nLoaded {len(coin_data)} coins with hourly data")
    return coin_data


# ============================================================================
# PHASE 3: TIMEFRAME AGGREGATION
# ============================================================================

def aggregate_to_timeframe(df_hourly, tf_hours):
    """Aggregate 1h OHLCV bars into higher timeframe bars."""
    if tf_hours == 1:
        return df_hourly.copy()

    df = df_hourly.copy()
    df = df.set_index('time')
    resampled = df.resample(f'{tf_hours}h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna(subset=['open'])
    resampled = resampled.reset_index()
    return resampled


# ============================================================================
# PHASE 3: SUB-DAILY DEAD ZONE BACKTEST ENGINE
# ============================================================================

def backtest_dead_zone_subdaily(coin_data_tf, regime_map, params, direction='long'):
    """Run Donchian breakout on sub-daily data, entries gated to dead zone only.

    Args:
        coin_data_tf: dict[symbol -> DataFrame] at target timeframe
        regime_map: dict[date -> str] from daily BTC classification
        params: strategy params (donchian_period, atr_mult, etc. scaled for timeframe)
        direction: 'long' or 'short'

    Returns: (trades, equity_curve, capital)
    """
    cost_per_side = (params['fee_pct'] + params['slippage_pct']) / 100
    max_positions = params['max_positions']
    risk_pct = params['risk_per_trade_pct'] / 100
    capital = params['starting_capital']
    funding_daily = params.get('funding_rate_daily', 0) / 100
    max_hold_bars = params.get('max_hold_bars', 999999)

    is_short = (direction == 'short')

    # Pre-calculate indicators
    prepared = {}
    for symbol, df in coin_data_tf.items():
        if is_short:
            df_ind = calculate_short_indicators(df, params)
            required = ['donchian_high', 'donchian_low', 'atr', 'volume_sma', 'ema_21', 'rsi', 'exit_high']
        else:
            df_ind = calculate_indicators(df, params)
            required = ['donchian_high', 'atr', 'volume_sma', 'ema_21', 'rsi']

        df_ind = df_ind.dropna(subset=required)
        df_ind = df_ind.reset_index(drop=True)
        if len(df_ind) > 50:
            prepared[symbol] = df_ind

    if not prepared:
        return [], [], capital

    # Build unified timeline (using bar indices)
    all_times = set()
    for symbol, df in prepared.items():
        all_times.update(df['time'].tolist())
    all_times = sorted(all_times)

    # Build lookups: symbol -> {time -> row}
    lookups = {}
    for symbol, df in prepared.items():
        lookup = {}
        for _, row in df.iterrows():
            lookup[row['time']] = row
        lookups[symbol] = lookup

    # Previous bar lookups
    prev_lookups = {}
    for symbol, df in prepared.items():
        prev_lookup = {}
        times_list = sorted(df['time'].tolist())
        for j in range(1, len(times_list)):
            prev_lookup[times_list[j]] = lookups[symbol].get(times_list[j-1])
        prev_lookups[symbol] = prev_lookup

    # State
    positions = {}
    trades = []
    equity_curve = []

    for t in all_times:
        bar_date = t.date() if hasattr(t, 'date') else pd.Timestamp(t).date()
        regime = regime_map.get(bar_date, 'WARMUP')
        is_dead_zone = regime in ('DEAD_ZONE_1', 'DEAD_ZONE_2')

        # === EXITS (always active) ===
        symbols_to_close = []
        for symbol, pos in list(positions.items()):
            row = lookups[symbol].get(t)
            if row is None:
                continue

            current_close = float(row['close'])
            current_atr = float(row['atr'])
            pos['bars_held'] += 1

            exit_reason = None

            # Blow-off / bounce risk
            vol_sma = float(row['volume_sma']) if row['volume_sma'] > 0 else 1
            volume_ratio = float(row['volume']) / vol_sma
            if is_short:
                is_risk = volume_ratio > params.get('volume_blowoff', 3.0) and float(row['rsi']) < params.get('rsi_blowoff', 20)
            else:
                is_risk = volume_ratio > params.get('volume_blowoff', 3.0) and float(row['rsi']) > params.get('rsi_blowoff', 80)
            stop_mult = params.get('atr_mult_tight', 1.5) if is_risk else params['atr_mult']

            if is_short:
                # Short trailing stop: low_watermark + ATR
                current_low = float(row['low'])
                pos['low_watermark'] = min(pos['low_watermark'], current_low)
                trailing_stop = pos['low_watermark'] + (stop_mult * current_atr)
                if current_close >= trailing_stop:
                    exit_reason = 'Trailing stop'

                # Donchian exit: close > N-day high
                prev_row = prev_lookups[symbol].get(t)
                if not exit_reason and prev_row is not None and pd.notna(prev_row.get('exit_high')):
                    if current_close > float(prev_row['exit_high']):
                        exit_reason = 'Donchian exit (high)'

                # Emergency stop
                emergency = pos['entry_price'] * (1 + params.get('emergency_stop_pct', 15.0) / 100)
                if not exit_reason and current_close >= emergency:
                    exit_reason = 'Emergency stop'

                # Max hold
                if not exit_reason and pos['bars_held'] >= max_hold_bars:
                    exit_reason = 'Max hold'

                # Partial TPs
                if not exit_reason:
                    gain_pct = ((pos['entry_price'] - current_close) / pos['entry_price']) * 100
                    if pos['partials_taken'] == 0 and gain_pct >= params.get('tp1_pct', 10):
                        partial_exit = current_close * (1 + cost_per_side)
                        partial_pnl = ((pos['entry_price'] - partial_exit) / pos['entry_price']) * 100
                        partial_size = pos['size_usd'] * params.get('tp1_fraction', 0.25)
                        capital += partial_size + partial_size * (partial_pnl / 100)
                        pos['size_usd'] -= partial_size
                        pos['partials_taken'] = 1
                        trades.append({
                            'symbol': symbol, 'pnl_pct': partial_pnl,
                            'exit_reason': 'Partial TP1', 'size_usd': partial_size,
                            'side': 'SHORT', 'win': partial_pnl > 0,
                            'entry_time': pos['entry_time'], 'exit_time': t,
                            'entry_price': pos['entry_price'], 'exit_price': partial_exit,
                        })
                    elif pos['partials_taken'] == 1 and gain_pct >= params.get('tp2_pct', 20):
                        partial_exit = current_close * (1 + cost_per_side)
                        partial_pnl = ((pos['entry_price'] - partial_exit) / pos['entry_price']) * 100
                        partial_size = pos['size_usd'] * params.get('tp2_fraction', 0.25)
                        capital += partial_size + partial_size * (partial_pnl / 100)
                        pos['size_usd'] -= partial_size
                        pos['partials_taken'] = 2
                        trades.append({
                            'symbol': symbol, 'pnl_pct': partial_pnl,
                            'exit_reason': 'Partial TP2', 'size_usd': partial_size,
                            'side': 'SHORT', 'win': partial_pnl > 0,
                            'entry_time': pos['entry_time'], 'exit_time': t,
                            'entry_price': pos['entry_price'], 'exit_price': partial_exit,
                        })

                # Full exit
                if exit_reason:
                    exit_price_adj = current_close * (1 + cost_per_side)
                    pnl_pct = ((pos['entry_price'] - exit_price_adj) / pos['entry_price']) * 100
                    # Add funding
                    hold_days = pos['bars_held'] * (params.get('tf_hours', 1) / 24.0)
                    pnl_pct += funding_daily * hold_days * 100
                    pnl_usd = pos['size_usd'] * (pnl_pct / 100)
                    capital += pos['size_usd'] + pnl_usd
                    trades.append({
                        'symbol': symbol, 'pnl_pct': pnl_pct, 'exit_reason': exit_reason,
                        'size_usd': pos['size_usd'], 'side': 'SHORT', 'win': pnl_pct > 0,
                        'entry_time': pos['entry_time'], 'exit_time': t,
                        'entry_price': pos['entry_price'], 'exit_price': exit_price_adj,
                    })
                    symbols_to_close.append(symbol)

            else:
                # Long trailing stop: high_watermark - ATR
                current_high = float(row['high'])
                pos['high_watermark'] = max(pos['high_watermark'], current_high)
                trailing_stop = pos['high_watermark'] - (stop_mult * current_atr)
                if current_close <= trailing_stop:
                    exit_reason = 'Trailing stop'

                # Donchian exit: close < N-day low
                prev_row = prev_lookups[symbol].get(t)
                if not exit_reason and prev_row is not None and pd.notna(prev_row.get('exit_low')):
                    if current_close < float(prev_row['exit_low']):
                        exit_reason = 'Donchian exit (low)'

                # Emergency stop (15%)
                if not exit_reason and current_close <= pos['entry_price'] * 0.85:
                    exit_reason = 'Emergency stop'

                # Partial TPs
                if not exit_reason:
                    gain_pct = ((current_close - pos['entry_price']) / pos['entry_price']) * 100
                    if pos['partials_taken'] == 0 and gain_pct >= params.get('tp1_pct', 10):
                        partial_price = current_close * (1 - cost_per_side)
                        partial_pnl = ((partial_price - pos['entry_price']) / pos['entry_price']) * 100
                        partial_size = pos['size_usd'] * params.get('tp1_fraction', 0.25)
                        capital += partial_size + partial_size * (partial_pnl / 100)
                        pos['size_usd'] -= partial_size
                        pos['partials_taken'] = 1
                        trades.append({
                            'symbol': symbol, 'pnl_pct': partial_pnl,
                            'exit_reason': 'Partial TP1', 'size_usd': partial_size,
                            'side': 'LONG', 'win': partial_pnl > 0,
                            'entry_time': pos['entry_time'], 'exit_time': t,
                            'entry_price': pos['entry_price'], 'exit_price': partial_price,
                        })
                    elif pos['partials_taken'] == 1 and gain_pct >= params.get('tp2_pct', 20):
                        partial_price = current_close * (1 - cost_per_side)
                        partial_pnl = ((partial_price - pos['entry_price']) / pos['entry_price']) * 100
                        partial_size = pos['size_usd'] * params.get('tp2_fraction', 0.25)
                        capital += partial_size + partial_size * (partial_pnl / 100)
                        pos['size_usd'] -= partial_size
                        pos['partials_taken'] = 2
                        trades.append({
                            'symbol': symbol, 'pnl_pct': partial_pnl,
                            'exit_reason': 'Partial TP2', 'size_usd': partial_size,
                            'side': 'LONG', 'win': partial_pnl > 0,
                            'entry_time': pos['entry_time'], 'exit_time': t,
                            'entry_price': pos['entry_price'], 'exit_price': partial_price,
                        })

                # Full exit
                if exit_reason:
                    exit_price_adj = current_close * (1 - cost_per_side)
                    pnl_pct = ((exit_price_adj - pos['entry_price']) / pos['entry_price']) * 100
                    pnl_usd = pos['size_usd'] * (pnl_pct / 100)
                    capital += pos['size_usd'] + pnl_usd
                    trades.append({
                        'symbol': symbol, 'pnl_pct': pnl_pct, 'exit_reason': exit_reason,
                        'size_usd': pos['size_usd'], 'side': 'LONG', 'win': pnl_pct > 0,
                        'entry_time': pos['entry_time'], 'exit_time': t,
                        'entry_price': pos['entry_price'], 'exit_price': exit_price_adj,
                    })
                    symbols_to_close.append(symbol)

        for sym in symbols_to_close:
            del positions[sym]

        # === NEW ENTRIES (gated to dead zone) ===
        if len(positions) < max_positions and is_dead_zone:
            for symbol in prepared:
                if symbol in positions:
                    continue
                if len(positions) >= max_positions:
                    break

                row = lookups[symbol].get(t)
                prev_row = prev_lookups[symbol].get(t)
                if row is None or prev_row is None:
                    continue

                current_close = float(row['close'])

                if is_short:
                    if pd.isna(prev_row.get('donchian_low')):
                        continue
                    breakdown = current_close < float(prev_row['donchian_low'])
                    if params.get('volume_mult', 0) > 0:
                        vol_sma = float(row['volume_sma']) if row['volume_sma'] > 0 else 1
                        volume_ok = float(row['volume']) > params['volume_mult'] * vol_sma
                    else:
                        volume_ok = True
                    trend_ok = current_close < float(row['ema_21'])
                    signal = breakdown and volume_ok and trend_ok
                else:
                    if pd.isna(prev_row.get('donchian_high')):
                        continue
                    breakout = current_close > float(prev_row['donchian_high'])
                    if params.get('volume_mult', 0) > 0:
                        vol_sma = float(row['volume_sma']) if row['volume_sma'] > 0 else 1
                        volume_ok = float(row['volume']) > params['volume_mult'] * vol_sma
                    else:
                        volume_ok = True
                    trend_ok = current_close > float(row['ema_21'])
                    signal = breakout and volume_ok and trend_ok

                if signal:
                    total_equity = capital + sum(p['size_usd'] for p in positions.values())
                    risk_amount = total_equity * risk_pct
                    atr_val = float(row['atr'])
                    stop_distance = params['atr_mult'] * atr_val

                    if is_short:
                        entry_price = current_close * (1 - cost_per_side)
                    else:
                        entry_price = current_close * (1 + cost_per_side)

                    stop_pct = stop_distance / entry_price
                    if stop_pct > 0:
                        position_size = risk_amount / stop_pct
                    else:
                        position_size = total_equity / max_positions

                    position_size = min(position_size, capital * 0.95)
                    if position_size < 100:
                        continue

                    capital -= position_size
                    pos_dict = {
                        'entry_price': entry_price,
                        'entry_time': t,
                        'partials_taken': 0,
                        'size_usd': position_size,
                        'bars_held': 0,
                    }
                    if is_short:
                        pos_dict['low_watermark'] = float(row['low'])
                    else:
                        pos_dict['high_watermark'] = float(row['high'])

                    positions[symbol] = pos_dict

        # Equity tracking
        total_equity = capital
        for symbol, pos in positions.items():
            row = lookups[symbol].get(t)
            if row is not None:
                if is_short:
                    price_change = (pos['entry_price'] - float(row['close'])) / pos['entry_price']
                    total_equity += max(pos['size_usd'] * (1 + price_change), 0)
                else:
                    current_val = pos['size_usd'] * (float(row['close']) / pos['entry_price'])
                    total_equity += current_val
            else:
                total_equity += pos['size_usd']

        equity_curve.append({'date': bar_date, 'equity': total_equity})

    # Close remaining
    for symbol, pos in list(positions.items()):
        df = prepared[symbol]
        last = df.iloc[-1]
        if is_short:
            exit_price_adj = float(last['close']) * (1 + cost_per_side)
            pnl_pct = ((pos['entry_price'] - exit_price_adj) / pos['entry_price']) * 100
            hold_days = pos['bars_held'] * (params.get('tf_hours', 1) / 24.0)
            pnl_pct += funding_daily * hold_days * 100
        else:
            exit_price_adj = float(last['close']) * (1 - cost_per_side)
            pnl_pct = ((exit_price_adj - pos['entry_price']) / pos['entry_price']) * 100
        pnl_usd = pos['size_usd'] * (pnl_pct / 100)
        capital += pos['size_usd'] + pnl_usd
        trades.append({
            'symbol': symbol, 'pnl_pct': pnl_pct, 'exit_reason': 'End of backtest',
            'size_usd': pos['size_usd'], 'side': 'SHORT' if is_short else 'LONG',
            'win': pnl_pct > 0,
            'entry_time': pos['entry_time'], 'exit_time': last['time'],
            'entry_price': pos['entry_price'], 'exit_price': exit_price_adj,
        })

    return trades, equity_curve, capital


# ============================================================================
# PHASE 3: SWEEP RUNNER
# ============================================================================

def run_subdaily_sweep(coin_data_hourly, regime_map):
    """Run full sub-daily timeframe sweep during dead zone."""
    print_section("PHASE 3: SUB-DAILY TIMEFRAME SWEEP (DEAD ZONE ONLY)")

    # Timeframe configs: (tf_hours, label, [(donchian_period, exit_period, atr_period, ema_period)])
    SWEEP_CONFIGS = [
        (12, '12H', [(40, 20, 28, 42), (30, 15, 21, 32), (20, 10, 14, 21)]),
        (6,  '6H',  [(80, 40, 56, 84), (60, 30, 42, 63), (40, 20, 28, 42), (20, 10, 14, 21)]),
        (4,  '4H',  [(120, 60, 84, 126), (80, 40, 56, 84), (60, 30, 42, 63), (40, 20, 28, 42), (20, 10, 14, 21)]),
        (2,  '2H',  [(120, 60, 84, 126), (80, 40, 56, 84), (40, 20, 28, 42), (20, 10, 14, 21)]),
        (1,  '1H',  [(120, 60, 84, 126), (80, 40, 56, 84), (40, 20, 28, 42), (20, 10, 14, 21)]),
    ]

    fee_models = {
        'spot':    {'fee_pct': 0.40, 'slippage_pct': 0.05, 'funding_rate_daily': 0},
        'futures': {'fee_pct': 0.06, 'slippage_pct': 0.05, 'funding_rate_daily': -0.03},
    }

    all_results = []
    total_combos = sum(len(lookbacks) for _, _, lookbacks in SWEEP_CONFIGS) * 2 * 2  # × fee × direction
    combo_num = 0

    for tf_hours, tf_label, lookback_configs in SWEEP_CONFIGS:
        # Aggregate hourly data to this timeframe
        print(f"\n  Aggregating to {tf_label}...")
        coin_data_tf = {}
        for symbol, df in coin_data_hourly.items():
            agg = aggregate_to_timeframe(df, tf_hours)
            if len(agg) > 200:
                coin_data_tf[symbol] = agg

        if not coin_data_tf:
            print(f"  {tf_label}: no data after aggregation, skipping")
            continue

        for donchian_period, exit_period, atr_period, ema_period in lookback_configs:
            for fee_name, fee_config in fee_models.items():
                for direction in ['long', 'short']:
                    combo_num += 1

                    base_params = SHORT_DEFAULT_PARAMS.copy() if direction == 'short' else DEFAULT_PARAMS.copy()
                    params = {
                        **base_params,
                        **fee_config,
                        'donchian_period': donchian_period,
                        'exit_period': exit_period,
                        'atr_period': atr_period,
                        'ema_period': ema_period,
                        'atr_mult': 2.0 if direction == 'short' else 4.0,
                        'volume_mult': 1.5,
                        'tf_hours': tf_hours,
                        'max_hold_bars': int(30 * 24 / tf_hours) if direction == 'short' else 999999,
                        'label': f'{tf_label} D{donchian_period} {fee_name} {direction}',
                    }

                    trades, eq, cap = backtest_dead_zone_subdaily(
                        coin_data_tf, regime_map, params, direction=direction)

                    stats = compute_stats(trades, params['label'])
                    sharpe = compute_sharpe(eq) if eq else 0
                    max_dd, _ = compute_max_drawdown(eq) if eq else (0, 0)
                    final = eq[-1]['equity'] if eq else params['starting_capital']

                    result = {
                        'tf': tf_label,
                        'tf_hours': tf_hours,
                        'donchian': donchian_period,
                        'exit_period': exit_period,
                        'fee_model': fee_name,
                        'direction': direction,
                        'trades': stats.get('total_trades', 0),
                        'wr': stats.get('win_rate', 0) * 100,
                        'pf': stats.get('profit_factor', 0),
                        'return_pct': stats.get('total_return_pct', 0),
                        'max_dd': max_dd,
                        'sharpe': sharpe,
                        'final': final,
                    }
                    all_results.append(result)

                    if combo_num % 10 == 0:
                        print(f"    [{combo_num}/{total_combos}] {params['label']}: "
                              f"{result['trades']} trades, {result['return_pct']:+.1f}%")

    # Print results tables
    print(f"\n\n  {'='*140}")
    print(f"  SUB-DAILY SWEEP RESULTS — LONGS (Dead Zone Only)")
    print(f"  {'='*140}")
    print(f"  {'TF':<5s} {'Donch':>5s} {'Fees':>8s} {'Trades':>7s} {'WR':>6s} {'PF':>6s} "
          f"{'Return':>8s} {'MaxDD':>7s} {'Sharpe':>7s} {'Final':>10s}")
    print(f"  {'─'*80}")
    long_results = sorted([r for r in all_results if r['direction'] == 'long'],
                          key=lambda x: x['return_pct'], reverse=True)
    for r in long_results:
        print(f"  {r['tf']:<5s} {r['donchian']:>5d} {r['fee_model']:>8s} {r['trades']:>7d} "
              f"{r['wr']:>5.1f}% {r['pf']:>5.2f} {r['return_pct']:>+7.1f}% "
              f"{r['max_dd']:>6.1f}% {r['sharpe']:>6.2f} ${r['final']:>9,.0f}")

    print(f"\n\n  {'='*140}")
    print(f"  SUB-DAILY SWEEP RESULTS — SHORTS (Dead Zone Only)")
    print(f"  {'='*140}")
    print(f"  {'TF':<5s} {'Donch':>5s} {'Fees':>8s} {'Trades':>7s} {'WR':>6s} {'PF':>6s} "
          f"{'Return':>8s} {'MaxDD':>7s} {'Sharpe':>7s} {'Final':>10s}")
    print(f"  {'─'*80}")
    short_results = sorted([r for r in all_results if r['direction'] == 'short'],
                           key=lambda x: x['return_pct'], reverse=True)
    for r in short_results:
        print(f"  {r['tf']:<5s} {r['donchian']:>5d} {r['fee_model']:>8s} {r['trades']:>7d} "
              f"{r['wr']:>5.1f}% {r['pf']:>5.2f} {r['return_pct']:>+7.1f}% "
              f"{r['max_dd']:>6.1f}% {r['sharpe']:>6.2f} ${r['final']:>9,.0f}")

    # Top 5 overall
    print(f"\n\n  {'='*140}")
    print(f"  TOP 10 COMBINATIONS (by return)")
    print(f"  {'='*140}")
    top = sorted(all_results, key=lambda x: x['return_pct'], reverse=True)[:10]
    for i, r in enumerate(top, 1):
        print(f"  #{i:>2d}  {r['tf']:<5s} D{r['donchian']:<4d} {r['fee_model']:>8s} {r['direction']:>6s}  "
              f"{r['trades']:>4d}t  {r['wr']:>5.1f}%WR  PF{r['pf']:>5.2f}  "
              f"{r['return_pct']:>+7.1f}%  DD{r['max_dd']:>5.1f}%  Sh{r['sharpe']:>5.2f}")

    return all_results


# ============================================================================
# PHASE 4: SUMMARY
# ============================================================================

def print_final_summary(regime_stats, full_results, subdaily_results):
    """Print consolidated summary and verdict."""
    print_section("PHASE 4: FINAL SUMMARY & VERDICT")

    total = regime_stats['total']
    counts = regime_stats['counts']
    dz_total = counts.get('DEAD_ZONE_1', 0) + counts.get('DEAD_ZONE_2', 0)
    dz_pct = dz_total / total * 100 if total > 0 else 0

    print(f"\n  REGIME DISTRIBUTION:")
    print(f"    BULL (longs):         {counts.get('BULL', 0):>4d} days ({counts.get('BULL', 0)/total*100:>5.1f}%)")
    print(f"    DEATH_CROSS (shorts): {counts.get('DEATH_CROSS', 0):>4d} days ({counts.get('DEATH_CROSS', 0)/total*100:>5.1f}%)")
    print(f"    DEAD ZONE:            {dz_total:>4d} days ({dz_pct:>5.1f}%)")

    if full_results:
        print(f"\n  DAILY A/B TEST (full period, combined long+short):")
        for r in full_results:
            c_stats, c_eq = r['combined']
            ret = c_stats.get('total_return_pct', 0)
            trades = c_stats.get('total_trades', 0)
            pf = c_stats.get('profit_factor', 0)
            max_dd, _ = compute_max_drawdown(c_eq) if c_eq else (0, 0)
            print(f"    {r['name']:<38s}  {trades:>4d}t  PF{pf:>5.2f}  {ret:>+7.1f}%  DD{max_dd:>5.1f}%")

    if subdaily_results:
        profitable = [r for r in subdaily_results if r['return_pct'] > 0 and r['trades'] >= 5]
        if profitable:
            best = max(profitable, key=lambda x: x['return_pct'])
            print(f"\n  BEST SUB-DAILY (dead zone only):")
            print(f"    {best['tf']} D{best['donchian']} {best['fee_model']} {best['direction']}  —  "
                  f"{best['trades']}t  PF{best['pf']:.2f}  {best['return_pct']:+.1f}%  DD{best['max_dd']:.1f}%")
        else:
            print(f"\n  BEST SUB-DAILY: No profitable combination found with 5+ trades")

        # Futures vs spot comparison
        futures_avg = np.mean([r['return_pct'] for r in subdaily_results if r['fee_model'] == 'futures' and r['trades'] >= 3]) if any(r['fee_model'] == 'futures' and r['trades'] >= 3 for r in subdaily_results) else 0
        spot_avg = np.mean([r['return_pct'] for r in subdaily_results if r['fee_model'] == 'spot' and r['trades'] >= 3]) if any(r['fee_model'] == 'spot' and r['trades'] >= 3 for r in subdaily_results) else 0
        print(f"\n  FEE MODEL COMPARISON (avg return across combos with 3+ trades):")
        print(f"    Futures (0.06%):  {futures_avg:>+.1f}%")
        print(f"    Spot (0.40%):     {spot_avg:>+.1f}%")

    # Verdict
    print(f"\n  {'='*70}")
    print(f"  VERDICT:")
    print(f"  {'='*70}")

    # Check if any dead zone approach beats sitting out
    baseline_ret = full_results[0]['combined'][0].get('total_return_pct', 0) if full_results else 0
    dz_only_ret = full_results[4]['combined'][0].get('total_return_pct', 0) if len(full_results) > 4 else 0

    if dz_only_ret > 2:
        print(f"    Dead zone trading is PROFITABLE ({dz_only_ret:+.1f}%) — worth investigating further")
    elif dz_only_ret > -2:
        print(f"    Dead zone trading is MARGINAL ({dz_only_ret:+.1f}%) — likely not worth the complexity")
    else:
        print(f"    Dead zone trading is LOSING ({dz_only_ret:+.1f}%) — current filter approach is correct")

    # Check if relaxed filters beat baseline
    if len(full_results) > 1:
        relaxed_long_ret = full_results[1]['combined'][0].get('total_return_pct', 0)
        relaxed_short_ret = full_results[2]['combined'][0].get('total_return_pct', 0)
        if relaxed_long_ret > baseline_ret + 2:
            print(f"    Relaxing bull filter (>SMA200 only) IMPROVES returns: {relaxed_long_ret:+.1f}% vs {baseline_ret:+.1f}%")
        if relaxed_short_ret > baseline_ret + 2:
            print(f"    Relaxing bear filter (<SMA200 only) IMPROVES returns: {relaxed_short_ret:+.1f}% vs {baseline_ret:+.1f}%")

    print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 100)
    print("DEAD ZONE ANALYSIS — REGIME GAPS + A/B TESTING + SUB-DAILY SWEEP")
    print("=" * 100)
    print(f"  Run time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Long coins: {', '.join(LONG_COINS)}")
    print(f"  Short coins: {', '.join(SHORT_COINS_DZ)}")

    # ── Fetch daily data ──
    all_coins = list(set(LONG_COINS + SHORT_COINS_DZ))
    coin_data_all = fetch_all_coins(coins=all_coins)
    coin_data_long = {s: coin_data_all[s] for s in LONG_COINS if s in coin_data_all}
    coin_data_short = {s: coin_data_all[s] for s in SHORT_COINS_DZ if s in coin_data_all}

    btc_df = coin_data_all.get('BTC-USD')
    if btc_df is None:
        print("ERROR: No BTC data")
        return

    # ── Phase 1: Regime classification ──
    print_section("PHASE 1: REGIME CLASSIFICATION (4-YEAR BTC)")
    regime_map, regime_df = classify_btc_regime(btc_df)
    regime_stats = compute_regime_stats(regime_map)
    print_regime_analysis(regime_stats)

    # ── Phase 2: Daily A/B test ──
    full_results = run_daily_ab_test(coin_data_long, coin_data_short, btc_df, regime_map)

    # ── Phase 3: Sub-daily sweep ──
    # Combine all coins for hourly fetch
    hourly_coins = list(set(LONG_COINS + SHORT_COINS_DZ))
    coin_data_hourly = fetch_all_coins_hourly(coins=hourly_coins)

    subdaily_results = []
    if coin_data_hourly:
        subdaily_results = run_subdaily_sweep(coin_data_hourly, regime_map)
    else:
        print("\n  SKIPPING Phase 3: No hourly data available")

    # ── Phase 4: Summary ──
    print_final_summary(regime_stats, full_results, subdaily_results)


if __name__ == '__main__':
    main()
