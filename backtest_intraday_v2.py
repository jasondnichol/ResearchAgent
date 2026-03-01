"""
Intraday Strategy V2 — Regime-Filtered + 12-Month Walk-Forward
================================================================
Adds regime filters to avoid ranging market losses:
1. ADX filter: Only trade when daily ADX > threshold (trending)
2. Volatility filter: Only trade when daily ATR rank is above median
3. BTC trend filter: Only trade when BTC daily EMA(21) slope is directional
4. Combined: ADX + volatility + BTC trend

Tests on 12 months of 1h data with proper walk-forward windows.

Usage:
    PYTHONIOENCODING=utf-8 venv/Scripts/python backtest_intraday_v2.py
"""

import os
import sys
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

from fetch_intraday import fetch_all_intraday, INTRADAY_COINS
from backtest_intraday import (
    add_indicators, backtest_strategy, compute_stats,
    strategy_mtf_momentum, strategy_intraday_donchian,
    STARTING_CAPITAL, FUTURES_FEE, MAX_POSITIONS, RISK_PER_TRADE,
    LEVERAGE, TRADE_COINS,
)
from backtest_intraday_wf import (
    fix_daily_context, walk_forward_backtest, print_walk_forward_results,
    parameter_robustness, print_robustness_results, slippage_stress_test,
)


def print_section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


# ── Regime Classification ────────────────────────────────────────────────────

def compute_daily_regime(coin_data, btc_key='BTC-USD'):
    """Classify each day as TRENDING or RANGING using multiple signals.

    Uses prior-day data only (no look-ahead):
    - ADX(14) on daily bars
    - Daily ATR rank (percentile over 20 days)
    - BTC EMA(21) slope (3-day rate of change)

    Returns: dict {date: {'regime': str, 'adx': float, 'atr_rank': float, 'btc_slope': float}}
    """
    # Build daily bars from hourly data
    daily_bars = {}
    for symbol, df in coin_data.items():
        df_copy = df.copy()
        df_copy['date'] = df_copy['time'].dt.date
        daily = df_copy.groupby('date').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }).reset_index()
        daily_bars[symbol] = daily

    # Compute ADX on BTC daily
    btc = daily_bars.get(btc_key)
    if btc is None or len(btc) < 30:
        return {}

    # True Range
    btc['prev_close'] = btc['close'].shift(1)
    btc['tr'] = pd.concat([
        btc['high'] - btc['low'],
        (btc['high'] - btc['prev_close']).abs(),
        (btc['low'] - btc['prev_close']).abs(),
    ], axis=1).max(axis=1)

    # +DM / -DM
    btc['plus_dm'] = np.where(
        (btc['high'] - btc['high'].shift(1)) > (btc['low'].shift(1) - btc['low']),
        np.maximum(btc['high'] - btc['high'].shift(1), 0), 0)
    btc['minus_dm'] = np.where(
        (btc['low'].shift(1) - btc['low']) > (btc['high'] - btc['high'].shift(1)),
        np.maximum(btc['low'].shift(1) - btc['low'], 0), 0)

    # Smoothed (Wilder's, period 14)
    period = 14
    btc['atr_14'] = btc['tr'].ewm(alpha=1/period, adjust=False).mean()
    btc['plus_di'] = 100 * btc['plus_dm'].ewm(alpha=1/period, adjust=False).mean() / btc['atr_14']
    btc['minus_di'] = 100 * btc['minus_dm'].ewm(alpha=1/period, adjust=False).mean() / btc['atr_14']
    btc['dx'] = 100 * (btc['plus_di'] - btc['minus_di']).abs() / (btc['plus_di'] + btc['minus_di']).replace(0, np.nan)
    btc['adx'] = btc['dx'].ewm(alpha=1/period, adjust=False).mean()

    # ATR rank (percentile over 20 days)
    btc['atr_rank'] = btc['atr_14'].rolling(20).rank(pct=True)

    # EMA(21) slope (3-day rate of change in %)
    btc['ema_21'] = btc['close'].ewm(span=21, adjust=False).mean()
    btc['ema_slope'] = (btc['ema_21'] - btc['ema_21'].shift(3)) / btc['ema_21'].shift(3) * 100

    # Classify — shift by 1 to use PRIOR day's values
    regime_map = {}
    for i in range(1, len(btc)):
        prev = btc.iloc[i-1]  # Use yesterday's indicators for today
        date = btc.iloc[i]['date']

        adx = prev['adx'] if not pd.isna(prev['adx']) else 15
        atr_rank = prev['atr_rank'] if not pd.isna(prev['atr_rank']) else 0.5
        ema_slope = prev['ema_slope'] if not pd.isna(prev['ema_slope']) else 0

        regime_map[date] = {
            'adx': adx,
            'atr_rank': atr_rank,
            'btc_slope': ema_slope,
            'regime': 'trending' if adx > 25 else 'ranging',
        }

    return regime_map


def add_regime_to_hourly(df, regime_map):
    """Map daily regime classification onto hourly bars."""
    df = df.copy()
    df['date'] = df['time'].dt.date
    df['regime'] = df['date'].map(lambda d: regime_map.get(d, {}).get('regime', 'unknown'))
    df['d_adx'] = df['date'].map(lambda d: regime_map.get(d, {}).get('adx', 15))
    df['d_atr_rank'] = df['date'].map(lambda d: regime_map.get(d, {}).get('atr_rank', 0.5))
    df['d_btc_slope'] = df['date'].map(lambda d: regime_map.get(d, {}).get('btc_slope', 0))
    return df


# ── Regime-Filtered Strategies ───────────────────────────────────────────────

def strategy_mtf_regime_filtered(df, params):
    """Multi-TF Momentum with regime filter.

    Only generates signals when the market is trending:
    - ADX filter: d_adx > adx_threshold (default 20)
    - Optional: ATR rank > 0.4 (above-median volatility)
    - Optional: |BTC slope| > 0.3% (BTC moving directionally)
    """
    vol_thresh = params.get('vol_threshold', 1.2)
    atr_stop = params.get('atr_stop', 2.0)
    atr_target = params.get('atr_target', 3.0)
    max_hold = params.get('max_hold', 24)
    adx_threshold = params.get('adx_threshold', 20)
    atr_rank_min = params.get('atr_rank_min', 0.0)  # 0 = no filter
    btc_slope_min = params.get('btc_slope_min', 0.0)  # 0 = no filter

    signals = []
    for i in range(52, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]

        if pd.isna(row.get('d_trend')) or pd.isna(row['atr']):
            continue
        if row['atr'] <= 0:
            continue

        # ── Regime filter ──
        d_adx = row.get('d_adx', 15)
        d_atr_rank = row.get('d_atr_rank', 0.5)
        d_btc_slope = row.get('d_btc_slope', 0)

        if pd.isna(d_adx):
            d_adx = 15
        if pd.isna(d_atr_rank):
            d_atr_rank = 0.5
        if pd.isna(d_btc_slope):
            d_btc_slope = 0

        if d_adx < adx_threshold:
            continue
        if d_atr_rank < atr_rank_min:
            continue
        if abs(d_btc_slope) < btc_slope_min:
            continue

        # ── Signal logic (same as original MTF) ──
        # Long: daily bullish, RSI dipped then bouncing
        if (row['d_trend'] == 1 and
            prev['rsi'] < 45 and row['rsi'] > prev['rsi'] and
            row['rsi'] > 35 and
            row['vol_ratio'] > vol_thresh and
            row['close'] > row['ema_21']):
            signals.append({
                'idx': i, 'time': row['time'], 'side': 'long',
                'entry': row['close'],
                'stop': row['close'] - atr_stop * row['atr'],
                'target': row['close'] + atr_target * row['atr'],
                'atr': row['atr'], 'max_hold': max_hold,
            })

        # Short: daily bearish, RSI rose then reversing
        if (row['d_trend'] == 0 and
            prev['rsi'] > 55 and row['rsi'] < prev['rsi'] and
            row['rsi'] < 65 and
            row['vol_ratio'] > vol_thresh and
            row['close'] < row['ema_21']):
            signals.append({
                'idx': i, 'time': row['time'], 'side': 'short',
                'entry': row['close'],
                'stop': row['close'] + atr_stop * row['atr'],
                'target': row['close'] - atr_target * row['atr'],
                'atr': row['atr'], 'max_hold': max_hold,
            })

    return signals


def strategy_donchian_regime_filtered(df, params):
    """Intraday Donchian with regime filter."""
    dc_period = params.get('dc_period', 48)
    atr_stop = params.get('atr_stop', 2.0)
    atr_target = params.get('atr_target', 3.0)
    vol_thresh = params.get('vol_threshold', 1.5)
    max_hold = params.get('max_hold', 24)
    adx_threshold = params.get('adx_threshold', 20)
    atr_rank_min = params.get('atr_rank_min', 0.0)

    signals = []
    for i in range(dc_period + 1, len(df)):
        row = df.iloc[i]

        if pd.isna(row['atr']) or row['atr'] <= 0:
            continue

        # Regime filter
        d_adx = row.get('d_adx', 15)
        if pd.isna(d_adx):
            d_adx = 15
        if d_adx < adx_threshold:
            continue

        d_atr_rank = row.get('d_atr_rank', 0.5)
        if pd.isna(d_atr_rank):
            d_atr_rank = 0.5
        if d_atr_rank < atr_rank_min:
            continue

        # Long signal
        dc_high = df['high'].iloc[i-dc_period:i].max()
        if (row['close'] > dc_high and
            row['vol_ratio'] > vol_thresh and
            row['close'] > row['ema_50']):
            signals.append({
                'idx': i, 'time': row['time'], 'side': 'long',
                'entry': row['close'],
                'stop': row['close'] - atr_stop * row['atr'],
                'target': row['close'] + atr_target * row['atr'],
                'atr': row['atr'], 'max_hold': max_hold,
            })

        # Short signal
        dc_low = df['low'].iloc[i-dc_period:i].min()
        if (row['close'] < dc_low and
            row['vol_ratio'] > vol_thresh and
            row['close'] < row['ema_50']):
            signals.append({
                'idx': i, 'time': row['time'], 'side': 'short',
                'entry': row['close'],
                'stop': row['close'] + atr_stop * row['atr'],
                'target': row['close'] - atr_target * row['atr'],
                'atr': row['atr'], 'max_hold': max_hold,
            })

    return signals


# ── Reporting ────────────────────────────────────────────────────────────────

def print_strategy_summary(name, result):
    """One-line summary of a strategy result."""
    s = result['stats']
    if not s:
        print(f"  {name:<40s}   — no trades —")
        return
    passed = (s['win_rate'] >= 52 and s['profit_factor'] >= 1.5) or \
             (s['profit_factor'] >= 1.8 and s['n_trades'] >= 20)
    v = "PASS" if passed else "FAIL"
    print(f"  [{v}] {name:<36s} {s['n_trades']:>4d} tr  {s['trades_per_day']:.1f}/d  "
          f"{s['win_rate']:.1f}% WR  PF {s['profit_factor']:.2f}  "
          f"{s['net_return_pct']:>+7.1f}%  DD {s['max_drawdown']:.1f}%  "
          f"${s['avg_daily_pnl']:>+.0f}/d")


def print_monthly_pnl(trades):
    """Print monthly P&L from trade list."""
    monthly = defaultdict(float)
    for t in trades:
        m = t['exit_time'].strftime('%Y-%m') if hasattr(t['exit_time'], 'strftime') else str(t['exit_time'])[:7]
        monthly[m] += t['net_pnl_dollar']

    print(f"\n  Monthly P&L:")
    for m in sorted(monthly.keys()):
        val = monthly[m]
        bar = '+' * min(60, max(0, int(val / 100))) if val > 0 else '-' * min(60, max(0, int(-val / 100)))
        print(f"    {m}: ${val:>+8,.0f}  {bar}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print_section("INTRADAY V2 — REGIME-FILTERED + 12-MONTH WALK-FORWARD")
    print(f"  Capital: ${STARTING_CAPITAL:,}  |  Leverage: {LEVERAGE}x  |  Fee: {FUTURES_FEE*100:.2f}%/side")

    # ── Load 12-month data ────────────────────────────────────
    print_section("LOADING 12-MONTH DATA")
    coin_data_raw = fetch_all_intraday(timeframe='1h', months=12)

    # ── Process: indicators + fixed daily context + regime ────
    print_section("COMPUTING INDICATORS + REGIME CLASSIFICATION")

    # Compute regime map from raw data (uses BTC daily bars)
    regime_map = compute_daily_regime(coin_data_raw)
    trending_days = sum(1 for v in regime_map.values() if v['regime'] == 'trending')
    ranging_days = sum(1 for v in regime_map.values() if v['regime'] == 'ranging')
    print(f"  Regime classification: {trending_days} trending days, {ranging_days} ranging days "
          f"({trending_days/(trending_days+ranging_days)*100:.0f}% trending)")

    # ADX distribution
    adx_values = [v['adx'] for v in regime_map.values()]
    print(f"  ADX distribution: min={min(adx_values):.1f}, median={np.median(adx_values):.1f}, "
          f"max={max(adx_values):.1f}")
    for thresh in [15, 20, 25, 30]:
        pct = sum(1 for a in adx_values if a >= thresh) / len(adx_values) * 100
        print(f"    ADX >= {thresh}: {pct:.0f}% of days")

    # Add indicators + regime to all coins
    coin_data = {}
    for symbol, df in coin_data_raw.items():
        print(f"  {symbol}: processing...", end='', flush=True)
        df = add_indicators(df)
        df = fix_daily_context(df)
        df = add_regime_to_hourly(df, regime_map)
        coin_data[symbol] = df
        print(f" {len(df):,} candles")

    # ═══════════════════════════════════════════════════════════
    # SECTION 1: Compare unfiltered vs regime-filtered
    # ═══════════════════════════════════════════════════════════
    print_section("SECTION 1: UNFILTERED vs REGIME-FILTERED")

    mtf_base = {'vol_threshold': 1.2, 'atr_stop': 2.0, 'atr_target': 3.0, 'max_hold': 24}
    dc_base = {'dc_period': 48, 'atr_stop': 2.0, 'atr_target': 3.0,
               'vol_threshold': 1.5, 'max_hold': 24}

    # Unfiltered baselines
    r_mtf_raw = backtest_strategy(coin_data, strategy_mtf_momentum, mtf_base)
    r_dc_raw = backtest_strategy(coin_data, strategy_intraday_donchian, dc_base)

    print(f"\n  Unfiltered baselines (12 months):")
    print_strategy_summary("MTF Momentum (no filter)", r_mtf_raw)
    print_strategy_summary("Donchian 48h (no filter)", r_dc_raw)

    # Regime-filtered variants — sweep ADX thresholds
    print(f"\n  ADX Filter Sweep — MTF Momentum:")
    print(f"  {'ADX Threshold':<20s} {'Trades':>6s} {'T/day':>5s} {'WR':>7s} {'PF':>6s} "
          f"{'Return':>9s} {'DD':>7s} {'$/day':>8s}")
    print(f"  {'─'*75}")

    best_mtf_adx = None
    best_mtf_daily = -9999

    for adx_t in [0, 15, 18, 20, 22, 25, 28, 30, 35]:
        p = {**mtf_base, 'adx_threshold': adx_t}
        r = backtest_strategy(coin_data, strategy_mtf_regime_filtered, p)
        s = r['stats']
        if s and s['n_trades'] > 0:
            marker = ""
            if s['avg_daily_pnl'] > best_mtf_daily and s['profit_factor'] >= 1.3:
                best_mtf_daily = s['avg_daily_pnl']
                best_mtf_adx = adx_t
                marker = " <--"
            print(f"  ADX >= {adx_t:<13d} {s['n_trades']:>6d} {s['trades_per_day']:>4.1f} "
                  f"{s['win_rate']:>6.1f}% {s['profit_factor']:>5.2f} "
                  f"{s['net_return_pct']:>+8.1f}% {s['max_drawdown']:>6.1f}% "
                  f"${s['avg_daily_pnl']:>+7.0f}{marker}")
        else:
            print(f"  ADX >= {adx_t:<13d}      0                — no trades —")

    print(f"\n  Best ADX for MTF: >= {best_mtf_adx}")

    print(f"\n  ADX Filter Sweep — Donchian 48h:")
    print(f"  {'ADX Threshold':<20s} {'Trades':>6s} {'T/day':>5s} {'WR':>7s} {'PF':>6s} "
          f"{'Return':>9s} {'DD':>7s} {'$/day':>8s}")
    print(f"  {'─'*75}")

    best_dc_adx = None
    best_dc_daily = -9999

    for adx_t in [0, 15, 18, 20, 22, 25, 28, 30, 35]:
        p = {**dc_base, 'adx_threshold': adx_t}
        r = backtest_strategy(coin_data, strategy_donchian_regime_filtered, p)
        s = r['stats']
        if s and s['n_trades'] > 0:
            marker = ""
            if s['avg_daily_pnl'] > best_dc_daily and s['profit_factor'] >= 1.2:
                best_dc_daily = s['avg_daily_pnl']
                best_dc_adx = adx_t
                marker = " <--"
            print(f"  ADX >= {adx_t:<13d} {s['n_trades']:>6d} {s['trades_per_day']:>4.1f} "
                  f"{s['win_rate']:>6.1f}% {s['profit_factor']:>5.2f} "
                  f"{s['net_return_pct']:>+8.1f}% {s['max_drawdown']:>6.1f}% "
                  f"${s['avg_daily_pnl']:>+7.0f}{marker}")
        else:
            print(f"  ADX >= {adx_t:<13d}      0                — no trades —")

    print(f"\n  Best ADX for Donchian: >= {best_dc_adx}")

    # ── Combined regime filter: ADX + ATR rank ────────────────
    print(f"\n  Combined Filter Sweep — MTF (ADX + ATR rank):")
    print(f"  {'Filter':<30s} {'Trades':>6s} {'T/day':>5s} {'WR':>7s} {'PF':>6s} "
          f"{'Return':>9s} {'DD':>7s} {'$/day':>8s}")
    print(f"  {'─'*82}")

    combined_configs = [
        ('ADX>=20 + ATR>0.3', {'adx_threshold': 20, 'atr_rank_min': 0.3, 'btc_slope_min': 0}),
        ('ADX>=20 + ATR>0.4', {'adx_threshold': 20, 'atr_rank_min': 0.4, 'btc_slope_min': 0}),
        ('ADX>=20 + Slope>0.2', {'adx_threshold': 20, 'atr_rank_min': 0, 'btc_slope_min': 0.2}),
        ('ADX>=20 + ATR>0.3 + Slope>0.2', {'adx_threshold': 20, 'atr_rank_min': 0.3, 'btc_slope_min': 0.2}),
        ('ADX>=25 + ATR>0.3', {'adx_threshold': 25, 'atr_rank_min': 0.3, 'btc_slope_min': 0}),
        ('ADX>=25 + Slope>0.3', {'adx_threshold': 25, 'atr_rank_min': 0, 'btc_slope_min': 0.3}),
        ('ADX>=22 + ATR>0.4', {'adx_threshold': 22, 'atr_rank_min': 0.4, 'btc_slope_min': 0}),
        ('ADX>=22 + ATR>0.3 + Slope>0.1', {'adx_threshold': 22, 'atr_rank_min': 0.3, 'btc_slope_min': 0.1}),
    ]

    best_combined_name = None
    best_combined_params = None
    best_combined_daily = -9999

    for label, extra in combined_configs:
        p = {**mtf_base, **extra}
        r = backtest_strategy(coin_data, strategy_mtf_regime_filtered, p)
        s = r['stats']
        if s and s['n_trades'] > 0:
            marker = ""
            if s['avg_daily_pnl'] > best_combined_daily and s['profit_factor'] >= 1.4:
                best_combined_daily = s['avg_daily_pnl']
                best_combined_name = label
                best_combined_params = p
                marker = " <--"
            print(f"  {label:<30s} {s['n_trades']:>6d} {s['trades_per_day']:>4.1f} "
                  f"{s['win_rate']:>6.1f}% {s['profit_factor']:>5.2f} "
                  f"{s['net_return_pct']:>+8.1f}% {s['max_drawdown']:>6.1f}% "
                  f"${s['avg_daily_pnl']:>+7.0f}{marker}")
        else:
            print(f"  {label:<30s}      0                — no trades —")

    print(f"\n  Best combined filter: {best_combined_name}")

    # ═══════════════════════════════════════════════════════════
    # SECTION 2: Walk-Forward on best configs
    # ═══════════════════════════════════════════════════════════
    print_section("SECTION 2: WALK-FORWARD VALIDATION (12-month data)")

    # Best MTF config
    mtf_best_params = {**mtf_base, 'adx_threshold': best_mtf_adx or 20}

    # Also test the best combined if different
    test_configs = [
        ('MTF Unfiltered', strategy_mtf_momentum, mtf_base),
        (f'MTF + ADX>={best_mtf_adx}', strategy_mtf_regime_filtered, mtf_best_params),
        ('Donchian Unfiltered', strategy_intraday_donchian, dc_base),
        (f'Donchian + ADX>={best_dc_adx}', strategy_donchian_regime_filtered,
         {**dc_base, 'adx_threshold': best_dc_adx or 20}),
    ]

    if best_combined_params:
        test_configs.append(
            (f'MTF + {best_combined_name}', strategy_mtf_regime_filtered, best_combined_params)
        )

    # Walk-forward: 3mo train / 1mo test / 1mo step (should give ~8 windows on 12mo data)
    for name, func, params in test_configs:
        print(f"\n  --- {name} ---")
        wf = walk_forward_backtest(coin_data, func, params,
                                    train_months=3, test_months=1, step_months=1)
        print_walk_forward_results(name, wf)

    # Also test with 4mo train / 2mo test
    print_section("WALK-FORWARD (4mo train / 2mo test)")
    for name, func, params in test_configs:
        print(f"\n  --- {name} ---")
        wf = walk_forward_backtest(coin_data, func, params,
                                    train_months=4, test_months=2, step_months=1)
        print_walk_forward_results(name, wf)

    # ═══════════════════════════════════════════════════════════
    # SECTION 3: Best strategy deep dive
    # ═══════════════════════════════════════════════════════════
    print_section("SECTION 3: BEST STRATEGY DEEP DIVE")

    # Determine best overall
    best_overall_name = None
    best_overall_result = None
    best_overall_daily = -9999

    for name, func, params in test_configs:
        r = backtest_strategy(coin_data, func, params)
        s = r['stats']
        if s and s['avg_daily_pnl'] > best_overall_daily and s['profit_factor'] >= 1.3:
            best_overall_daily = s['avg_daily_pnl']
            best_overall_name = name
            best_overall_result = r

    if best_overall_result:
        s = best_overall_result['stats']
        print(f"\n  Best strategy: {best_overall_name}")
        print(f"  Trades: {s['n_trades']}  |  Trades/day: {s['trades_per_day']:.1f}")
        print(f"  Win Rate: {s['win_rate']:.1f}%  |  PF: {s['profit_factor']:.2f}  |  Sharpe: {s['sharpe']:.2f}")
        print(f"  Net Return: {s['net_return_pct']:+.1f}% (${s['net_return_dollar']:+,.0f})")
        print(f"  Max Drawdown: {s['max_drawdown']:.1f}%")
        print(f"  Avg Win: {s['avg_win_pct']:+.2f}%  |  Avg Loss: {s['avg_loss_pct']:.2f}%")
        print(f"  Longs: {s['n_long']} ({s['long_wr']:.0f}% WR)  |  Shorts: {s['n_short']} ({s['short_wr']:.0f}% WR)")
        print(f"  Profitable Days: {s['profitable_days_pct']:.1f}%  |  Avg Daily P&L: ${s['avg_daily_pnl']:+.0f}")
        print(f"  Exits: {s['exit_counts']}")

        # Per-coin breakdown
        coin_pnl = defaultdict(float)
        coin_trades_n = defaultdict(int)
        for t in best_overall_result['trades']:
            coin_pnl[t['symbol']] += t['net_pnl_dollar']
            coin_trades_n[t['symbol']] += 1

        print(f"\n  Per-Coin P&L:")
        for sym in sorted(coin_pnl.keys(), key=lambda s: coin_pnl[s], reverse=True):
            print(f"    {sym:10s}: ${coin_pnl[sym]:>+8,.0f} ({coin_trades_n[sym]} trades)")

        print_monthly_pnl(best_overall_result['trades'])

    # ── Slippage stress test on best ──────────────────────────
    print_section("SLIPPAGE STRESS TEST (Best Strategy)")
    if best_overall_name:
        # Find the func/params
        for name, func, params in test_configs:
            if name == best_overall_name:
                slippage_stress_test(coin_data, func, params)
                break

    # ═══════════════════════════════════════════════════════════
    # SECTION 4: Path to $500/day
    # ═══════════════════════════════════════════════════════════
    print_section("PATH TO $500/DAY")

    if best_overall_result:
        s = best_overall_result['stats']
        daily = s['avg_daily_pnl']
        daily_rate = daily / STARTING_CAPITAL

        print(f"\n  Best strategy: {best_overall_name}")
        print(f"  Avg daily P&L on ${STARTING_CAPITAL:,}: ${daily:+.0f} ({daily_rate*100:.2f}%/day)")

        if daily > 0:
            # Compounding projection
            eq = STARTING_CAPITAL
            days = 0
            milestones = {20000: None, 30000: None, 50000: None}
            while eq < 100000 and days < 365:
                eq *= (1 + daily_rate)
                days += 1
                for m in milestones:
                    if milestones[m] is None and eq >= m:
                        milestones[m] = days

            print(f"\n  Compounding Projection:")
            for m, d in sorted(milestones.items()):
                if d:
                    daily_at_m = m * daily_rate
                    print(f"    ${m/1000:.0f}K: day {d:>3d}  |  daily P&L at this capital: ${daily_at_m:+.0f}")

            capital_500 = 500 / daily_rate if daily_rate > 0 else float('inf')
            print(f"\n  Capital needed for $500/day: ${capital_500:,.0f}")
            print(f"  Capital needed for $300/day: ${300/daily_rate:,.0f}")

            # Conservative estimate (use WF OOS avg instead of full-period)
            print(f"\n  NOTE: These projections use full-period averages.")
            print(f"  Walk-forward OOS performance is the realistic expectation.")
            print(f"  Check WF results above for true out-of-sample daily P&L.")
        else:
            print(f"\n  Strategy has negative daily P&L. Not viable for income target.")

    # ── Final comparison table ────────────────────────────────
    print_section("FINAL COMPARISON")
    print(f"\n  {'Strategy':<40s} {'Tr':>4s} {'T/d':>4s} {'WR':>6s} {'PF':>5s} "
          f"{'Ret':>8s} {'DD':>6s} {'$/d':>6s}")
    print(f"  {'─'*80}")

    for name, func, params in test_configs:
        r = backtest_strategy(coin_data, func, params)
        s = r['stats']
        if s:
            print(f"  {name:<40s} {s['n_trades']:>4d} {s['trades_per_day']:>3.1f} "
                  f"{s['win_rate']:>5.1f}% {s['profit_factor']:>4.2f} "
                  f"{s['net_return_pct']:>+7.1f}% {s['max_drawdown']:>5.1f}% "
                  f"${s['avg_daily_pnl']:>+5.0f}")

    print(f"\n{'='*80}")
    print(f"  COMPLETE")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
