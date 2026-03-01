"""
Walk-Forward Validation for Intraday Strategies
=================================================
Tests whether the intraday edges hold out-of-sample using
rolling walk-forward windows.

Validation approach:
1. Walk-forward: 4-month train → 2-month test, rolling monthly
2. Parameter robustness: sweep key params, check stability
3. Regime analysis: performance in trending vs ranging markets
4. Slippage stress test: add 0.05%, 0.10%, 0.15% slippage
5. Look-ahead bias audit: verify daily context uses prior-day only

Usage:
    PYTHONIOENCODING=utf-8 venv/Scripts/python backtest_intraday_wf.py
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
    add_indicators, add_daily_context, strategy_mtf_momentum,
    strategy_intraday_donchian, backtest_strategy, compute_stats,
    STARTING_CAPITAL, FUTURES_FEE, MAX_POSITIONS, RISK_PER_TRADE,
    LEVERAGE, TRADE_COINS, print_section,
)


# ── Walk-Forward Engine ──────────────────────────────────────────────────────

def walk_forward_backtest(coin_data_processed, strategy_func, params,
                          train_months=4, test_months=2, step_months=1):
    """Run rolling walk-forward validation.

    Splits data into overlapping windows:
    - Window 1: Train months 1-4, Test months 5-6
    - Window 2: Train months 2-5, Test months 6-7 (if data available)
    - etc.

    For each window, computes in-sample and out-of-sample stats.
    The strategy parameters are FIXED (no re-optimization) — we're testing
    whether the edge generalizes, not whether we can overfit each window.
    """
    # Find the common date range across all coins
    all_starts = []
    all_ends = []
    for symbol, df in coin_data_processed.items():
        if symbol in TRADE_COINS:
            all_starts.append(df['time'].min())
            all_ends.append(df['time'].max())

    if not all_starts:
        return []

    data_start = max(all_starts)
    data_end = min(all_ends)
    total_days = (data_end - data_start).days

    # Generate windows
    windows = []
    train_days = train_months * 30
    test_days = test_months * 30
    step_days = step_months * 30
    window_days = train_days + test_days

    offset = 0
    while offset + window_days <= total_days:
        w_start = data_start + timedelta(days=offset)
        train_end = w_start + timedelta(days=train_days)
        test_end = train_end + timedelta(days=test_days)

        windows.append({
            'train_start': w_start,
            'train_end': train_end,
            'test_start': train_end,
            'test_end': min(test_end, data_end),
        })
        offset += step_days

    if not windows:
        print("  Not enough data for walk-forward windows")
        return []

    print(f"  Walk-forward: {len(windows)} windows "
          f"({train_months}mo train / {test_months}mo test / {step_months}mo step)")
    print(f"  Data range: {data_start.date()} to {data_end.date()} ({total_days} days)")

    # Run each window
    results = []
    for i, w in enumerate(windows):
        # Split data for each coin
        train_data = {}
        test_data = {}
        for symbol, df in coin_data_processed.items():
            if symbol not in TRADE_COINS:
                continue
            train_df = df[(df['time'] >= w['train_start']) & (df['time'] < w['train_end'])].copy()
            test_df = df[(df['time'] >= w['test_start']) & (df['time'] < w['test_end'])].copy()

            if len(train_df) > 50:
                train_data[symbol] = train_df.reset_index(drop=True)
            if len(test_df) > 50:
                test_data[symbol] = test_df.reset_index(drop=True)

        # Run backtest on train and test
        train_result = backtest_strategy(train_data, strategy_func, params)
        test_result = backtest_strategy(test_data, strategy_func, params)

        train_stats = train_result['stats']
        test_stats = test_result['stats']

        results.append({
            'window': i + 1,
            'train_period': f"{w['train_start'].date()} → {w['train_end'].date()}",
            'test_period': f"{w['test_start'].date()} → {w['test_end'].date()}",
            'train_stats': train_stats,
            'test_stats': test_stats,
        })

    return results


def print_walk_forward_results(name, results):
    """Print walk-forward results in a clean table."""
    print(f"\n  {name} — Walk-Forward Results")
    print(f"  {'─'*90}")
    print(f"  {'Window':<8s} {'Train Period':<26s} {'Test Period':<26s} "
          f"{'IS Trades':>9s} {'IS WR':>7s} {'IS PF':>6s} {'IS Ret':>8s} "
          f"{'OOS Trades':>10s} {'OOS WR':>7s} {'OOS PF':>7s} {'OOS Ret':>8s}")
    print(f"  {'─'*90}")

    oos_returns = []
    oos_pfs = []
    oos_wrs = []
    oos_trades = []
    oos_dailys = []

    for r in results:
        ts = r['train_stats']
        os_s = r['test_stats']

        ts_trades = ts['n_trades'] if ts else 0
        ts_wr = f"{ts['win_rate']:.0f}%" if ts else "—"
        ts_pf = f"{ts['profit_factor']:.2f}" if ts else "—"
        ts_ret = f"{ts['net_return_pct']:+.1f}%" if ts else "—"

        os_trades_n = os_s['n_trades'] if os_s else 0
        os_wr = f"{os_s['win_rate']:.0f}%" if os_s else "—"
        os_pf = f"{os_s['profit_factor']:.2f}" if os_s else "—"
        os_ret = f"{os_s['net_return_pct']:+.1f}%" if os_s else "—"

        print(f"  W{r['window']:<7d} {r['train_period']:<26s} {r['test_period']:<26s} "
              f"{ts_trades:>9d} {ts_wr:>7s} {ts_pf:>6s} {ts_ret:>8s} "
              f"{os_trades_n:>10d} {os_wr:>7s} {os_pf:>7s} {os_ret:>8s}")

        if os_s:
            oos_returns.append(os_s['net_return_pct'])
            oos_pfs.append(os_s['profit_factor'])
            oos_wrs.append(os_s['win_rate'])
            oos_trades.append(os_s['n_trades'])
            oos_dailys.append(os_s['avg_daily_pnl'])

    if oos_returns:
        print(f"  {'─'*90}")
        avg_ret = np.mean(oos_returns)
        avg_pf = np.mean(oos_pfs)
        avg_wr = np.mean(oos_wrs)
        avg_trades = np.mean(oos_trades)
        avg_daily = np.mean(oos_dailys)
        profitable_windows = sum(1 for r in oos_returns if r > 0)

        print(f"\n  OOS Summary ({len(oos_returns)} windows):")
        print(f"    Avg OOS return: {avg_ret:+.1f}%  |  Avg PF: {avg_pf:.2f}  |  Avg WR: {avg_wr:.1f}%")
        print(f"    Avg trades/window: {avg_trades:.0f}  |  Avg daily P&L: ${avg_daily:+.0f}")
        print(f"    Profitable windows: {profitable_windows}/{len(oos_returns)} ({profitable_windows/len(oos_returns)*100:.0f}%)")
        print(f"    OOS return range: {min(oos_returns):+.1f}% to {max(oos_returns):+.1f}%")

        # Verdict
        if avg_pf >= 1.5 and profitable_windows >= len(oos_returns) * 0.6:
            print(f"\n  VERDICT: PASS — Edge holds out-of-sample")
        elif avg_pf >= 1.2 and profitable_windows >= len(oos_returns) * 0.5:
            print(f"\n  VERDICT: MARGINAL — Some edge but inconsistent")
        else:
            print(f"\n  VERDICT: FAIL — Edge does not generalize")

    return oos_returns, oos_pfs


# ── Parameter Robustness ─────────────────────────────────────────────────────

def parameter_robustness(coin_data_processed, strategy_func, base_params, param_grid):
    """Test stability across parameter variations.

    param_grid: dict of {param_name: [values_to_test]}
    Runs a 1-at-a-time sweep, holding others at base values.
    """
    results = []

    # Baseline
    base_result = backtest_strategy(coin_data_processed, strategy_func, base_params)
    base_stats = base_result['stats']
    if base_stats:
        results.append({
            'label': 'BASELINE',
            'params': dict(base_params),
            'n_trades': base_stats['n_trades'],
            'win_rate': base_stats['win_rate'],
            'profit_factor': base_stats['profit_factor'],
            'net_return': base_stats['net_return_pct'],
            'max_dd': base_stats['max_drawdown'],
            'avg_daily': base_stats['avg_daily_pnl'],
            'trades_per_day': base_stats['trades_per_day'],
        })

    for param_name, values in param_grid.items():
        for val in values:
            test_params = dict(base_params)
            test_params[param_name] = val

            # Skip if same as baseline
            if test_params == base_params:
                continue

            r = backtest_strategy(coin_data_processed, strategy_func, test_params)
            s = r['stats']
            if s:
                results.append({
                    'label': f'{param_name}={val}',
                    'params': test_params,
                    'n_trades': s['n_trades'],
                    'win_rate': s['win_rate'],
                    'profit_factor': s['profit_factor'],
                    'net_return': s['net_return_pct'],
                    'max_dd': s['max_drawdown'],
                    'avg_daily': s['avg_daily_pnl'],
                    'trades_per_day': s['trades_per_day'],
                })
            else:
                results.append({
                    'label': f'{param_name}={val}',
                    'params': test_params,
                    'n_trades': 0,
                    'win_rate': 0,
                    'profit_factor': 0,
                    'net_return': 0,
                    'max_dd': 0,
                    'avg_daily': 0,
                    'trades_per_day': 0,
                })

    return results


def print_robustness_results(name, results):
    """Print parameter robustness table."""
    print(f"\n  {name} — Parameter Robustness")
    print(f"  {'─'*90}")
    print(f"  {'Variant':<30s} {'Trades':>6s} {'T/day':>5s} {'WR':>7s} {'PF':>6s} "
          f"{'Return':>9s} {'DD':>7s} {'$/day':>8s}")
    print(f"  {'─'*90}")

    profitable = 0
    for r in results:
        marker = " <<<" if r['label'] == 'BASELINE' else ""
        print(f"  {r['label']:<30s} {r['n_trades']:>6d} {r['trades_per_day']:>4.1f} "
              f"{r['win_rate']:>6.1f}% {r['profit_factor']:>5.2f} "
              f"{r['net_return']:>+8.1f}% {r['max_dd']:>6.1f}% ${r['avg_daily']:>+7.0f}{marker}")
        if r['profit_factor'] > 1.0:
            profitable += 1

    print(f"\n  Profitable variants: {profitable}/{len(results)} ({profitable/max(len(results),1)*100:.0f}%)")

    # Check stability
    pfs = [r['profit_factor'] for r in results if r['n_trades'] > 10]
    if pfs:
        pf_std = np.std(pfs)
        pf_mean = np.mean(pfs)
        cv = pf_std / pf_mean if pf_mean > 0 else float('inf')
        print(f"  PF stability: mean={pf_mean:.2f}, std={pf_std:.2f}, CV={cv:.2f}")
        if cv < 0.3:
            print(f"  Robustness: STABLE (CV < 0.30)")
        elif cv < 0.5:
            print(f"  Robustness: MODERATE (CV 0.30-0.50)")
        else:
            print(f"  Robustness: FRAGILE (CV > 0.50)")


# ── Slippage Stress Test ─────────────────────────────────────────────────────

def slippage_stress_test(coin_data_processed, strategy_func, params):
    """Test performance degradation with increasing slippage."""
    slippage_levels = [0.0, 0.0003, 0.0005, 0.001, 0.0015, 0.002]

    print(f"\n  Slippage Stress Test")
    print(f"  {'─'*70}")
    print(f"  {'Slippage':>10s} {'Trades':>7s} {'WR':>7s} {'PF':>6s} "
          f"{'Return':>9s} {'$/day':>8s} {'vs Base':>9s}")
    print(f"  {'─'*70}")

    base_return = None
    for slip in slippage_levels:
        # Add slippage as additional fee
        total_fee = FUTURES_FEE + slip
        r = backtest_strategy(coin_data_processed, strategy_func, params,
                              fee_rate=total_fee)
        s = r['stats']
        if s:
            if base_return is None:
                base_return = s['net_return_pct']
            degradation = s['net_return_pct'] - base_return if base_return else 0
            print(f"  {slip*100:>9.2f}% {s['n_trades']:>7d} {s['win_rate']:>6.1f}% "
                  f"{s['profit_factor']:>5.2f} {s['net_return_pct']:>+8.1f}% "
                  f"${s['avg_daily_pnl']:>+7.0f} {degradation:>+8.1f}%")


# ── Regime Analysis ──────────────────────────────────────────────────────────

def regime_analysis(coin_data_processed, strategy_func, params):
    """Analyze performance in different market regimes."""
    # Get BTC data to classify regimes
    btc_df = coin_data_processed.get('BTC-USD')
    if btc_df is None:
        print("  No BTC data for regime classification")
        return

    # Classify each day as trending or ranging using daily ATR percentile
    btc_daily = btc_df.copy()
    btc_daily['date'] = btc_daily['time'].dt.date

    daily_agg = btc_daily.groupby('date').agg({
        'high': 'max', 'low': 'min', 'close': 'last', 'open': 'first'
    })
    daily_agg['daily_range'] = (daily_agg['high'] - daily_agg['low']) / daily_agg['close'] * 100
    daily_agg['daily_return'] = daily_agg['close'].pct_change() * 100

    # Trending: |daily return| > 2% or daily range > 4%
    # Ranging: everything else
    daily_agg['regime'] = 'ranging'
    daily_agg.loc[(daily_agg['daily_return'].abs() > 2) | (daily_agg['daily_range'] > 4), 'regime'] = 'trending'

    regime_map = daily_agg['regime'].to_dict()

    # Run full backtest and classify trades by regime
    result = backtest_strategy(coin_data_processed, strategy_func, params)
    trades = result['trades']

    if not trades:
        print("  No trades for regime analysis")
        return

    trending_trades = []
    ranging_trades = []

    for t in trades:
        d = t['entry_time'].date() if hasattr(t['entry_time'], 'date') else t['entry_time']
        regime = regime_map.get(d, 'ranging')
        if regime == 'trending':
            trending_trades.append(t)
        else:
            ranging_trades.append(t)

    trending_days = sum(1 for v in regime_map.values() if v == 'trending')
    ranging_days = sum(1 for v in regime_map.values() if v == 'ranging')

    print(f"\n  Market Regime Analysis")
    print(f"  {'─'*60}")
    print(f"  Total days: {len(regime_map)} (Trending: {trending_days}, Ranging: {ranging_days})")

    for label, t_list in [('TRENDING', trending_trades), ('RANGING', ranging_trades)]:
        if not t_list:
            print(f"\n  {label}: No trades")
            continue
        winners = [t for t in t_list if t['net_pnl_dollar'] > 0]
        wr = len(winners) / len(t_list) * 100
        total_pnl = sum(t['net_pnl_dollar'] for t in t_list)
        gross_win = sum(t['net_pnl_dollar'] for t in winners)
        gross_loss = abs(sum(t['net_pnl_dollar'] for t in t_list if t['net_pnl_dollar'] <= 0))
        pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

        print(f"\n  {label}: {len(t_list)} trades, {wr:.1f}% WR, PF {pf:.2f}, "
              f"P&L ${total_pnl:+,.0f}")
        print(f"    Longs: {len([t for t in t_list if t['side'] == 'long'])}, "
              f"Shorts: {len([t for t in t_list if t['side'] == 'short'])}")


# ── Look-Ahead Bias Audit ───────────────────────────────────────────────────

def audit_look_ahead_bias(coin_data_processed):
    """Verify that daily context in hourly data uses prior-day-only values.

    The add_daily_context function maps daily EMA/SMA/trend to hourly bars
    by date. This is valid ONLY if the daily values are computed from the
    PREVIOUS day's close (not the current day's close, which isn't known
    during the first hours of the day).
    """
    print(f"\n  Look-Ahead Bias Audit")
    print(f"  {'─'*60}")

    # Check: does d_trend use today's close or yesterday's?
    # In add_daily_context, daily['d_ema_21'] is computed on daily close,
    # then mapped to ALL hourly bars of that same day.
    # This means at hour 0 of day X, we're using day X's EMA which includes
    # day X's close — that's look-ahead bias!

    # Test: compare d_trend at first hour vs last hour of same day
    issues = 0
    checked = 0

    for symbol in ['BTC-USD', 'ETH-USD', 'SOL-USD']:
        df = coin_data_processed.get(symbol)
        if df is None:
            continue

        df_test = df.copy()
        df_test['date'] = df_test['time'].dt.date

        for date, group in df_test.groupby('date'):
            if len(group) < 10:
                continue
            checked += 1
            # d_trend should be same for all hours in the day
            trends = group['d_trend'].dropna().unique()
            if len(trends) > 1:
                issues += 1

    if issues > 0:
        print(f"  WARNING: {issues}/{checked} days have inconsistent d_trend within day")
    else:
        print(f"  d_trend consistency: OK ({checked} days checked)")

    # The real issue: d_trend for day X uses day X close
    # Fix: we need to shift daily indicators by 1 day
    print(f"\n  CRITICAL: Current implementation uses SAME-DAY daily close for d_trend.")
    print(f"  This is look-ahead bias — at 01:00 UTC, the daily close is unknown.")
    print(f"  FIX: Shift daily indicators by 1 day (use YESTERDAY's trend for today).")
    print(f"  Re-running with shifted daily context...")

    return True  # indicates bias was found


def fix_daily_context(df_1h):
    """Fixed version of add_daily_context that uses PRIOR day's values."""
    df = df_1h.copy()
    df['date'] = df['time'].dt.date

    # Daily aggregates
    daily = df.groupby('date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).reset_index()

    # Daily EMAs
    daily['d_ema_21'] = daily['close'].ewm(span=21, adjust=False).mean()
    daily['d_sma_200'] = daily['close'].rolling(200).mean()
    daily['d_trend'] = (daily['close'] > daily['d_ema_21']).astype(int)

    # Daily ATR
    d_hl = daily['high'] - daily['low']
    d_hc = (daily['high'] - daily['close'].shift(1)).abs()
    d_lc = (daily['low'] - daily['close'].shift(1)).abs()
    d_tr = pd.concat([d_hl, d_hc, d_lc], axis=1).max(axis=1)
    daily['d_atr'] = d_tr.rolling(14).mean()

    # SHIFT BY 1 DAY — use prior day's values for today
    daily['d_ema_21_shifted'] = daily['d_ema_21'].shift(1)
    daily['d_sma_200_shifted'] = daily['d_sma_200'].shift(1)
    daily['d_trend_shifted'] = daily['d_trend'].shift(1)
    daily['d_atr_shifted'] = daily['d_atr'].shift(1)

    # Map shifted values back to hourly
    daily_map = daily.set_index('date')[[
        'd_ema_21_shifted', 'd_sma_200_shifted', 'd_trend_shifted', 'd_atr_shifted'
    ]].to_dict('index')

    df['d_ema_21'] = df['date'].map(lambda d: daily_map.get(d, {}).get('d_ema_21_shifted'))
    df['d_sma_200'] = df['date'].map(lambda d: daily_map.get(d, {}).get('d_sma_200_shifted'))
    df['d_trend'] = df['date'].map(lambda d: daily_map.get(d, {}).get('d_trend_shifted', 0))
    df['d_atr'] = df['date'].map(lambda d: daily_map.get(d, {}).get('d_atr_shifted'))

    return df


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print_section("INTRADAY WALK-FORWARD VALIDATION")
    print(f"  Validating: Multi-TF Momentum + Intraday Donchian")
    print(f"  Capital: ${STARTING_CAPITAL:,}  |  Leverage: {LEVERAGE}x  |  Fee: {FUTURES_FEE*100:.2f}%/side")

    # ── Load data ─────────────────────────────────────────────
    print_section("LOADING DATA")
    coin_data = fetch_all_intraday(timeframe='1h', months=6)

    # ── Compute indicators ────────────────────────────────────
    print_section("COMPUTING INDICATORS (with look-ahead fix)")

    # First: compute with look-ahead bias (original)
    coin_data_biased = {}
    for symbol, df in coin_data.items():
        df = add_indicators(df)
        df = add_daily_context(df)
        coin_data_biased[symbol] = df

    # Second: compute with fixed daily context (no look-ahead)
    coin_data_fixed = {}
    for symbol, df in coin_data.items():
        print(f"  {symbol}: indicators + fixed daily context...", end='')
        df = add_indicators(df)
        df = fix_daily_context(df)
        coin_data_fixed[symbol] = df
        print(f" done")

    # ── Look-Ahead Bias Audit ─────────────────────────────────
    print_section("LOOK-AHEAD BIAS AUDIT")
    audit_look_ahead_bias(coin_data_biased)

    # Compare biased vs fixed results for Multi-TF
    mtf_params = {'vol_threshold': 1.2, 'atr_stop': 2.0, 'atr_target': 3.0, 'max_hold': 24}
    dc_params = {'dc_period': 48, 'atr_stop': 2.0, 'atr_target': 3.0,
                 'vol_threshold': 1.5, 'max_hold': 24}

    print(f"\n  Multi-TF Momentum: Biased vs Fixed comparison")
    print(f"  {'─'*70}")

    r_biased = backtest_strategy(coin_data_biased, strategy_mtf_momentum, mtf_params)
    r_fixed = backtest_strategy(coin_data_fixed, strategy_mtf_momentum, mtf_params)

    for label, r in [('WITH look-ahead (original)', r_biased), ('WITHOUT look-ahead (fixed)', r_fixed)]:
        s = r['stats']
        if s:
            print(f"  {label}:")
            print(f"    Trades: {s['n_trades']}  |  WR: {s['win_rate']:.1f}%  |  PF: {s['profit_factor']:.2f}  |  "
                  f"Return: {s['net_return_pct']:+.1f}%  |  DD: {s['max_drawdown']:.1f}%  |  $/day: ${s['avg_daily_pnl']:+.0f}")

    # Use FIXED data for all remaining tests
    coin_data_processed = coin_data_fixed

    # ── Walk-Forward: Multi-TF Momentum ───────────────────────
    print_section("WALK-FORWARD: Multi-TF Momentum")
    mtf_wf = walk_forward_backtest(coin_data_processed, strategy_mtf_momentum, mtf_params)
    mtf_oos_ret, mtf_oos_pf = print_walk_forward_results("Multi-TF Momentum", mtf_wf)

    # ── Walk-Forward: Intraday Donchian ───────────────────────
    print_section("WALK-FORWARD: Intraday Donchian (48h)")
    dc_wf = walk_forward_backtest(coin_data_processed, strategy_intraday_donchian, dc_params)
    dc_oos_ret, dc_oos_pf = print_walk_forward_results("Intraday Donchian", dc_wf)

    # ── Also test the best Donchian variant from param sweep ──
    dc_best_params = {'dc_period': 72, 'atr_stop': 2.0, 'atr_target': 3.0,
                      'vol_threshold': 1.5, 'max_hold': 36}
    print_section("WALK-FORWARD: Intraday Donchian (72h best)")
    dc72_wf = walk_forward_backtest(coin_data_processed, strategy_intraday_donchian, dc_best_params)
    dc72_oos_ret, dc72_oos_pf = print_walk_forward_results("Intraday Donchian 72h", dc72_wf)

    # ── Parameter Robustness: Multi-TF Momentum ──────────────
    print_section("PARAMETER ROBUSTNESS: Multi-TF Momentum")
    mtf_grid = {
        'vol_threshold': [1.0, 1.1, 1.2, 1.3, 1.5, 1.8],
        'atr_stop': [1.0, 1.5, 2.0, 2.5, 3.0],
        'atr_target': [2.0, 2.5, 3.0, 3.5, 4.0],
        'max_hold': [8, 12, 16, 24, 36, 48],
    }
    mtf_robust = parameter_robustness(coin_data_processed, strategy_mtf_momentum,
                                       mtf_params, mtf_grid)
    print_robustness_results("Multi-TF Momentum", mtf_robust)

    # ── Parameter Robustness: Intraday Donchian ───────────────
    print_section("PARAMETER ROBUSTNESS: Intraday Donchian")
    dc_grid = {
        'dc_period': [24, 36, 48, 60, 72, 96],
        'atr_stop': [1.0, 1.5, 2.0, 2.5, 3.0],
        'atr_target': [2.0, 2.5, 3.0, 4.0, 5.0],
        'vol_threshold': [1.0, 1.2, 1.3, 1.5, 1.8, 2.0],
        'max_hold': [12, 16, 24, 36, 48],
    }
    dc_robust = parameter_robustness(coin_data_processed, strategy_intraday_donchian,
                                      dc_params, dc_grid)
    print_robustness_results("Intraday Donchian", dc_robust)

    # ── Slippage Stress Test ──────────────────────────────────
    print_section("SLIPPAGE STRESS TEST")
    print(f"\n  Multi-TF Momentum:")
    slippage_stress_test(coin_data_processed, strategy_mtf_momentum, mtf_params)
    print(f"\n  Intraday Donchian:")
    slippage_stress_test(coin_data_processed, strategy_intraday_donchian, dc_params)

    # ── Regime Analysis ───────────────────────────────────────
    print_section("REGIME ANALYSIS")
    print(f"\n  Multi-TF Momentum:")
    regime_analysis(coin_data_processed, strategy_mtf_momentum, mtf_params)
    print(f"\n  Intraday Donchian:")
    regime_analysis(coin_data_processed, strategy_intraday_donchian, dc_params)

    # ── Combined Strategy ─────────────────────────────────────
    print_section("COMBINED: Multi-TF + Donchian")
    # Run both strategies and merge trade lists
    r_mtf = backtest_strategy(coin_data_processed, strategy_mtf_momentum, mtf_params)
    r_dc = backtest_strategy(coin_data_processed, strategy_intraday_donchian, dc_params)

    mtf_s = r_mtf['stats']
    dc_s = r_dc['stats']

    if mtf_s and dc_s:
        combined_daily = mtf_s['avg_daily_pnl'] + dc_s['avg_daily_pnl']
        combined_return = mtf_s['net_return_pct'] + dc_s['net_return_pct']

        print(f"\n  Individual results (on $10K each):")
        print(f"    Multi-TF:  ${mtf_s['avg_daily_pnl']:+.0f}/day, {mtf_s['net_return_pct']:+.1f}% return")
        print(f"    Donchian:  ${dc_s['avg_daily_pnl']:+.0f}/day, {dc_s['net_return_pct']:+.1f}% return")
        print(f"\n  If running both on shared $10K (50/50 allocation):")
        est_combined = (mtf_s['avg_daily_pnl'] + dc_s['avg_daily_pnl']) / 2
        print(f"    Est. daily P&L: ${est_combined:+.0f}")
        print(f"\n  If running both on separate capital ($5K each):")
        print(f"    Est. daily P&L: ${mtf_s['avg_daily_pnl']/2 + dc_s['avg_daily_pnl']/2:+.0f}")

    # ── Final Summary ─────────────────────────────────────────
    print_section("FINAL VERDICT")

    print(f"\n  Strategy Performance (fixed, no look-ahead bias):")
    print(f"  {'─'*70}")

    for name, func, params in [
        ('Multi-TF Momentum', strategy_mtf_momentum, mtf_params),
        ('Intraday Donchian 48h', strategy_intraday_donchian, dc_params),
        ('Intraday Donchian 72h', strategy_intraday_donchian, dc_best_params),
    ]:
        r = backtest_strategy(coin_data_processed, func, params)
        s = r['stats']
        if s:
            passed = (s['win_rate'] >= 52 and s['profit_factor'] >= 1.5) or \
                     (s['profit_factor'] >= 1.8 and s['n_trades'] >= 20)
            v = "PASS" if passed else "FAIL"
            print(f"  [{v}] {name}: {s['n_trades']} trades, {s['win_rate']:.1f}% WR, "
                  f"PF {s['profit_factor']:.2f}, {s['net_return_pct']:+.1f}%, "
                  f"DD {s['max_drawdown']:.1f}%, ${s['avg_daily_pnl']:+.0f}/day")

    print(f"\n  Path to $500/day depends on walk-forward OOS results above.")
    print(f"  If OOS PF >= 1.5 and >60% windows profitable → viable for live testing.")

    print(f"\n{'='*80}")
    print(f"  VALIDATION COMPLETE")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
