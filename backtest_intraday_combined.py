"""
Combined Intraday System: Trending + Ranging Strategies
=========================================================
Runs TWO complementary strategies on the same capital:
1. MTF Momentum (ADX>=25) — trades only in TRENDING markets
2. Mean Reversion (ADX<25) — trades only in RANGING markets

This ensures the bot is always active regardless of regime,
with the appropriate strategy for the conditions.

Also tests higher leverage (2.5x, 3x) impact.

Usage:
    PYTHONIOENCODING=utf-8 venv/Scripts/python backtest_intraday_combined.py
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
    add_indicators, compute_stats,
    STARTING_CAPITAL, FUTURES_FEE, MAX_POSITIONS, RISK_PER_TRADE,
    LEVERAGE, TRADE_COINS,
)
from backtest_intraday_wf import fix_daily_context
from backtest_intraday_v2 import (
    compute_daily_regime, add_regime_to_hourly, strategy_mtf_regime_filtered,
    print_section,
)


# ── Mean Reversion Strategy (for RANGING markets) ───────────────────────────

def strategy_ranging_mean_reversion(df, params):
    """Mean reversion for ranging/low-ADX markets.

    Only trades when ADX < threshold (ranging). Uses:
    - RSI extremes (oversold/overbought) for entry
    - Bollinger Band touch/pierce for confirmation
    - Quick exit at BB midline or EMA(21)
    - Tight stops (1.5x ATR)
    - Short max hold (8-12h) — mean reversion should be fast

    Long: ADX < threshold + RSI < rsi_entry + close < BB lower + vol > threshold
    Short: ADX < threshold + RSI > (100-rsi_entry) + close > BB upper + vol > threshold
    """
    adx_max = params.get('adx_max', 25)  # Only in ranging markets
    rsi_entry = params.get('rsi_entry', 30)  # RSI oversold for longs
    vol_thresh = params.get('vol_threshold', 1.2)
    atr_stop = params.get('atr_stop', 1.5)
    max_hold = params.get('max_hold', 10)
    use_bb = params.get('use_bb', True)  # Require BB confirmation

    signals = []
    for i in range(52, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]

        if pd.isna(row['atr']) or row['atr'] <= 0:
            continue
        if pd.isna(row.get('d_adx')):
            continue

        d_adx = row.get('d_adx', 30)
        if pd.isna(d_adx):
            d_adx = 30

        # Only trade in RANGING markets
        if d_adx >= adx_max:
            continue

        # Long: RSI oversold + bouncing + optional BB lower touch
        bb_long_ok = (not use_bb) or (row['close'] <= row.get('bb_lower', row['close'] + 1))
        if (row['rsi'] < rsi_entry and
            row['rsi'] > prev['rsi'] and  # bouncing
            row['vol_ratio'] > vol_thresh and
            bb_long_ok):

            # Target: BB midline or EMA(21), whichever is closer
            target = row.get('bb_mid', row['ema_21'])
            if pd.isna(target):
                target = row['ema_21']

            # Only enter if target is above entry (there's room)
            if target > row['close'] * 1.003:  # at least 0.3% room
                signals.append({
                    'idx': i, 'time': row['time'], 'side': 'long',
                    'entry': row['close'],
                    'stop': row['close'] - atr_stop * row['atr'],
                    'target': target,
                    'atr': row['atr'], 'max_hold': max_hold,
                })

        # Short: RSI overbought + reversing + optional BB upper touch
        rsi_short = 100 - rsi_entry
        bb_short_ok = (not use_bb) or (row['close'] >= row.get('bb_upper', row['close'] - 1))
        if (row['rsi'] > rsi_short and
            row['rsi'] < prev['rsi'] and  # reversing
            row['vol_ratio'] > vol_thresh and
            bb_short_ok):

            target = row.get('bb_mid', row['ema_21'])
            if pd.isna(target):
                target = row['ema_21']

            if target < row['close'] * 0.997:
                signals.append({
                    'idx': i, 'time': row['time'], 'side': 'short',
                    'entry': row['close'],
                    'stop': row['close'] + atr_stop * row['atr'],
                    'target': target,
                    'atr': row['atr'], 'max_hold': max_hold,
                })

    return signals


def strategy_ranging_rsi_bounce(df, params):
    """Simpler RSI bounce for ranging markets (no BB requirement).

    Long: ADX < threshold + RSI < 35 + RSI rising + close near EMA(50) support
    Short: ADX < threshold + RSI > 65 + RSI falling + close near EMA(50) resistance
    Target: 1.5x ATR (small, quick)
    """
    adx_max = params.get('adx_max', 25)
    rsi_long = params.get('rsi_long', 35)
    rsi_short = params.get('rsi_short', 65)
    vol_thresh = params.get('vol_threshold', 1.0)
    atr_stop = params.get('atr_stop', 1.0)
    atr_target = params.get('atr_target', 1.5)
    max_hold = params.get('max_hold', 8)

    signals = []
    for i in range(52, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]

        if pd.isna(row['atr']) or row['atr'] <= 0:
            continue

        d_adx = row.get('d_adx', 30)
        if pd.isna(d_adx):
            d_adx = 30
        if d_adx >= adx_max:
            continue

        # Long: oversold bounce
        if (row['rsi'] < rsi_long and
            row['rsi'] > prev['rsi'] and
            row['vol_ratio'] > vol_thresh):
            signals.append({
                'idx': i, 'time': row['time'], 'side': 'long',
                'entry': row['close'],
                'stop': row['close'] - atr_stop * row['atr'],
                'target': row['close'] + atr_target * row['atr'],
                'atr': row['atr'], 'max_hold': max_hold,
            })

        # Short: overbought reversal
        if (row['rsi'] > rsi_short and
            row['rsi'] < prev['rsi'] and
            row['vol_ratio'] > vol_thresh):
            signals.append({
                'idx': i, 'time': row['time'], 'side': 'short',
                'entry': row['close'],
                'stop': row['close'] + atr_stop * row['atr'],
                'target': row['close'] - atr_target * row['atr'],
                'atr': row['atr'], 'max_hold': max_hold,
            })

    return signals


# ── Combined Portfolio Backtester ────────────────────────────────────────────

def backtest_combined(coin_data, trending_func, trending_params,
                       ranging_func, ranging_params,
                       starting_capital=STARTING_CAPITAL, max_positions=MAX_POSITIONS,
                       risk_per_trade=RISK_PER_TRADE, leverage=LEVERAGE,
                       fee_rate=FUTURES_FEE, max_daily_loss=0.03):
    """Run two strategies on the same capital pool.

    Each strategy generates signals independently, but they share:
    - The same equity pool
    - The same max position limit
    - The same daily loss circuit breaker
    """
    equity = starting_capital
    peak_equity = starting_capital
    positions = {}  # {symbol: position_dict}
    trades = []
    equity_curve = []

    # Gather signals from both strategies
    all_signals = []
    for symbol, df in coin_data.items():
        if symbol not in TRADE_COINS:
            continue
        # Trending signals
        t_sigs = trending_func(df, trending_params)
        for s in t_sigs:
            s['symbol'] = symbol
            s['strategy'] = 'trending'
        all_signals.extend(t_sigs)

        # Ranging signals
        r_sigs = ranging_func(df, ranging_params)
        for s in r_sigs:
            s['symbol'] = symbol
            s['strategy'] = 'ranging'
        all_signals.extend(r_sigs)

    all_signals.sort(key=lambda s: s['time'])

    if not all_signals:
        return {'trades': [], 'equity_curve': [], 'stats': None, 'strategy_breakdown': {}}

    # Price lookups
    price_lookup = {}
    for symbol, df in coin_data.items():
        lookup = {}
        for _, row in df.iterrows():
            lookup[row['time']] = row
        price_lookup[symbol] = lookup

    # All timestamps
    all_times = set()
    for symbol, df in coin_data.items():
        if symbol in TRADE_COINS:
            for t in df['time']:
                all_times.add(t)
    all_times = sorted(all_times)

    signal_by_time = defaultdict(list)
    for s in all_signals:
        signal_by_time[s['time']].append(s)

    current_day = None
    day_pnl = 0.0
    day_locked = False

    for t in all_times:
        day = t.date() if hasattr(t, 'date') else t
        if day != current_day:
            current_day = day
            day_pnl = 0.0
            day_locked = False

        if day_locked:
            continue

        # Check exits
        to_close = []
        for sym, pos in list(positions.items()):
            row = price_lookup.get(sym, {}).get(t)
            if row is None:
                continue

            close_price = row['close']
            high = row['high']
            low = row['low']
            pos['hold_candles'] += 1

            exit_reason = None
            exit_price = close_price

            if pos['side'] == 'long':
                if low <= pos['stop']:
                    exit_reason = 'stop'
                    exit_price = pos['stop']
                elif high >= pos['target']:
                    exit_reason = 'target'
                    exit_price = pos['target']
                elif close_price > pos.get('high_watermark', pos['entry']):
                    pos['high_watermark'] = close_price
                    new_stop = close_price - 1.5 * pos['atr']
                    if new_stop > pos['stop']:
                        pos['stop'] = new_stop
                if not exit_reason and pos['hold_candles'] >= pos['max_hold']:
                    exit_reason = 'max_hold'
                    exit_price = close_price
            elif pos['side'] == 'short':
                if high >= pos['stop']:
                    exit_reason = 'stop'
                    exit_price = pos['stop']
                elif low <= pos['target']:
                    exit_reason = 'target'
                    exit_price = pos['target']
                elif close_price < pos.get('low_watermark', pos['entry']):
                    pos['low_watermark'] = close_price
                    new_stop = close_price + 1.5 * pos['atr']
                    if new_stop < pos['stop']:
                        pos['stop'] = new_stop
                if not exit_reason and pos['hold_candles'] >= pos['max_hold']:
                    exit_reason = 'max_hold'
                    exit_price = close_price

            if exit_reason:
                to_close.append((sym, pos, exit_price, exit_reason))

        for sym, pos, exit_price, exit_reason in to_close:
            if pos['side'] == 'long':
                gross_pnl = (exit_price - pos['entry']) / pos['entry']
            else:
                gross_pnl = (pos['entry'] - exit_price) / pos['entry']

            leveraged_pnl = gross_pnl * leverage
            net_pnl_pct = leveraged_pnl - 2 * fee_rate
            net_pnl_dollar = pos['size'] * net_pnl_pct
            equity += net_pnl_dollar
            day_pnl += net_pnl_dollar

            if equity > peak_equity:
                peak_equity = equity

            trades.append({
                'symbol': sym, 'side': pos['side'],
                'strategy': pos['strategy'],
                'entry_time': pos['entry_time'], 'exit_time': t,
                'entry_price': pos['entry'], 'exit_price': exit_price,
                'hold_candles': pos['hold_candles'],
                'gross_pnl_pct': gross_pnl * 100,
                'net_pnl_pct': net_pnl_pct * 100,
                'net_pnl_dollar': net_pnl_dollar,
                'exit_reason': exit_reason, 'equity_after': equity,
            })
            del positions[sym]

            if day_pnl / starting_capital < -max_daily_loss:
                day_locked = True
                break

        if day_locked:
            equity_curve.append({'time': t, 'equity': equity})
            continue

        # Check entries
        if t in signal_by_time:
            for sig in signal_by_time[t]:
                sym = sig['symbol']
                if sym in positions:
                    continue
                if len(positions) >= max_positions:
                    break
                if equity < starting_capital * 0.5:
                    break

                stop_distance = abs(sig['entry'] - sig['stop']) / sig['entry']
                if stop_distance <= 0:
                    continue

                risk_amount = equity * risk_per_trade
                position_size = risk_amount / stop_distance
                position_size = min(position_size, equity * 0.95 / leverage)

                positions[sym] = {
                    'side': sig['side'], 'strategy': sig['strategy'],
                    'entry': sig['entry'], 'entry_time': sig['time'],
                    'stop': sig['stop'], 'target': sig['target'],
                    'atr': sig['atr'], 'size': position_size,
                    'hold_candles': 0, 'max_hold': sig['max_hold'],
                    'high_watermark': sig['entry'], 'low_watermark': sig['entry'],
                }
                entry_fee = position_size * fee_rate
                equity -= entry_fee

        equity_curve.append({'time': t, 'equity': equity})

    # Close remaining
    for sym, pos in list(positions.items()):
        last_df = coin_data[sym]
        exit_price = last_df.iloc[-1]['close']
        if pos['side'] == 'long':
            gross_pnl = (exit_price - pos['entry']) / pos['entry']
        else:
            gross_pnl = (pos['entry'] - exit_price) / pos['entry']
        leveraged_pnl = gross_pnl * leverage
        net_pnl_pct = leveraged_pnl - 2 * fee_rate
        net_pnl_dollar = pos['size'] * net_pnl_pct
        equity += net_pnl_dollar
        trades.append({
            'symbol': sym, 'side': pos['side'], 'strategy': pos['strategy'],
            'entry_time': pos['entry_time'], 'exit_time': last_df.iloc[-1]['time'],
            'entry_price': pos['entry'], 'exit_price': exit_price,
            'hold_candles': pos['hold_candles'],
            'gross_pnl_pct': gross_pnl * 100, 'net_pnl_pct': net_pnl_pct * 100,
            'net_pnl_dollar': net_pnl_dollar,
            'exit_reason': 'end_of_data', 'equity_after': equity,
        })

    # Strategy breakdown
    strategy_breakdown = {}
    for strat_name in ['trending', 'ranging']:
        strat_trades = [t for t in trades if t['strategy'] == strat_name]
        if strat_trades:
            strategy_breakdown[strat_name] = compute_stats(strat_trades, equity, starting_capital)
            strategy_breakdown[strat_name]['trades'] = strat_trades

    stats = compute_stats(trades, equity, starting_capital)
    return {
        'trades': trades,
        'equity_curve': equity_curve,
        'stats': stats,
        'strategy_breakdown': strategy_breakdown,
    }


# ── Walk-Forward for Combined ────────────────────────────────────────────────

def walk_forward_combined(coin_data, trending_func, trending_params,
                           ranging_func, ranging_params,
                           train_months=4, test_months=2, step_months=1,
                           leverage=LEVERAGE):
    """Walk-forward for the combined strategy."""
    all_starts = []
    all_ends = []
    for symbol, df in coin_data.items():
        if symbol in TRADE_COINS:
            all_starts.append(df['time'].min())
            all_ends.append(df['time'].max())

    data_start = max(all_starts)
    data_end = min(all_ends)
    total_days = (data_end - data_start).days

    train_days = train_months * 30
    test_days = test_months * 30
    step_days = step_months * 30

    windows = []
    offset = 0
    while offset + train_days + test_days <= total_days:
        w_start = data_start + timedelta(days=offset)
        train_end = w_start + timedelta(days=train_days)
        test_end = train_end + timedelta(days=test_days)
        windows.append({
            'train_start': w_start, 'train_end': train_end,
            'test_start': train_end, 'test_end': min(test_end, data_end),
        })
        offset += step_days

    print(f"  Walk-forward: {len(windows)} windows ({train_months}mo/{test_months}mo/{step_months}mo)")

    results = []
    for i, w in enumerate(windows):
        test_data = {}
        for symbol, df in coin_data.items():
            if symbol not in TRADE_COINS:
                continue
            test_df = df[(df['time'] >= w['test_start']) & (df['time'] < w['test_end'])].copy()
            if len(test_df) > 50:
                test_data[symbol] = test_df.reset_index(drop=True)

        if not test_data:
            continue

        test_result = backtest_combined(test_data, trending_func, trending_params,
                                         ranging_func, ranging_params, leverage=leverage)
        results.append({
            'window': i + 1,
            'test_period': f"{w['test_start'].date()} → {w['test_end'].date()}",
            'stats': test_result['stats'],
            'breakdown': test_result['strategy_breakdown'],
        })

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print_section("COMBINED INTRADAY: TRENDING + RANGING STRATEGIES")
    print(f"  Capital: ${STARTING_CAPITAL:,}  |  Fee: {FUTURES_FEE*100:.2f}%/side")

    # Load data
    print_section("LOADING DATA")
    coin_data_raw = fetch_all_intraday(timeframe='1h', months=12)
    regime_map = compute_daily_regime(coin_data_raw)
    coin_data = {}
    for symbol, df in coin_data_raw.items():
        print(f"  {symbol}...", end='', flush=True)
        df = add_indicators(df)
        df = fix_daily_context(df)
        df = add_regime_to_hourly(df, regime_map)
        coin_data[symbol] = df
        print(f" done")

    # ═══════════════════════════════════════════════════════════
    # SECTION 1: Ranging Strategy Sweep
    # ═══════════════════════════════════════════════════════════
    print_section("SECTION 1: RANGING STRATEGY SWEEP")

    from backtest_intraday import backtest_strategy

    ranging_configs = [
        ('MR: RSI<30 + BB, hold 10', strategy_ranging_mean_reversion,
         {'adx_max': 25, 'rsi_entry': 30, 'vol_threshold': 1.2, 'atr_stop': 1.5, 'max_hold': 10, 'use_bb': True}),
        ('MR: RSI<35 + BB, hold 10', strategy_ranging_mean_reversion,
         {'adx_max': 25, 'rsi_entry': 35, 'vol_threshold': 1.2, 'atr_stop': 1.5, 'max_hold': 10, 'use_bb': True}),
        ('MR: RSI<30 no BB, hold 8', strategy_ranging_mean_reversion,
         {'adx_max': 25, 'rsi_entry': 30, 'vol_threshold': 1.2, 'atr_stop': 1.5, 'max_hold': 8, 'use_bb': False}),
        ('MR: RSI<35 no BB, hold 12', strategy_ranging_mean_reversion,
         {'adx_max': 25, 'rsi_entry': 35, 'vol_threshold': 1.0, 'atr_stop': 1.5, 'max_hold': 12, 'use_bb': False}),
        ('RSI Bounce: 35/65, 1.5T', strategy_ranging_rsi_bounce,
         {'adx_max': 25, 'rsi_long': 35, 'rsi_short': 65, 'vol_threshold': 1.0,
          'atr_stop': 1.0, 'atr_target': 1.5, 'max_hold': 8}),
        ('RSI Bounce: 30/70, 2.0T', strategy_ranging_rsi_bounce,
         {'adx_max': 25, 'rsi_long': 30, 'rsi_short': 70, 'vol_threshold': 1.0,
          'atr_stop': 1.0, 'atr_target': 2.0, 'max_hold': 10}),
        ('RSI Bounce: 35/65, 1.0T', strategy_ranging_rsi_bounce,
         {'adx_max': 25, 'rsi_long': 35, 'rsi_short': 65, 'vol_threshold': 1.0,
          'atr_stop': 0.8, 'atr_target': 1.0, 'max_hold': 6}),
        ('RSI Bounce: 30/70, 1.5T tight', strategy_ranging_rsi_bounce,
         {'adx_max': 25, 'rsi_long': 30, 'rsi_short': 70, 'vol_threshold': 1.2,
          'atr_stop': 1.0, 'atr_target': 1.5, 'max_hold': 8}),
        ('RSI Bounce: ADX<22', strategy_ranging_rsi_bounce,
         {'adx_max': 22, 'rsi_long': 35, 'rsi_short': 65, 'vol_threshold': 1.0,
          'atr_stop': 1.0, 'atr_target': 1.5, 'max_hold': 8}),
        ('RSI Bounce: ADX<30', strategy_ranging_rsi_bounce,
         {'adx_max': 30, 'rsi_long': 35, 'rsi_short': 65, 'vol_threshold': 1.0,
          'atr_stop': 1.0, 'atr_target': 1.5, 'max_hold': 8}),
    ]

    print(f"\n  {'Strategy':<35s} {'Trades':>6s} {'T/day':>5s} {'WR':>7s} {'PF':>6s} "
          f"{'Return':>9s} {'DD':>7s} {'$/day':>8s}")
    print(f"  {'─'*82}")

    best_ranging = None
    best_ranging_params = None
    best_ranging_func = None
    best_ranging_pf = 0

    for name, func, params in ranging_configs:
        r = backtest_strategy(coin_data, func, params)
        s = r['stats']
        if s and s['n_trades'] > 0:
            marker = ""
            if s['profit_factor'] > best_ranging_pf and s['n_trades'] >= 20:
                best_ranging_pf = s['profit_factor']
                best_ranging = name
                best_ranging_params = params
                best_ranging_func = func
                marker = " <--"
            print(f"  {name:<35s} {s['n_trades']:>6d} {s['trades_per_day']:>4.1f} "
                  f"{s['win_rate']:>6.1f}% {s['profit_factor']:>5.2f} "
                  f"{s['net_return_pct']:>+8.1f}% {s['max_drawdown']:>6.1f}% "
                  f"${s['avg_daily_pnl']:>+7.0f}{marker}")
        else:
            print(f"  {name:<35s}   — no trades —")

    print(f"\n  Best ranging: {best_ranging}")

    if not best_ranging_func:
        print("  No viable ranging strategy found")
        return

    # ═══════════════════════════════════════════════════════════
    # SECTION 2: Combined — Trending + Best Ranging
    # ═══════════════════════════════════════════════════════════
    print_section("SECTION 2: COMBINED STRATEGY")

    trending_params = {'vol_threshold': 1.2, 'atr_stop': 2.0, 'atr_target': 3.0,
                        'max_hold': 24, 'adx_threshold': 25}

    # Test at different leverage levels
    for lev in [2.0, 2.5, 3.0]:
        print(f"\n  --- {lev}x Leverage ---")
        r = backtest_combined(coin_data, strategy_mtf_regime_filtered, trending_params,
                               best_ranging_func, best_ranging_params, leverage=lev)
        s = r['stats']
        if s:
            print(f"  COMBINED: {s['n_trades']} trades, {s['trades_per_day']:.1f}/day, "
                  f"{s['win_rate']:.1f}% WR, PF {s['profit_factor']:.2f}")
            print(f"  Return: {s['net_return_pct']:+.1f}%  |  DD: {s['max_drawdown']:.1f}%  |  "
                  f"$/day: ${s['avg_daily_pnl']:+.0f}")

            # Breakdown
            for strat_name, strat_stats in r['strategy_breakdown'].items():
                if strat_stats:
                    st = strat_stats
                    print(f"    {strat_name:>10s}: {st['n_trades']} trades, {st['win_rate']:.1f}% WR, "
                          f"PF {st['profit_factor']:.2f}")

            # Monthly P&L
            monthly = defaultdict(float)
            for t in r['trades']:
                m = t['exit_time'].strftime('%Y-%m')
                monthly[m] += t['net_pnl_dollar']

            print(f"\n  Monthly P&L:")
            for m in sorted(monthly.keys()):
                val = monthly[m]
                bar = '+' * min(40, max(0, int(val / 200))) if val > 0 else '-' * min(40, max(0, int(-val / 200)))
                print(f"    {m}: ${val:>+8,.0f}  {bar}")

    # ═══════════════════════════════════════════════════════════
    # SECTION 3: Walk-Forward on Combined
    # ═══════════════════════════════════════════════════════════
    print_section("SECTION 3: WALK-FORWARD — COMBINED STRATEGY")

    for lev in [2.0, 2.5, 3.0]:
        print(f"\n  --- Combined @ {lev}x Leverage (4mo/2mo) ---")
        wf = walk_forward_combined(coin_data, strategy_mtf_regime_filtered, trending_params,
                                    best_ranging_func, best_ranging_params, leverage=lev)

        oos_returns = []
        oos_dailys = []
        print(f"  {'Window':<8s} {'Test Period':<26s} {'Trades':>7s} {'WR':>7s} {'PF':>6s} "
              f"{'Return':>9s} {'$/day':>8s}")
        print(f"  {'─'*70}")

        for w in wf:
            s = w['stats']
            if s and s['n_trades'] > 0:
                oos_returns.append(s['net_return_pct'])
                oos_dailys.append(s['avg_daily_pnl'])
                print(f"  W{w['window']:<7d} {w['test_period']:<26s} {s['n_trades']:>7d} "
                      f"{s['win_rate']:>6.1f}% {s['profit_factor']:>5.2f} "
                      f"{s['net_return_pct']:>+8.1f}% ${s['avg_daily_pnl']:>+7.0f}")

                # Show breakdown
                for sn, ss in w['breakdown'].items():
                    if ss:
                        print(f"         {sn:>10s}: {ss['n_trades']} tr, PF {ss['profit_factor']:.2f}")
            else:
                print(f"  W{w['window']:<7d} {w['test_period']:<26s}    — no trades —")

        if oos_returns:
            avg_ret = np.mean(oos_returns)
            avg_daily = np.mean(oos_dailys)
            profitable = sum(1 for r in oos_returns if r > 0)
            print(f"\n  OOS Summary @ {lev}x:")
            print(f"    Avg return: {avg_ret:+.1f}%  |  Avg $/day: ${avg_daily:+.0f}")
            print(f"    Profitable: {profitable}/{len(oos_returns)} ({profitable/len(oos_returns)*100:.0f}%)")
            print(f"    Range: {min(oos_returns):+.1f}% to {max(oos_returns):+.1f}%")

    # ═══════════════════════════════════════════════════════════
    # SECTION 4: Trending-only vs Combined comparison
    # ═══════════════════════════════════════════════════════════
    print_section("SECTION 4: TRENDING-ONLY vs COMBINED vs COMBINED+3x")

    configs = [
        ('Trending only (2x)', strategy_mtf_regime_filtered, trending_params, None, None, 2.0),
        ('Trending only (3x)', strategy_mtf_regime_filtered, trending_params, None, None, 3.0),
    ]

    # Combined at various leverage
    for lev in [2.0, 2.5, 3.0]:
        configs.append((f'Combined ({lev}x)', strategy_mtf_regime_filtered, trending_params,
                        best_ranging_func, best_ranging_params, lev))

    print(f"\n  {'Config':<30s} {'Trades':>6s} {'T/day':>5s} {'WR':>7s} {'PF':>6s} "
          f"{'Return':>9s} {'DD':>7s} {'$/day':>8s}")
    print(f"  {'─'*80}")

    for name, t_func, t_params, r_func, r_params, lev in configs:
        if r_func:
            r = backtest_combined(coin_data, t_func, t_params, r_func, r_params, leverage=lev)
        else:
            r = backtest_strategy(coin_data, t_func, t_params, leverage=lev)
        s = r['stats']
        if s:
            print(f"  {name:<30s} {s['n_trades']:>6d} {s['trades_per_day']:>4.1f} "
                  f"{s['win_rate']:>6.1f}% {s['profit_factor']:>5.2f} "
                  f"{s['net_return_pct']:>+8.1f}% {s['max_drawdown']:>6.1f}% "
                  f"${s['avg_daily_pnl']:>+7.0f}")

    # ═══════════════════════════════════════════════════════════
    # SECTION 5: Path to $500/day
    # ═══════════════════════════════════════════════════════════
    print_section("PATH TO $500/DAY — REALISTIC PROJECTIONS")

    # Use the best combined WF OOS result
    print(f"\n  Based on walk-forward OOS averages:")
    print(f"  (These are the REALISTIC numbers, not full-period backtests)")

    # We'll use the WF results computed above
    for lev in [2.0, 2.5, 3.0]:
        wf = walk_forward_combined(coin_data, strategy_mtf_regime_filtered, trending_params,
                                    best_ranging_func, best_ranging_params, leverage=lev)
        oos_dailys = [w['stats']['avg_daily_pnl'] for w in wf if w['stats'] and w['stats']['n_trades'] > 0]
        if oos_dailys:
            avg_daily = np.mean(oos_dailys)
            daily_rate = avg_daily / STARTING_CAPITAL

            if avg_daily > 0:
                eq = STARTING_CAPITAL
                days_30k = 0
                while eq < 30000 and days_30k < 500:
                    eq *= (1 + daily_rate)
                    days_30k += 1

                capital_500 = 500 / daily_rate if daily_rate > 0 else float('inf')
                eq2 = STARTING_CAPITAL
                days_target = 0
                while eq2 < capital_500 and days_target < 500:
                    eq2 *= (1 + daily_rate)
                    days_target += 1

                print(f"\n  {lev}x leverage:")
                print(f"    OOS avg daily: ${avg_daily:+.0f} ({daily_rate*100:.2f}%/day)")
                print(f"    $10K → $30K: {days_30k} days")
                print(f"    Capital for $500/day: ${capital_500:,.0f}")
                print(f"    $10K → ${capital_500/1000:.0f}K: {days_target} days")

    print(f"\n{'='*80}")
    print(f"  ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
