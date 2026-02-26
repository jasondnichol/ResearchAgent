"""5-Minute Enhanced Backtest — Intra-Hour Stop/TP Simulation

Extends the unified hourly backtest with 5-minute candle data to simulate
production's check_between_signals() behavior. Entries and indicator-based
exits still fire on hourly bars; price-based stops and take-profits are
checked on each 5-min bar between hourly bars.

Compares hourly-only vs 5-min enhanced results side by side.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import math
import time as time_module
import requests

from backtest_unified_4year import (
    STRICT_PARAMS, LOOSENED_PARAMS,
    fetch_and_cache_hourly_data,
    classify_regimes,
    calculate_all_indicators,
    backtest_unified,
    compute_stats,
    compute_strategy_stats,
)


# ============================================================================
# 5-MINUTE DATA FETCHING & CACHING
# ============================================================================

def fetch_and_cache_5min_data(cache_file='btc_4year_5min_cache.json', years=4, force_refresh=False):
    """Fetch and cache 4 years of 5-minute BTC data from Coinbase"""

    if not force_refresh and os.path.exists(cache_file):
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        age_days = (datetime.now() - file_time).days
        if age_days <= 7:
            print(f"5-min cache found ({age_days} days old), loading...")
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            df = pd.DataFrame(cache_data['data'])
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time').reset_index(drop=True)
            print(f"Loaded {len(df):,} five-minute candles")
            print(f"Range: {df['time'].min()} to {df['time'].max()}")
            return df

    coinbase_api = "https://api.exchange.coinbase.com"
    all_data = []
    end_time = datetime.utcnow()
    total_minutes = years * 365 * 24 * 60
    chunk_minutes = 300 * 5  # 300 candles * 5 min each = 1500 minutes per chunk
    num_chunks = math.ceil(total_minutes / chunk_minutes)

    print(f"Fetching {years} years of 5-min BTC data ({num_chunks} chunks, ~{num_chunks // 2}s)...")
    print(f"Estimated time: ~{num_chunks // 2 // 60} minutes")

    for i in range(num_chunks):
        chunk_end = end_time - timedelta(minutes=i * chunk_minutes)
        chunk_start = chunk_end - timedelta(minutes=chunk_minutes)

        earliest = end_time - timedelta(minutes=total_minutes)
        if chunk_start < earliest:
            chunk_start = earliest

        url = f"{coinbase_api}/products/BTC-USD/candles"
        params = {
            'start': chunk_start.isoformat(),
            'end': chunk_end.isoformat(),
            'granularity': 300  # 5 minutes
        }

        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    all_data.extend(data)
            if (i + 1) % 100 == 0:
                print(f"  Chunk {i+1}/{num_chunks} done... ({len(all_data):,} candles so far)")
            time_module.sleep(0.5)
        except Exception as e:
            print(f"  Chunk {i+1} error: {e}")
            time_module.sleep(1)

    if not all_data:
        print("ERROR: No 5-min data fetched")
        return None

    df = pd.DataFrame(all_data, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.sort_values('time').reset_index(drop=True)
    df = df.drop_duplicates(subset=['time'])

    print(f"Fetched {len(df):,} five-minute candles")
    print(f"Range: {df['time'].min()} to {df['time'].max()}")

    # Save cache
    df_export = df.copy()
    df_export['time'] = df_export['time'].astype(str)
    cache_data = {
        'metadata': {
            'cached_at': datetime.now().isoformat(),
            'start_date': str(df['time'].min().date()),
            'end_date': str(df['time'].max().date()),
            'total_candles': len(df),
            'granularity': '5min',
            'symbol': 'BTC-USD'
        },
        'data': df_export.to_dict('records')
    }
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f)
    print(f"Cached to {cache_file} ({os.path.getsize(cache_file) / 1024 / 1024:.1f} MB)")

    return df


# ============================================================================
# 5-MINUTE LOOKUP TABLE
# ============================================================================

def build_5min_lookup(df_5min):
    """Build dict mapping hourly timestamp -> sorted list of 5-min bar dicts.

    Each hourly key (e.g., 14:00) maps to up to 12 five-minute bars
    (14:00, 14:05, ... 14:55).
    """
    print("Building 5-min lookup table...")
    df_5min = df_5min.copy()
    df_5min['hour_key'] = df_5min['time'].dt.floor('h')

    lookup = {}
    for hour_key, group in df_5min.groupby('hour_key'):
        bars = []
        for _, row in group.sort_values('time').iterrows():
            bars.append({
                'time': row['time'],
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
            })
        lookup[hour_key] = bars

    # Stats
    bar_counts = [len(v) for v in lookup.values()]
    full_hours = sum(1 for c in bar_counts if c == 12)
    print(f"  {len(lookup):,} hourly slots, {sum(bar_counts):,} total 5-min bars")
    print(f"  {full_hours:,} hours with full 12 bars ({full_hours/len(lookup)*100:.1f}%)")
    print(f"  Avg bars/hour: {np.mean(bar_counts):.1f}, Min: {min(bar_counts)}, Max: {max(bar_counts)}")

    return lookup


# ============================================================================
# BACKTEST WITH 5-MINUTE EXIT SIMULATION
# ============================================================================

STRATEGY_REGIME = {
    'Williams %R Mean Reversion': 'RANGING',
    'ADX Momentum Thrust': 'TRENDING',
    'Bollinger Band Mean Reversion': 'VOLATILE',
}


def backtest_unified_with_5min(df, params, five_min_lookup):
    """Run unified backtest with 5-min intra-hour stop/TP checks.

    Entries and indicator exits on hourly bars.
    Price-based stops and take-profits checked on 5-min bars.
    """
    label = params['label']
    print(f"\n{'='*80}")
    print(f"  Running (5-min enhanced): {label}")
    print(f"{'='*80}")

    df = calculate_all_indicators(df, params)
    df = df.dropna().reset_index(drop=True)

    portfolio = 1000.0
    position = None
    entry_price = 0
    entry_time = None
    entry_bar = 0
    active_strategy = None
    trades = []
    high_watermark = 0
    non_matching_count = 0
    wr_min_hold = params.get('wr_min_hold', 0)
    atr_mult_trend = params.get('atr_mult_trend', 1.5)
    atr_mult_reversion = params.get('atr_mult_reversion', 1.5)
    cost_per_side = (params.get('slippage_pct', 0) + params.get('fee_pct', 0)) / 100

    # 5-min specific state
    stop_price = None
    take_profit_price = None
    five_min_bars_held = 0

    for i in range(5, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i - 1]
        regime = current['regime']
        current_atr = float(current['atr']) if not pd.isna(current['atr']) else 0

        # === POSITION EXIT LOGIC ===
        if position == 'LONG':
            exit_reason = None
            bars_held = i - entry_bar

            # ── STEP 1: 5-MIN PRICE CHECKS ──
            hour_key = current['time'].floor('h')
            five_min_bars = five_min_lookup.get(hour_key, [])
            exited_in_5min = False

            for bar in five_min_bars:
                five_min_bars_held += 1

                # ADX: update high watermark (ratchets up, used at hourly close)
                if active_strategy == 'ADX Momentum Thrust':
                    if bar['high'] > high_watermark:
                        high_watermark = bar['high']

                exit_5min = None

                # Emergency stop ONLY (5% below entry — always applies)
                if bar['close'] <= entry_price * 0.95:
                    exit_5min = ('Emergency stop (5-min)', float(bar['close']), bar['time'])

                if exit_5min:
                    reason, exit_px, exit_ts = exit_5min
                    exit_px *= (1 - cost_per_side)  # Apply exit cost
                    pnl_pct = ((exit_px - entry_price) / entry_price) * 100
                    portfolio *= (1 + pnl_pct / 100)
                    trades.append({
                        'entry_time': entry_time,
                        'entry_price': float(entry_price),
                        'exit_time': exit_ts,
                        'exit_price': exit_px,
                        'pnl_pct': float(pnl_pct),
                        'exit_reason': reason,
                        'strategy': active_strategy,
                        'portfolio': float(portfolio),
                        'win': bool(pnl_pct > 0),
                        'exit_via': '5min',
                    })
                    position = None
                    active_strategy = None
                    stop_price = None
                    take_profit_price = None
                    five_min_bars_held = 0
                    non_matching_count = 0
                    exited_in_5min = True
                    break

            if exited_in_5min:
                continue

            # ── STEP 2: HOURLY INDICATOR EXITS ──
            if active_strategy == 'Williams %R Mean Reversion':
                if bars_held >= wr_min_hold:
                    if current['williams_r'] < -20:
                        exit_reason = 'Williams %R < -20 (overbought zone)'
                    elif current['close'] >= current['sma_21'] * 1.015:
                        exit_reason = 'Price reached SMA(21) + 1.5% target'

            elif active_strategy == 'ADX Momentum Thrust':
                high_watermark = max(high_watermark, float(current['close']))
                # Recompute trailing stop with updated watermark
                stop_price = high_watermark - (atr_mult_trend * current_atr)
                if current['minus_di'] > current['plus_di'] and prev['minus_di'] <= prev['plus_di']:
                    exit_reason = '-DI crossed above +DI'
                elif current['adx'] < 20:
                    exit_reason = 'ADX dropped below 20'
                elif current['rsi'] > 75:
                    exit_reason = 'RSI > 75 (overbought)'
                elif current['close'] <= stop_price:
                    exit_reason = f'Trailing stop ({atr_mult_trend}x ATR from high)'

            elif active_strategy == 'Bollinger Band Mean Reversion':
                if current['close'] >= current['sma_20']:
                    exit_reason = 'Price reached SMA(20) middle band'
                elif current['rsi'] > 70:
                    exit_reason = 'RSI > 70 (overbought)'
                elif current['close'] <= entry_price - (atr_mult_reversion * current_atr):
                    exit_reason = f'Stop loss ({atr_mult_reversion}x ATR below entry)'

            # Emergency stop on hourly close (fallback for hours with no 5-min data)
            if not exit_reason and current['close'] <= entry_price * 0.95:
                exit_reason = 'Emergency stop (5% below entry)'

            # Force-sell on regime mismatch (3-bar buffer)
            if regime != STRATEGY_REGIME.get(active_strategy):
                non_matching_count += 1
                if non_matching_count >= 3 and not exit_reason:
                    exit_reason = f'Force-sell: regime changed to {regime}'
            else:
                non_matching_count = 0

            if exit_reason:
                exit_price = float(current['close']) * (1 - cost_per_side)
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                portfolio *= (1 + pnl_pct / 100)
                trades.append({
                    'entry_time': entry_time,
                    'entry_price': float(entry_price),
                    'exit_time': current['time'],
                    'exit_price': exit_price,
                    'pnl_pct': float(pnl_pct),
                    'exit_reason': exit_reason,
                    'strategy': active_strategy,
                    'portfolio': float(portfolio),
                    'win': bool(pnl_pct > 0),
                    'exit_via': 'hourly',
                })
                position = None
                active_strategy = None
                stop_price = None
                take_profit_price = None
                five_min_bars_held = 0
                non_matching_count = 0
                continue

        # === POSITION ENTRY LOGIC (same as hourly backtest) ===
        if position is None:

            # RANGING -> Williams %R Mean Reversion
            if regime == 'RANGING':
                wr_cross = (prev['williams_r'] <= params['wr_threshold'] and
                            current['williams_r'] > params['wr_threshold'])
                sma_ok = (not params['wr_sma_filter']) or (current['close'] < current['sma_21'])

                if wr_cross and sma_ok:
                    position = 'LONG'
                    entry_price = float(current['close']) * (1 + cost_per_side)
                    entry_time = current['time']
                    entry_bar = i
                    active_strategy = 'Williams %R Mean Reversion'
                    stop_price = None  # WR: emergency 5% only
                    take_profit_price = float(current['sma_21']) * 1.015
                    five_min_bars_held = 0

            # TRENDING -> ADX Momentum Thrust
            elif regime == 'TRENDING':
                adx_rising = df.iloc[i]['adx'] > df.iloc[i - 5]['adx']
                adx_above_20 = current['adx'] > 20
                plus_di_above = current['plus_di'] > current['minus_di']
                rsi_ok = params['adx_rsi_low'] <= current['rsi'] <= params['adx_rsi_high']
                price_above_sma = current['close'] > current['sma_50']

                di_cross_ok = True
                if params['adx_di_cross_required']:
                    di_cross_ok = False
                    for offset in range(0, 15):
                        idx = i - offset
                        if idx >= 1:
                            bar = df.iloc[idx]
                            bar_prev = df.iloc[idx - 1]
                            if (bar['plus_di'] > bar['minus_di'] and
                                    bar_prev['plus_di'] <= bar_prev['minus_di']):
                                di_cross_ok = True
                                break

                if adx_above_20 and adx_rising and plus_di_above and di_cross_ok and rsi_ok and price_above_sma:
                    position = 'LONG'
                    entry_price = float(current['close']) * (1 + cost_per_side)
                    entry_time = current['time']
                    entry_bar = i
                    active_strategy = 'ADX Momentum Thrust'
                    high_watermark = float(current['close'])
                    stop_price = high_watermark - (atr_mult_trend * current_atr)
                    take_profit_price = None  # ADX: trailing stop only
                    five_min_bars_held = 0

            # VOLATILE -> Bollinger Band Mean Reversion
            elif regime == 'VOLATILE':
                if current['close'] < current['bb_lower'] and current['rsi'] < params['bb_rsi_entry']:
                    position = 'LONG'
                    entry_price = float(current['close']) * (1 + cost_per_side)
                    entry_time = current['time']
                    entry_bar = i
                    active_strategy = 'Bollinger Band Mean Reversion'
                    stop_price = entry_price - (atr_mult_reversion * current_atr)
                    take_profit_price = float(current['sma_20'])
                    five_min_bars_held = 0

    # Close any open position at end
    if position == 'LONG':
        last = df.iloc[-1]
        exit_price = float(last['close']) * (1 - cost_per_side)
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        portfolio *= (1 + pnl_pct / 100)
        trades.append({
            'entry_time': entry_time,
            'entry_price': float(entry_price),
            'exit_time': last['time'],
            'exit_price': exit_price,
            'pnl_pct': float(pnl_pct),
            'exit_reason': 'End of backtest period',
            'strategy': active_strategy,
            'portfolio': float(portfolio),
            'win': bool(pnl_pct > 0),
            'exit_via': 'hourly',
        })

    return trades, portfolio


# ============================================================================
# COMPARISON OUTPUT
# ============================================================================

def print_5min_comparison(hourly_stats, fivemin_stats, hourly_trades, fivemin_trades, label):
    """Print side-by-side comparison: hourly-only vs 5-min enhanced"""

    h = hourly_stats
    f = fivemin_stats

    print(f"\n{'=' * 100}")
    print(f"  HOURLY-ONLY vs 5-MIN ENHANCED — {label}")
    print(f"{'=' * 100}")

    print(f"\n{'METRIC':<30s} {'HOURLY-ONLY':>18s} {'5-MIN ENHANCED':>18s} {'DELTA':>12s}")
    print("-" * 82)

    def row(lbl, hv, fv, fmt='.2f', pct=False):
        suffix = '%' if pct else ''
        h_str = f"{hv:{fmt}}{suffix}" if hv is not None else "N/A"
        f_str = f"{fv:{fmt}}{suffix}" if fv is not None else "N/A"
        if hv is not None and fv is not None:
            delta = fv - hv
            d_str = f"{delta:+{fmt}}{suffix}"
        else:
            d_str = ""
        print(f"  {lbl:<28s} {h_str:>18s} {f_str:>18s} {d_str:>12s}")

    row('Total Trades', h.get('trades', 0), f.get('trades', 0), 'd')
    row('Wins', h.get('wins', 0), f.get('wins', 0), 'd')
    row('Losses', h.get('losses', 0), f.get('losses', 0), 'd')
    row('Win Rate', h.get('win_rate', 0) * 100, f.get('win_rate', 0) * 100, '.1f', pct=True)
    row('Avg Win', h.get('avg_win', 0), f.get('avg_win', 0), '.2f', pct=True)
    row('Avg Loss', h.get('avg_loss', 0), f.get('avg_loss', 0), '.2f', pct=True)
    row('Profit Factor', h.get('profit_factor', 0), f.get('profit_factor', 0), '.2f')
    row('Total Return', h.get('total_return', 0), f.get('total_return', 0), '.2f', pct=True)
    row('Final Portfolio ($)', h.get('final_portfolio', 1000), f.get('final_portfolio', 1000), ',.2f')
    row('Max Drawdown', h.get('max_drawdown', 0), f.get('max_drawdown', 0), '.2f', pct=True)

    # Per-strategy breakdown
    strategies = [
        ('Williams %R Mean Reversion', 'RANGING'),
        ('ADX Momentum Thrust', 'TRENDING'),
        ('Bollinger Band Mean Reversion', 'VOLATILE'),
    ]

    print(f"\n  {'Strategy':<35s} {'Mode':<14s} {'Trades':>7s} {'WR':>7s} {'PF':>7s} {'P&L':>9s}")
    print("  " + "-" * 82)

    for strat_name, _ in strategies:
        hs = compute_strategy_stats(hourly_trades, strat_name)
        fs = compute_strategy_stats(fivemin_trades, strat_name)
        for mode, st in [('Hourly', hs), ('5-min', fs)]:
            if st:
                print(f"  {strat_name:<35s} {mode:<14s} {st['trades']:>7d} "
                      f"{st['win_rate']*100:>6.1f}% {st['profit_factor']:>7.2f} {st['pnl']:>+8.2f}%")
            else:
                print(f"  {strat_name:<35s} {mode:<14s}       0    N/A     N/A      N/A")

    # Exit mechanism breakdown for 5-min run
    print(f"\n  EXIT MECHANISM BREAKDOWN (5-min enhanced):")
    print("  " + "-" * 60)
    via_5min = [t for t in fivemin_trades if t.get('exit_via') == '5min']
    via_hourly = [t for t in fivemin_trades if t.get('exit_via') == 'hourly']
    total = len(fivemin_trades)
    print(f"  Exited via 5-min check:  {len(via_5min):>5d} trades ({len(via_5min)/total*100:.1f}%)" if total else "")
    print(f"  Exited via hourly check: {len(via_hourly):>5d} trades ({len(via_hourly)/total*100:.1f}%)" if total else "")

    # Detailed exit reasons for 5-min run
    print(f"\n  EXIT REASONS (5-min enhanced):")
    print("  " + "-" * 60)
    reasons = {}
    for t in fivemin_trades:
        r = t['exit_reason']
        if r not in reasons:
            reasons[r] = {'count': 0, 'pnl': 0}
        reasons[r]['count'] += 1
        reasons[r]['pnl'] += t['pnl_pct']
    for reason, stats in sorted(reasons.items(), key=lambda x: -x[1]['count']):
        print(f"  {reason}: {stats['count']} trades, {stats['pnl']:+.2f}% total P&L")


def save_5min_results(all_results):
    """Save all comparison results to JSON"""
    def clean_trades(trades):
        return [{**t, 'entry_time': str(t['entry_time']), 'exit_time': str(t['exit_time'])} for t in trades]

    output = {}
    for key, data in all_results.items():
        output[key] = {
            **data['stats'],
            'trades_detail': clean_trades(data['trades']),
        }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"backtest_5min_comparison_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {filename}")
    return filename


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 100)
    print("  5-MINUTE ENHANCED BACKTEST — HOURLY vs 5-MIN EXIT SIMULATION")
    print("  Starting Portfolio: $1,000 each")
    print("=" * 100 + "\n")

    # 1. Load hourly data + classify regimes
    df = fetch_and_cache_hourly_data()
    if df is None:
        return
    df, regime_counts = classify_regimes(df)

    # 2. Load 5-min data + build lookup
    df_5min = fetch_and_cache_5min_data()
    if df_5min is None:
        print("ERROR: Could not load 5-min data")
        return
    five_min_lookup = build_5min_lookup(df_5min)

    # Free the full 5-min DataFrame to save memory
    del df_5min

    all_results = {}

    for param_set, param_label in [(STRICT_PARAMS, 'STRICT'), (LOOSENED_PARAMS, 'LOOSENED')]:
        # Run hourly-only baseline
        hourly_trades, hourly_portfolio = backtest_unified(df, param_set)
        hourly_stats = compute_stats(hourly_trades, f'{param_label} Hourly')

        # Run 5-min enhanced
        fivemin_trades, fivemin_portfolio = backtest_unified_with_5min(df, param_set, five_min_lookup)
        fivemin_stats = compute_stats(fivemin_trades, f'{param_label} 5-min')

        # Print comparison
        print_5min_comparison(hourly_stats, fivemin_stats, hourly_trades, fivemin_trades, param_label)

        all_results[f'{param_label}_hourly'] = {'stats': hourly_stats, 'trades': hourly_trades}
        all_results[f'{param_label}_5min'] = {'stats': fivemin_stats, 'trades': fivemin_trades}

    # Save
    save_5min_results(all_results)

    print("\n" + "=" * 100)
    print("  DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
