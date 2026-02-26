"""Analyze why Williams %R Mean Reversion underperforms on hourly BTC data.

Investigates:
1. Trade duration distribution (how many bars/hours do trades last?)
2. Williams %R oscillation frequency (hourly vs daily noise)
3. Period sensitivity (14 vs 21, 28, 42, 56)
4. Minimum hold time (force holding 3, 6, 12, 24 bars before exit check)
5. Exit threshold analysis (-20 vs -30, -40, -50)

Uses RegimeClassifier from market_regime.py for regime classification.
Only trades during RANGING regime bars.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
from market_regime import RegimeClassifier


# ============================================================================
# DATA LOADING
# ============================================================================

def load_hourly_data(cache_file='btc_4year_hourly_cache.json'):
    """Load cached 4-year hourly BTC data"""
    if not os.path.exists(cache_file):
        print(f"ERROR: Cache file not found: {cache_file}")
        return None

    with open(cache_file, 'r') as f:
        cache_data = json.load(f)

    df = pd.DataFrame(cache_data['data'])
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    print(f"Loaded {len(df):,} hourly candles")
    print(f"Range: {df['time'].min()} to {df['time'].max()}")
    return df


# ============================================================================
# WILLIAMS %R CALCULATION
# ============================================================================

def calculate_williams_r(df, period=14):
    """Calculate Williams %R for a given period"""
    high_roll = df['high'].rolling(window=period).max()
    low_roll = df['low'].rolling(window=period).min()
    wr = -100 * ((high_roll - df['close']) / (high_roll - low_roll))
    return wr


# ============================================================================
# BASELINE BACKTEST (exact production params)
# ============================================================================

def backtest_williams_r(df, wr_period=14, exit_threshold=-20, min_hold_bars=0,
                        sma_filter=True, sma_period=21, entry_threshold=-80,
                        label=""):
    """Run Williams %R backtest on RANGING bars only.

    Returns list of trade dicts with entry/exit times, pnl, duration, etc.
    """
    df = df.copy()

    # Calculate Williams %R with specified period
    df['williams_r'] = calculate_williams_r(df, wr_period)
    df['sma'] = df['close'].rolling(window=sma_period).mean()

    df = df.dropna().reset_index(drop=True)

    position = None
    entry_price = 0.0
    entry_time = None
    entry_bar = 0
    trades = []

    for i in range(1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i - 1]
        regime = current['regime']

        # === EXIT LOGIC ===
        if position == 'LONG':
            bars_held = i - entry_bar
            exit_reason = None

            # Force-sell if regime leaves RANGING for 3 consecutive bars
            # (simplified: immediate exit on non-RANGING for this analysis)
            if regime != 'RANGING':
                exit_reason = 'Regime changed'

            # Emergency stop: 5% below entry
            elif current['close'] <= entry_price * 0.95:
                exit_reason = 'Emergency stop (5%)'

            # Only check exit signals after min_hold_bars
            elif bars_held >= min_hold_bars:
                # Williams %R exit threshold
                if current['williams_r'] < exit_threshold:
                    exit_reason = f'WR < {exit_threshold}'
                # Price target: SMA + 1.5%
                elif current['close'] >= current['sma'] * 1.015:
                    exit_reason = 'Price >= SMA + 1.5%'

            if exit_reason:
                exit_price = float(current['close'])
                pnl = ((exit_price - entry_price) / entry_price) * 100
                duration = i - entry_bar
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current['time'],
                    'entry_price': float(entry_price),
                    'exit_price': exit_price,
                    'pnl_pct': float(pnl),
                    'duration_bars': duration,
                    'exit_reason': exit_reason,
                    'win': bool(pnl > 0),
                })
                position = None

        # === ENTRY LOGIC (only in RANGING) ===
        if position is None and regime == 'RANGING':
            wr_cross = (prev['williams_r'] <= entry_threshold and
                        current['williams_r'] > entry_threshold)
            sma_ok = (not sma_filter) or (current['close'] < current['sma'])

            if wr_cross and sma_ok:
                position = 'LONG'
                entry_price = float(current['close'])
                entry_time = current['time']
                entry_bar = i

    # Close any open position
    if position == 'LONG':
        last = df.iloc[-1]
        exit_price = float(last['close'])
        pnl = ((exit_price - entry_price) / entry_price) * 100
        duration = len(df) - 1 - entry_bar
        trades.append({
            'entry_time': entry_time,
            'exit_time': last['time'],
            'entry_price': float(entry_price),
            'exit_price': exit_price,
            'pnl_pct': float(pnl),
            'duration_bars': duration,
            'exit_reason': 'End of data',
            'win': bool(pnl > 0),
        })

    return trades


def compute_stats(trades):
    """Compute summary stats from trade list"""
    if not trades:
        return {
            'trades': 0, 'wins': 0, 'losses': 0,
            'win_rate': 0, 'avg_win': 0, 'avg_loss': 0,
            'total_pnl': 0, 'profit_factor': 0,
            'avg_duration': 0, 'median_duration': 0,
        }

    wins = sum(1 for t in trades if t['win'])
    losses = len(trades) - wins
    win_rate = wins / len(trades)
    avg_win = np.mean([t['pnl_pct'] for t in trades if t['win']]) if wins > 0 else 0
    avg_loss = np.mean([t['pnl_pct'] for t in trades if not t['win']]) if losses > 0 else 0
    total_pnl = sum(t['pnl_pct'] for t in trades)
    total_win_pnl = sum(t['pnl_pct'] for t in trades if t['win'])
    total_loss_pnl = abs(sum(t['pnl_pct'] for t in trades if not t['win']))
    profit_factor = total_win_pnl / total_loss_pnl if total_loss_pnl > 0 else float('inf')
    durations = [t['duration_bars'] for t in trades]
    avg_duration = np.mean(durations)
    median_duration = np.median(durations)

    return {
        'trades': len(trades),
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'total_pnl': total_pnl,
        'profit_factor': profit_factor,
        'avg_duration': avg_duration,
        'median_duration': median_duration,
    }


def print_stats_line(label, stats):
    """Print one-line summary for a parameter variation"""
    if stats['trades'] == 0:
        print(f"  {label:<45s}   0 trades")
        return
    print(f"  {label:<45s} "
          f"{stats['trades']:>5d} trades | "
          f"WR: {stats['win_rate']*100:>5.1f}% | "
          f"PF: {stats['profit_factor']:>5.2f} | "
          f"P&L: {stats['total_pnl']:>+8.2f}% | "
          f"Avg Win: {stats['avg_win']:>+6.2f}% | "
          f"Avg Loss: {stats['avg_loss']:>+6.2f}% | "
          f"Avg Dur: {stats['avg_duration']:>5.1f}h | "
          f"Med Dur: {stats['median_duration']:>4.0f}h")


# ============================================================================
# ANALYSIS 1: TRADE DURATION DISTRIBUTION
# ============================================================================

def analysis_1_trade_duration(df):
    """How many bars (hours) do WR trades typically last?"""
    print("\n" + "=" * 100)
    print("ANALYSIS 1: TRADE DURATION DISTRIBUTION (baseline params: WR=14, entry=-80, exit=-20)")
    print("=" * 100)

    trades = backtest_williams_r(df)
    stats = compute_stats(trades)

    print(f"\n  Total trades: {stats['trades']}")
    print(f"  Win rate: {stats['win_rate']*100:.1f}%")
    print(f"  Total P&L: {stats['total_pnl']:+.2f}%")
    print(f"  Profit factor: {stats['profit_factor']:.2f}")
    print(f"  Avg win: {stats['avg_win']:+.2f}% | Avg loss: {stats['avg_loss']:+.2f}%")

    durations = [t['duration_bars'] for t in trades]
    if not durations:
        print("\n  No trades generated.")
        return trades

    print(f"\n  Duration statistics (in hours/bars):")
    print(f"    Mean:   {np.mean(durations):.1f}")
    print(f"    Median: {np.median(durations):.0f}")
    print(f"    Min:    {min(durations)}")
    print(f"    Max:    {max(durations)}")
    print(f"    Std:    {np.std(durations):.1f}")

    # Distribution buckets
    buckets = [1, 2, 3, 4, 5, 6, 8, 10, 12, 18, 24, 48, 72, 168]
    print(f"\n  Duration distribution:")
    for b in buckets:
        count = sum(1 for d in durations if d <= b)
        pct = count / len(durations) * 100
        bar = '#' * int(pct / 2)
        print(f"    <= {b:>3d}h: {count:>4d} ({pct:>5.1f}%) {bar}")

    # Win rate by duration bucket
    print(f"\n  Win rate by trade duration:")
    duration_bins = [(1, 1, '1h'), (2, 2, '2h'), (3, 5, '3-5h'),
                     (6, 12, '6-12h'), (13, 24, '13-24h'), (25, 999999, '25h+')]
    for lo, hi, label in duration_bins:
        bucket_trades = [t for t in trades if lo <= t['duration_bars'] <= hi]
        if bucket_trades:
            bw = sum(1 for t in bucket_trades if t['win'])
            bpnl = sum(t['pnl_pct'] for t in bucket_trades)
            print(f"    {label:>8s}: {len(bucket_trades):>4d} trades, "
                  f"WR: {bw/len(bucket_trades)*100:>5.1f}%, P&L: {bpnl:>+8.2f}%")

    # Exit reason breakdown
    print(f"\n  Exit reason breakdown:")
    reasons = {}
    for t in trades:
        r = t['exit_reason']
        if r not in reasons:
            reasons[r] = {'count': 0, 'pnl': 0, 'wins': 0, 'durations': []}
        reasons[r]['count'] += 1
        reasons[r]['pnl'] += t['pnl_pct']
        reasons[r]['durations'].append(t['duration_bars'])
        if t['win']:
            reasons[r]['wins'] += 1
    for reason, info in sorted(reasons.items(), key=lambda x: -x[1]['count']):
        wr = info['wins'] / info['count'] * 100
        avg_dur = np.mean(info['durations'])
        print(f"    {reason:<30s}: {info['count']:>4d} trades, "
              f"WR: {wr:>5.1f}%, P&L: {info['pnl']:>+8.2f}%, "
              f"avg dur: {avg_dur:.1f}h")

    return trades


# ============================================================================
# ANALYSIS 2: WILLIAMS %R OSCILLATION FREQUENCY
# ============================================================================

def analysis_2_oscillation_frequency(df):
    """How often does WR cross -80 and -20 on hourly data?"""
    print("\n" + "=" * 100)
    print("ANALYSIS 2: WILLIAMS %R OSCILLATION FREQUENCY (hourly)")
    print("=" * 100)

    df = df.copy()

    for period in [14, 21, 28, 42, 56]:
        wr = calculate_williams_r(df, period)
        wr_clean = wr.dropna()

        # Count crosses above -80
        crosses_above_80 = 0
        crosses_below_20 = 0
        for i in range(1, len(wr_clean)):
            prev_val = wr_clean.iloc[i - 1]
            curr_val = wr_clean.iloc[i]
            if prev_val <= -80 and curr_val > -80:
                crosses_above_80 += 1
            if prev_val >= -20 and curr_val < -20:
                crosses_below_20 += 1

        total_bars = len(wr_clean)
        years = total_bars / (24 * 365)

        print(f"\n  WR period = {period}:")
        print(f"    Total bars: {total_bars:,} ({years:.1f} years)")
        print(f"    Crosses above -80: {crosses_above_80:>5d} (avg {crosses_above_80/years:.0f}/year, "
              f"1 every {total_bars/max(crosses_above_80,1):.0f} bars)")
        print(f"    Crosses below -20: {crosses_below_20:>5d} (avg {crosses_below_20/years:.0f}/year, "
              f"1 every {total_bars/max(crosses_below_20,1):.0f} bars)")

        # How many bars between a -80 cross-up and the next -20 cross-down?
        gaps = []
        state = 'waiting_for_80_cross'
        cross_bar = 0
        for i in range(1, len(wr_clean)):
            prev_val = wr_clean.iloc[i - 1]
            curr_val = wr_clean.iloc[i]
            if state == 'waiting_for_80_cross':
                if prev_val <= -80 and curr_val > -80:
                    state = 'waiting_for_20_cross'
                    cross_bar = i
            elif state == 'waiting_for_20_cross':
                if curr_val < -20:
                    gaps.append(i - cross_bar)
                    state = 'waiting_for_80_cross'

        if gaps:
            print(f"    Bars from -80 cross to -20 cross:")
            print(f"      Mean: {np.mean(gaps):.1f} bars | Median: {np.median(gaps):.0f} bars | "
                  f"Min: {min(gaps)} | Max: {max(gaps)}")
            pct_1bar = sum(1 for g in gaps if g <= 1) / len(gaps) * 100
            pct_2bar = sum(1 for g in gaps if g <= 2) / len(gaps) * 100
            pct_3bar = sum(1 for g in gaps if g <= 3) / len(gaps) * 100
            pct_5bar = sum(1 for g in gaps if g <= 5) / len(gaps) * 100
            print(f"      <= 1 bar: {pct_1bar:.1f}% | <= 2 bars: {pct_2bar:.1f}% | "
                  f"<= 3 bars: {pct_3bar:.1f}% | <= 5 bars: {pct_5bar:.1f}%")

    # Compare: what would daily WR(14) look like?
    # Simulate daily by resampling hourly to daily
    print(f"\n  --- Comparison: Simulated DAILY WR(14) from hourly data ---")
    df_daily_sim = df.copy()
    df_daily_sim = df_daily_sim.set_index('time')
    df_daily = df_daily_sim.resample('D').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()

    wr_daily = calculate_williams_r(df_daily, 14)
    wr_daily_clean = wr_daily.dropna()

    crosses_80_d = 0
    crosses_20_d = 0
    for i in range(1, len(wr_daily_clean)):
        prev_val = wr_daily_clean.iloc[i - 1]
        curr_val = wr_daily_clean.iloc[i]
        if prev_val <= -80 and curr_val > -80:
            crosses_80_d += 1
        if prev_val >= -20 and curr_val < -20:
            crosses_20_d += 1

    daily_bars = len(wr_daily_clean)
    daily_years = daily_bars / 365
    print(f"    Total bars: {daily_bars:,} ({daily_years:.1f} years)")
    print(f"    Crosses above -80: {crosses_80_d:>5d} (avg {crosses_80_d/daily_years:.0f}/year, "
          f"1 every {daily_bars/max(crosses_80_d,1):.0f} bars)")
    print(f"    Crosses below -20: {crosses_20_d:>5d} (avg {crosses_20_d/daily_years:.0f}/year, "
          f"1 every {daily_bars/max(crosses_20_d,1):.0f} bars)")

    # Frequency ratio
    hourly_wr14 = 0
    wr14 = calculate_williams_r(df, 14).dropna()
    for i in range(1, len(wr14)):
        if wr14.iloc[i - 1] <= -80 and wr14.iloc[i] > -80:
            hourly_wr14 += 1
    ratio = hourly_wr14 / max(crosses_80_d, 1)
    print(f"\n    ** Hourly WR(14) generates {ratio:.1f}x more -80 crossovers than daily WR(14)")
    print(f"    ** But only 24x more bars (hourly = {len(wr14):,}, daily = {daily_bars:,})")
    print(f"    ** Signal-to-bar ratio: hourly = 1/{len(wr14)/max(hourly_wr14,1):.0f}, "
          f"daily = 1/{daily_bars/max(crosses_80_d,1):.0f}")


# ============================================================================
# ANALYSIS 3: PERIOD SENSITIVITY
# ============================================================================

def analysis_3_period_sensitivity(df):
    """What happens with longer WR periods?"""
    print("\n" + "=" * 100)
    print("ANALYSIS 3: WILLIAMS %R PERIOD SENSITIVITY (exit=-20, SMA filter ON)")
    print("=" * 100)

    periods = [14, 21, 28, 42, 56, 72, 96]

    print(f"\n  Testing WR periods: {periods}")
    print()

    for period in periods:
        trades = backtest_williams_r(df, wr_period=period, exit_threshold=-20,
                                     min_hold_bars=0, sma_filter=True,
                                     sma_period=21, entry_threshold=-80)
        stats = compute_stats(trades)
        print_stats_line(f"WR({period}), entry=-80, exit=-20", stats)

    # Also test with wider entry thresholds for longer periods
    print(f"\n  --- With adjusted entry threshold (-85) for longer periods ---")
    for period in periods:
        trades = backtest_williams_r(df, wr_period=period, exit_threshold=-20,
                                     min_hold_bars=0, sma_filter=True,
                                     sma_period=21, entry_threshold=-85)
        stats = compute_stats(trades)
        print_stats_line(f"WR({period}), entry=-85, exit=-20", stats)


# ============================================================================
# ANALYSIS 4: MINIMUM HOLD TIME
# ============================================================================

def analysis_4_min_hold_time(df):
    """What if we require holding for N bars before checking exits?"""
    print("\n" + "=" * 100)
    print("ANALYSIS 4: MINIMUM HOLD TIME (WR=14, entry=-80, exit=-20)")
    print("=" * 100)

    hold_bars = [0, 1, 2, 3, 4, 6, 8, 12, 18, 24, 36, 48]

    print(f"\n  Testing minimum hold times: {hold_bars} bars")
    print()

    for hold in hold_bars:
        trades = backtest_williams_r(df, wr_period=14, exit_threshold=-20,
                                     min_hold_bars=hold, sma_filter=True,
                                     sma_period=21, entry_threshold=-80)
        stats = compute_stats(trades)
        print_stats_line(f"Min hold = {hold:>2d} bars ({hold}h)", stats)

    # Also test with longer WR period + hold time (combined)
    print(f"\n  --- Combined: WR(28) + minimum hold time ---")
    for hold in hold_bars:
        trades = backtest_williams_r(df, wr_period=28, exit_threshold=-20,
                                     min_hold_bars=hold, sma_filter=True,
                                     sma_period=21, entry_threshold=-80)
        stats = compute_stats(trades)
        print_stats_line(f"WR(28) + min hold {hold:>2d}h", stats)

    # WR(42) + hold time
    print(f"\n  --- Combined: WR(42) + minimum hold time ---")
    for hold in [0, 3, 6, 12, 24]:
        trades = backtest_williams_r(df, wr_period=42, exit_threshold=-20,
                                     min_hold_bars=hold, sma_filter=True,
                                     sma_period=21, entry_threshold=-80)
        stats = compute_stats(trades)
        print_stats_line(f"WR(42) + min hold {hold:>2d}h", stats)


# ============================================================================
# ANALYSIS 5: EXIT THRESHOLD
# ============================================================================

def analysis_5_exit_threshold(df):
    """What if we use less aggressive exit thresholds?"""
    print("\n" + "=" * 100)
    print("ANALYSIS 5: EXIT THRESHOLD ANALYSIS (WR=14, entry=-80)")
    print("=" * 100)

    exit_thresholds = [-10, -15, -20, -25, -30, -35, -40, -45, -50, -55, -60]

    print(f"\n  Testing exit thresholds: {exit_thresholds}")
    print(f"  (More negative = requires WR to drop further before selling)")
    print()

    for threshold in exit_thresholds:
        trades = backtest_williams_r(df, wr_period=14, exit_threshold=threshold,
                                     min_hold_bars=0, sma_filter=True,
                                     sma_period=21, entry_threshold=-80)
        stats = compute_stats(trades)
        print_stats_line(f"Exit at WR < {threshold}", stats)

    # Also test with WR(28) + exit threshold variations
    print(f"\n  --- With WR(28) period ---")
    for threshold in exit_thresholds:
        trades = backtest_williams_r(df, wr_period=28, exit_threshold=threshold,
                                     min_hold_bars=0, sma_filter=True,
                                     sma_period=21, entry_threshold=-80)
        stats = compute_stats(trades)
        print_stats_line(f"WR(28), exit at WR < {threshold}", stats)

    # Best combos: exit threshold + min hold
    print(f"\n  --- Best combo search: WR(14) + exit threshold + min hold ---")
    best_pnl = -999
    best_combo = ""
    for hold in [0, 3, 6, 12, 24]:
        for threshold in [-20, -30, -40, -50]:
            trades = backtest_williams_r(df, wr_period=14, exit_threshold=threshold,
                                         min_hold_bars=hold, sma_filter=True,
                                         sma_period=21, entry_threshold=-80)
            stats = compute_stats(trades)
            label = f"WR(14), exit={threshold}, hold={hold}h"
            print_stats_line(label, stats)
            if stats['total_pnl'] > best_pnl and stats['trades'] >= 5:
                best_pnl = stats['total_pnl']
                best_combo = label

    if best_combo:
        print(f"\n  ** Best combo (by P&L with 5+ trades): {best_combo} => {best_pnl:+.2f}%")

    # Best combos with WR(28)
    print(f"\n  --- Best combo search: WR(28) + exit threshold + min hold ---")
    best_pnl = -999
    best_combo = ""
    for hold in [0, 3, 6, 12, 24]:
        for threshold in [-20, -30, -40, -50]:
            trades = backtest_williams_r(df, wr_period=28, exit_threshold=threshold,
                                         min_hold_bars=hold, sma_filter=True,
                                         sma_period=21, entry_threshold=-80)
            stats = compute_stats(trades)
            label = f"WR(28), exit={threshold}, hold={hold}h"
            print_stats_line(label, stats)
            if stats['total_pnl'] > best_pnl and stats['trades'] >= 5:
                best_pnl = stats['total_pnl']
                best_combo = label

    if best_combo:
        print(f"\n  ** Best combo (by P&L with 5+ trades): {best_combo} => {best_pnl:+.2f}%")


# ============================================================================
# SUMMARY: OVERALL BEST CONFIGURATIONS
# ============================================================================

def summary_best_configs(df):
    """Run a comprehensive grid search and print top configurations"""
    print("\n" + "=" * 100)
    print("SUMMARY: TOP CONFIGURATIONS (grid search)")
    print("=" * 100)

    results = []

    for wr_period in [14, 21, 28, 42, 56]:
        for entry_thresh in [-80, -85, -90]:
            for exit_thresh in [-15, -20, -30, -40, -50]:
                for hold in [0, 3, 6, 12, 24]:
                    for sma_filter in [True, False]:
                        trades = backtest_williams_r(
                            df, wr_period=wr_period, exit_threshold=exit_thresh,
                            min_hold_bars=hold, sma_filter=sma_filter,
                            sma_period=21, entry_threshold=entry_thresh
                        )
                        stats = compute_stats(trades)
                        if stats['trades'] >= 5:
                            results.append({
                                'wr_period': wr_period,
                                'entry': entry_thresh,
                                'exit': exit_thresh,
                                'hold': hold,
                                'sma_filter': sma_filter,
                                **stats,
                            })

    if not results:
        print("  No configurations with 5+ trades found.")
        return

    # Sort by profit factor (minimum 10 trades for reliability)
    reliable = [r for r in results if r['trades'] >= 10]
    if reliable:
        print(f"\n  TOP 10 BY PROFIT FACTOR (10+ trades):")
        print(f"  {'Config':<55s} {'Trades':>6s} {'WR':>7s} {'PF':>7s} {'P&L':>9s} {'AvgW':>7s} {'AvgL':>7s} {'AvgDur':>7s}")
        print(f"  {'-'*106}")
        for r in sorted(reliable, key=lambda x: x['profit_factor'], reverse=True)[:10]:
            sma_str = 'SMA' if r['sma_filter'] else 'noSMA'
            config = f"WR({r['wr_period']}), e={r['entry']}, x={r['exit']}, h={r['hold']}h, {sma_str}"
            print(f"  {config:<55s} {r['trades']:>6d} {r['win_rate']*100:>6.1f}% "
                  f"{r['profit_factor']:>7.2f} {r['total_pnl']:>+8.2f}% "
                  f"{r['avg_win']:>+6.2f}% {r['avg_loss']:>+6.2f}% {r['avg_duration']:>6.1f}h")

    # Sort by total P&L (10+ trades)
    if reliable:
        print(f"\n  TOP 10 BY TOTAL P&L (10+ trades):")
        print(f"  {'Config':<55s} {'Trades':>6s} {'WR':>7s} {'PF':>7s} {'P&L':>9s} {'AvgW':>7s} {'AvgL':>7s} {'AvgDur':>7s}")
        print(f"  {'-'*106}")
        for r in sorted(reliable, key=lambda x: x['total_pnl'], reverse=True)[:10]:
            sma_str = 'SMA' if r['sma_filter'] else 'noSMA'
            config = f"WR({r['wr_period']}), e={r['entry']}, x={r['exit']}, h={r['hold']}h, {sma_str}"
            print(f"  {config:<55s} {r['trades']:>6d} {r['win_rate']*100:>6.1f}% "
                  f"{r['profit_factor']:>7.2f} {r['total_pnl']:>+8.2f}% "
                  f"{r['avg_win']:>+6.2f}% {r['avg_loss']:>+6.2f}% {r['avg_duration']:>6.1f}h")

    # Sort by win rate (10+ trades)
    if reliable:
        print(f"\n  TOP 10 BY WIN RATE (10+ trades):")
        print(f"  {'Config':<55s} {'Trades':>6s} {'WR':>7s} {'PF':>7s} {'P&L':>9s} {'AvgW':>7s} {'AvgL':>7s} {'AvgDur':>7s}")
        print(f"  {'-'*106}")
        for r in sorted(reliable, key=lambda x: x['win_rate'], reverse=True)[:10]:
            sma_str = 'SMA' if r['sma_filter'] else 'noSMA'
            config = f"WR({r['wr_period']}), e={r['entry']}, x={r['exit']}, h={r['hold']}h, {sma_str}"
            print(f"  {config:<55s} {r['trades']:>6d} {r['win_rate']*100:>6.1f}% "
                  f"{r['profit_factor']:>7.2f} {r['total_pnl']:>+8.2f}% "
                  f"{r['avg_win']:>+6.2f}% {r['avg_loss']:>+6.2f}% {r['avg_duration']:>6.1f}h")

    # Configs that pass approval (Path A: WR>=55%, PF>=1.5)
    approved_a = [r for r in reliable if r['win_rate'] >= 0.55 and r['profit_factor'] >= 1.5]
    if approved_a:
        print(f"\n  CONFIGS THAT PASS PATH A APPROVAL (WR>=55% AND PF>=1.5, 10+ trades):")
        print(f"  {'Config':<55s} {'Trades':>6s} {'WR':>7s} {'PF':>7s} {'P&L':>9s}")
        print(f"  {'-'*85}")
        for r in sorted(approved_a, key=lambda x: x['total_pnl'], reverse=True):
            sma_str = 'SMA' if r['sma_filter'] else 'noSMA'
            config = f"WR({r['wr_period']}), e={r['entry']}, x={r['exit']}, h={r['hold']}h, {sma_str}"
            print(f"  {config:<55s} {r['trades']:>6d} {r['win_rate']*100:>6.1f}% "
                  f"{r['profit_factor']:>7.2f} {r['total_pnl']:>+8.2f}%")
    else:
        print(f"\n  ** NO configurations pass Path A approval (WR>=55% AND PF>=1.5 with 10+ trades)")

    # Path B check
    approved_b = [r for r in reliable if r['profit_factor'] >= 1.8
                  and abs(r['avg_loss']) > 0
                  and r['avg_win'] / abs(r['avg_loss']) >= 1.5]
    if approved_b:
        print(f"\n  CONFIGS THAT PASS PATH B APPROVAL (PF>=1.8, avgW/avgL>=1.5, 10+ trades):")
        print(f"  {'Config':<55s} {'Trades':>6s} {'WR':>7s} {'PF':>7s} {'P&L':>9s} {'W/L':>6s}")
        print(f"  {'-'*95}")
        for r in sorted(approved_b, key=lambda x: x['total_pnl'], reverse=True):
            sma_str = 'SMA' if r['sma_filter'] else 'noSMA'
            config = f"WR({r['wr_period']}), e={r['entry']}, x={r['exit']}, h={r['hold']}h, {sma_str}"
            wl_ratio = r['avg_win'] / abs(r['avg_loss']) if r['avg_loss'] != 0 else 0
            print(f"  {config:<55s} {r['trades']:>6d} {r['win_rate']*100:>6.1f}% "
                  f"{r['profit_factor']:>7.2f} {r['total_pnl']:>+8.2f}% {wl_ratio:>5.2f}x")
    else:
        print(f"\n  ** NO configurations pass Path B approval (PF>=1.8, W/L>=1.5, 10+ trades)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 100)
    print("WILLIAMS %R HOURLY UNDERPERFORMANCE ANALYSIS")
    print(f"Run at: {datetime.now().isoformat()}")
    print("=" * 100)

    # Load hourly data
    print("\n--- Loading Data ---")
    df = load_hourly_data()
    if df is None:
        return

    # Classify regimes
    print("\n--- Classifying Regimes ---")
    df = RegimeClassifier.classify_dataframe(df, min_warmup=50)
    df = df[df['regime'] != 'UNKNOWN'].reset_index(drop=True)

    regime_counts = df['regime'].value_counts()
    total = len(df)
    print(f"Total classified bars: {total:,}")
    for regime in ['TRENDING', 'VOLATILE', 'RANGING']:
        count = regime_counts.get(regime, 0)
        print(f"  {regime}: {count:,} bars ({count/total*100:.1f}%)")

    ranging_bars = regime_counts.get('RANGING', 0)
    print(f"\nRANGING bars available for trading: {ranging_bars:,} "
          f"({ranging_bars/24:.0f} days equivalent)")

    # Run all analyses
    analysis_1_trade_duration(df)
    analysis_2_oscillation_frequency(df)
    analysis_3_period_sensitivity(df)
    analysis_4_min_hold_time(df)
    analysis_5_exit_threshold(df)
    summary_best_configs(df)

    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
