"""Batch 1: Entry Filter Backtests — Weekly MTF + ADX Conviction

Tests on top of the current best (4x ATR + pyramid + bull filter):
  1. Weekly MTF: coin's weekly close > 21-week EMA
  2. ADX conviction: coin's ADX(14) > 22
  3. Both combined

Each variant is run over the full period and walk-forward OOS.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from backtest_donchian_daily import (
    fetch_all_coins,
    calculate_indicators,
    compute_stats,
    compute_per_coin_stats,
    DEFAULT_PARAMS,
    COIN_UNIVERSE,
)
from backtest_bull_filter import compute_btc_bull_filter
from backtest_walkforward import (
    split_coin_data,
    compute_sharpe,
    compute_max_drawdown,
    print_section,
    print_stats_row,
)


# ============================================================================
# CURRENT BEST: 4x ATR + PYRAMID (Phase 3 winner)
# ============================================================================

PHASE3_PARAMS = {
    **DEFAULT_PARAMS,
    'atr_mult': 4.0,
}

PYRAMID_GAIN_PCT = 15.0
PYRAMID_RISK_PCT = 0.01


# ============================================================================
# ADX COMPUTATION (Wilder's method for daily bars)
# ============================================================================

def compute_adx(df, period=14):
    """Compute ADX(14) using Wilder's EWM smoothing.

    Returns a Series aligned with df index.
    """
    high = df['high']
    low = df['low']
    close = df['close']

    # +DM and -DM
    high_diff = high.diff()
    low_diff = -low.diff()

    plus_dm = pd.Series(
        np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0),
        index=df.index,
    )

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Wilder's EWM smoothing (alpha = 1/period)
    alpha = 1.0 / period
    smoothed_tr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    smoothed_plus_dm = plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    smoothed_minus_dm = minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    # Directional Indicators
    plus_di = 100.0 * smoothed_plus_dm / smoothed_tr
    minus_di = 100.0 * smoothed_minus_dm / smoothed_tr

    # DX -> ADX
    di_sum = plus_di + minus_di
    di_sum = di_sum.replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    return adx


# ============================================================================
# WEEKLY MTF COMPUTATION
# ============================================================================

def compute_weekly_ema_filter(df, ema_weeks=21):
    """Compute weekly close > weekly EMA(21) filter.

    Resamples daily data to weekly, computes EMA, then maps back to
    daily dates. Returns a dict: date -> bool (True = weekly trend is up).
    """
    df = df.copy()
    df = df.set_index('time')

    # Resample to weekly (ending Friday)
    weekly = df.resample('W-FRI').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna()

    weekly['ema_21w'] = weekly['close'].ewm(span=ema_weeks, adjust=False).mean()
    weekly['weekly_bullish'] = weekly['close'] > weekly['ema_21w']

    # Map each daily date to the most recent completed weekly bar
    # A weekly bar ending on Friday applies to the NEXT week's trading
    # (i.e., for any daily date, use the most recent Friday <= that date)
    weekly_dates = weekly.index.tolist()
    weekly_lookup = {}
    for i, wdate in enumerate(weekly_dates):
        weekly_lookup[wdate.date()] = bool(weekly['weekly_bullish'].iloc[i])

    # Build daily lookup: for each daily date, find the most recent weekly date
    daily_filter = {}
    sorted_weekly = sorted(weekly_lookup.keys())

    for ddate in df.index:
        d = ddate.date() if hasattr(ddate, 'date') else ddate
        # Find most recent weekly date <= d
        best = None
        for wd in sorted_weekly:
            if wd <= d:
                best = wd
            else:
                break
        if best is not None:
            daily_filter[d] = weekly_lookup[best]

    return daily_filter


# ============================================================================
# PORTFOLIO BACKTEST WITH ALL FILTERS
# ============================================================================

def backtest_portfolio_filtered(coin_data, params, bull_filter,
                                 weekly_filters=None, adx_threshold=0,
                                 pyramiding=True, label=''):
    """Portfolio backtest with bull filter + optional weekly MTF + optional ADX filter.

    Args:
        coin_data: dict of symbol -> DataFrame
        params: strategy params dict
        bull_filter: dict date -> bool (BTC macro gate)
        weekly_filters: dict of symbol -> (dict date -> bool), or None to skip
        adx_threshold: minimum ADX(14) for entry (0 = disabled)
        pyramiding: enable pyramiding
        label: display label
    """
    cost_per_side = (params['fee_pct'] + params['slippage_pct']) / 100
    max_positions = params['max_positions']
    risk_pct = params['risk_per_trade_pct'] / 100
    capital = params['starting_capital']

    # Pre-calculate indicators + ADX
    prepared = {}
    for symbol, df in coin_data.items():
        df_ind = calculate_indicators(df, params)
        df_ind['adx'] = compute_adx(df_ind)
        df_ind = df_ind.dropna(subset=['donchian_high', 'atr', 'volume_sma', 'ema_21', 'rsi'])
        df_ind = df_ind.reset_index(drop=True)
        if len(df_ind) > 30:
            prepared[symbol] = df_ind

    # Build unified daily timeline
    all_dates = set()
    for symbol, df in prepared.items():
        all_dates.update(df['time'].dt.date.tolist())
    all_dates = sorted(all_dates)

    # Build lookups
    lookups = {}
    for symbol, df in prepared.items():
        lookup = {}
        for _, row in df.iterrows():
            lookup[row['time'].date()] = row
        lookups[symbol] = lookup

    prev_lookups = {}
    for symbol, df in prepared.items():
        prev_lookup = {}
        dates_list = sorted(df['time'].dt.date.tolist())
        for j in range(1, len(dates_list)):
            prev_lookup[dates_list[j]] = lookups[symbol].get(dates_list[j - 1])
        prev_lookups[symbol] = prev_lookup

    # State
    positions = {}
    trades = []
    equity_curve = []
    pyramid_adds = 0
    entries_blocked_weekly = 0
    entries_blocked_adx = 0

    for date in all_dates:
        is_bull = bull_filter.get(date, False)

        # === EXITS (same as Phase 3 — no filter changes to exit logic) ===
        symbols_to_close = []
        for symbol, pos in list(positions.items()):
            row = lookups[symbol].get(date)
            if row is None:
                continue

            current_close = float(row['close'])
            current_high = float(row['high'])
            current_atr = float(row['atr'])
            pos['high_watermark'] = max(pos['high_watermark'], current_high)

            exit_reason = None

            # Blow-off detection
            vol_sma = float(row['volume_sma']) if row['volume_sma'] > 0 else 1
            volume_ratio = float(row['volume']) / vol_sma
            is_blowoff = (volume_ratio > params['volume_blowoff']
                          and float(row['rsi']) > params['rsi_blowoff'])
            stop_mult = params['atr_mult_tight'] if is_blowoff else params['atr_mult']

            # 1. Trailing stop
            trailing_stop = pos['high_watermark'] - (stop_mult * current_atr)
            if current_close <= trailing_stop:
                exit_reason = f'Trailing stop ({stop_mult}x ATR)'

            # 2. Donchian exit
            prev_row = prev_lookups[symbol].get(date)
            if not exit_reason and prev_row is not None and pd.notna(prev_row['exit_low']):
                if current_close < float(prev_row['exit_low']):
                    exit_reason = 'Donchian exit (10-day low)'

            # 3. Emergency stop (15%)
            if not exit_reason and current_close <= pos['entry_price'] * 0.85:
                exit_reason = 'Emergency stop (15%)'

            # Partial profit taking
            if not exit_reason:
                gain_pct = ((current_close - pos['entry_price']) / pos['entry_price']) * 100
                if pos['partials_taken'] == 0 and gain_pct >= params['tp1_pct']:
                    partial_price = current_close * (1 - cost_per_side)
                    partial_pnl = ((partial_price - pos['entry_price']) / pos['entry_price']) * 100
                    partial_size = pos['size_usd'] * params['tp1_fraction']
                    partial_gain = partial_size * (partial_pnl / 100)
                    capital += partial_size + partial_gain
                    pos['size_usd'] -= partial_size
                    pos['partials_taken'] = 1
                    trades.append({
                        'symbol': symbol, 'entry_time': pos['entry_time'],
                        'entry_price': pos['entry_price'], 'exit_time': row['time'],
                        'exit_price': partial_price, 'pnl_pct': partial_pnl,
                        'exit_reason': f'Partial TP1 (+{params["tp1_pct"]}%)',
                        'size_usd': partial_size, 'win': True,
                    })
                elif pos['partials_taken'] == 1 and gain_pct >= params['tp2_pct']:
                    partial_price = current_close * (1 - cost_per_side)
                    partial_pnl = ((partial_price - pos['entry_price']) / pos['entry_price']) * 100
                    partial_size = pos['size_usd'] * params['tp2_fraction']
                    partial_gain = partial_size * (partial_pnl / 100)
                    capital += partial_size + partial_gain
                    pos['size_usd'] -= partial_size
                    pos['partials_taken'] = 2
                    trades.append({
                        'symbol': symbol, 'entry_time': pos['entry_time'],
                        'entry_price': pos['entry_price'], 'exit_time': row['time'],
                        'exit_price': partial_price, 'pnl_pct': partial_pnl,
                        'exit_reason': f'Partial TP2 (+{params["tp2_pct"]}%)',
                        'size_usd': partial_size, 'win': True,
                    })

            if exit_reason:
                exit_price_adj = current_close * (1 - cost_per_side)
                pnl_pct = ((exit_price_adj - pos['entry_price']) / pos['entry_price']) * 100
                pnl_usd = pos['size_usd'] * (pnl_pct / 100)
                capital += pos['size_usd'] + pnl_usd
                trades.append({
                    'symbol': symbol, 'entry_time': pos['entry_time'],
                    'entry_price': pos['entry_price'], 'exit_time': row['time'],
                    'exit_price': exit_price_adj, 'pnl_pct': pnl_pct,
                    'exit_reason': exit_reason, 'size_usd': pos['size_usd'],
                    'win': pnl_pct > 0,
                })
                symbols_to_close.append(symbol)

        for sym in symbols_to_close:
            del positions[sym]

        # === PYRAMIDING ===
        if pyramiding and is_bull:
            for symbol, pos in list(positions.items()):
                if pos.get('pyramided'):
                    continue
                row = lookups[symbol].get(date)
                prev_row = prev_lookups[symbol].get(date)
                if row is None or prev_row is None:
                    continue

                current_close = float(row['close'])
                gain_pct = ((current_close - pos['entry_price']) / pos['entry_price']) * 100
                new_high = pd.notna(prev_row['donchian_high']) and current_close > float(prev_row['donchian_high'])

                if gain_pct >= PYRAMID_GAIN_PCT and new_high:
                    total_equity = capital + sum(p['size_usd'] for p in positions.values())
                    add_risk = total_equity * PYRAMID_RISK_PCT
                    atr_val = float(row['atr'])
                    stop_distance = params['atr_mult'] * atr_val
                    stop_pct = stop_distance / current_close

                    if stop_pct > 0:
                        add_size = add_risk / stop_pct
                    else:
                        add_size = total_equity * 0.05

                    add_size = min(add_size, capital * 0.50)
                    if add_size >= 100:
                        capital -= add_size
                        pos['size_usd'] += add_size
                        pos['pyramided'] = True
                        pyramid_adds += 1

                        trades.append({
                            'symbol': symbol, 'entry_time': row['time'],
                            'entry_price': current_close * (1 + cost_per_side),
                            'exit_time': row['time'], 'exit_price': current_close,
                            'pnl_pct': 0,
                            'exit_reason': f'Pyramid add (+{gain_pct:.0f}%, new 20d high)',
                            'size_usd': add_size, 'win': True,
                        })

        # === NEW ENTRIES (bull filter + optional weekly MTF + optional ADX) ===
        if len(positions) < max_positions and is_bull:
            for symbol in prepared:
                if symbol in positions:
                    continue
                if len(positions) >= max_positions:
                    break

                row = lookups[symbol].get(date)
                prev_row = prev_lookups[symbol].get(date)
                if row is None or prev_row is None:
                    continue
                if pd.isna(prev_row['donchian_high']):
                    continue

                current_close = float(row['close'])

                # Standard Donchian entry checks
                breakout = current_close > float(prev_row['donchian_high'])
                if params['volume_mult'] > 0:
                    vol_sma = float(row['volume_sma']) if row['volume_sma'] > 0 else 1
                    volume_ok = float(row['volume']) > params['volume_mult'] * vol_sma
                else:
                    volume_ok = True
                trend_ok = current_close > float(row['ema_21'])

                if not (breakout and volume_ok and trend_ok):
                    continue

                # --- NEW FILTER: Weekly MTF ---
                if weekly_filters and symbol in weekly_filters:
                    weekly_ok = weekly_filters[symbol].get(date, True)
                    if not weekly_ok:
                        entries_blocked_weekly += 1
                        continue

                # --- NEW FILTER: ADX conviction ---
                if adx_threshold > 0:
                    coin_adx = float(row['adx']) if pd.notna(row['adx']) else 0
                    if coin_adx < adx_threshold:
                        entries_blocked_adx += 1
                        continue

                # Entry!
                total_equity = capital + sum(p['size_usd'] for p in positions.values())
                risk_amount = total_equity * risk_pct
                entry_price = current_close * (1 + cost_per_side)
                atr_val = float(row['atr'])
                stop_distance = params['atr_mult'] * atr_val
                stop_pct = stop_distance / entry_price

                if stop_pct > 0:
                    position_size = risk_amount / stop_pct
                else:
                    position_size = total_equity / max_positions

                position_size = min(position_size, capital * 0.95)
                if position_size < 100:
                    continue

                capital -= position_size
                positions[symbol] = {
                    'entry_price': entry_price,
                    'entry_time': row['time'],
                    'high_watermark': float(row['high']),
                    'partials_taken': 0,
                    'remaining_fraction': 1.0,
                    'size_usd': position_size,
                    'pyramided': False,
                }

        # Equity tracking
        total_equity = capital
        for symbol, pos in positions.items():
            row = lookups[symbol].get(date)
            if row is not None:
                current_val = pos['size_usd'] * (float(row['close']) / pos['entry_price'])
                total_equity += current_val
            else:
                total_equity += pos['size_usd']
        equity_curve.append({'date': date, 'equity': total_equity})

    # Close remaining
    for symbol, pos in list(positions.items()):
        df = prepared[symbol]
        last = df.iloc[-1]
        exit_price_adj = float(last['close']) * (1 - cost_per_side)
        pnl_pct = ((exit_price_adj - pos['entry_price']) / pos['entry_price']) * 100
        pnl_usd = pos['size_usd'] * (pnl_pct / 100)
        capital += pos['size_usd'] + pnl_usd
        trades.append({
            'symbol': symbol, 'entry_time': pos['entry_time'],
            'entry_price': pos['entry_price'], 'exit_time': last['time'],
            'exit_price': exit_price_adj, 'pnl_pct': pnl_pct,
            'exit_reason': 'End of backtest', 'size_usd': pos['size_usd'],
            'win': pnl_pct > 0,
        })

    return trades, equity_curve, capital, pyramid_adds, entries_blocked_weekly, entries_blocked_adx


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 100)
    print("BATCH 1: ENTRY FILTER BACKTESTS — WEEKLY MTF + ADX CONVICTION")
    print("  All variants built on top of 4x ATR + pyramid + bull filter (Phase 3 winner)")
    print("=" * 100)

    coin_data = fetch_all_coins()
    btc_df = coin_data.get('BTC-USD')
    if btc_df is None:
        print("ERROR: No BTC data")
        return

    print(f"\n  Computing BTC bull filter...")
    bull_filter = compute_btc_bull_filter(btc_df)

    print(f"\n  Computing weekly MTF filters for each coin...")
    weekly_filters = {}
    for symbol, df in coin_data.items():
        weekly_filters[symbol] = compute_weekly_ema_filter(df)
        bull_weeks = sum(1 for v in weekly_filters[symbol].values() if v)
        total_weeks = len(weekly_filters[symbol])
        print(f"    {symbol}: {bull_weeks}/{total_weeks} days with weekly trend UP "
              f"({bull_weeks/total_weeks*100:.0f}%)" if total_weeks > 0 else f"    {symbol}: no data")

    # ======================================================================
    # VARIANT DEFINITIONS
    # ======================================================================

    # All variants use 4x ATR + pyramid (current production)
    variants = [
        {
            'label': 'Baseline (4x ATR + pyramid)',
            'weekly': False,
            'adx_threshold': 0,
        },
        {
            'label': '+ Weekly MTF (21w EMA)',
            'weekly': True,
            'adx_threshold': 0,
        },
        {
            'label': '+ ADX > 22',
            'weekly': False,
            'adx_threshold': 22,
        },
        {
            'label': '+ ADX > 25',
            'weekly': False,
            'adx_threshold': 25,
        },
        {
            'label': '+ Weekly MTF + ADX > 22',
            'weekly': True,
            'adx_threshold': 22,
        },
        {
            'label': '+ Weekly MTF + ADX > 25',
            'weekly': True,
            'adx_threshold': 25,
        },
    ]

    # ======================================================================
    # FULL PERIOD COMPARISON
    # ======================================================================
    print_section("FULL PERIOD COMPARISON (2022-2026)")

    results = []
    for v in variants:
        wf = weekly_filters if v['weekly'] else None
        trades, eq, cap, p_adds, blocked_w, blocked_a = backtest_portfolio_filtered(
            coin_data, PHASE3_PARAMS, bull_filter,
            weekly_filters=wf,
            adx_threshold=v['adx_threshold'],
            pyramiding=True,
            label=v['label'],
        )
        stats = compute_stats(trades, v['label'])
        results.append({
            'label': v['label'],
            'stats': stats,
            'eq': eq,
            'pyramid_adds': p_adds,
            'blocked_weekly': blocked_w,
            'blocked_adx': blocked_a,
        })

    print(f"\n\n  {'='*130}")
    print(f"  FULL PERIOD RESULTS")
    print(f"  {'='*130}")
    for r in results:
        print_stats_row(r['label'], r['stats'], r['eq'])
        extras = []
        if r['pyramid_adds'] > 0:
            extras.append(f"pyramid adds: {r['pyramid_adds']}")
        if r['blocked_weekly'] > 0:
            extras.append(f"blocked by weekly: {r['blocked_weekly']}")
        if r['blocked_adx'] > 0:
            extras.append(f"blocked by ADX: {r['blocked_adx']}")
        if extras:
            print(f"      ({', '.join(extras)})")

    # Per-coin for baseline and best
    best_idx = max(range(len(results)),
                   key=lambda i: results[i]['stats'].get('total_return_pct', 0))
    if best_idx != 0:
        print(f"\n  Best variant: {results[best_idx]['label']} "
              f"({results[best_idx]['stats'].get('total_return_pct', 0):+.1f}% return)")
    else:
        print(f"\n  No filter variant beat the baseline in full period.")

    # ======================================================================
    # WALK-FORWARD ON ALL VARIANTS
    # ======================================================================
    print_section("WALK-FORWARD (Train 2022-2024, Test 2025-2026)")

    cutoff = pd.Timestamp('2025-01-01')
    train_data, test_data = split_coin_data(coin_data, cutoff)

    btc_train = train_data.get('BTC-USD')
    btc_test = test_data.get('BTC-USD')

    print(f"\n  Computing period-specific bull filters...")
    train_bull = compute_btc_bull_filter(btc_train) if btc_train is not None else {}
    test_bull = compute_btc_bull_filter(btc_test) if btc_test is not None else {}

    print(f"\n  Computing period-specific weekly MTF filters...")
    train_weekly = {}
    test_weekly = {}
    for symbol in coin_data:
        if symbol in train_data:
            train_weekly[symbol] = compute_weekly_ema_filter(train_data[symbol])
        if symbol in test_data:
            test_weekly[symbol] = compute_weekly_ema_filter(test_data[symbol])

    print(f"\n  Running walk-forward on all variants...")

    wf_results = []
    for v in variants:
        # Train
        wf_train = train_weekly if v['weekly'] else None
        train_trades, train_eq, _, _, _, _ = backtest_portfolio_filtered(
            train_data, PHASE3_PARAMS, train_bull,
            weekly_filters=wf_train,
            adx_threshold=v['adx_threshold'],
            pyramiding=True, label=v['label'],
        )
        train_stats = compute_stats(train_trades, f'{v["label"]} [train]')

        # Test (OOS)
        wf_test = test_weekly if v['weekly'] else None
        test_trades, test_eq, _, t_adds, t_bw, t_ba = backtest_portfolio_filtered(
            test_data, PHASE3_PARAMS, test_bull,
            weekly_filters=wf_test,
            adx_threshold=v['adx_threshold'],
            pyramiding=True, label=v['label'],
        )
        test_stats = compute_stats(test_trades, f'{v["label"]} [test]')

        wf_results.append({
            'label': v['label'],
            'train_stats': train_stats,
            'train_eq': train_eq,
            'test_stats': test_stats,
            'test_eq': test_eq,
            'test_adds': t_adds,
            'test_blocked_w': t_bw,
            'test_blocked_a': t_ba,
        })

    print(f"\n\n  {'='*130}")
    print(f"  WALK-FORWARD: TRAIN (2022-2024)")
    print(f"  {'='*130}")
    for r in wf_results:
        print_stats_row(r['label'], r['train_stats'], r['train_eq'])

    print(f"\n  {'='*130}")
    print(f"  WALK-FORWARD: TEST / OUT-OF-SAMPLE (2025-2026)")
    print(f"  {'='*130}")
    for r in wf_results:
        print_stats_row(r['label'], r['test_stats'], r['test_eq'])
        extras = []
        if r['test_adds'] > 0:
            extras.append(f"pyramid adds: {r['test_adds']}")
        if r['test_blocked_w'] > 0:
            extras.append(f"blocked by weekly: {r['test_blocked_w']}")
        if r['test_blocked_a'] > 0:
            extras.append(f"blocked by ADX: {r['test_blocked_a']}")
        if extras:
            print(f"      ({', '.join(extras)})")

    # ======================================================================
    # PER-COIN BREAKDOWN FOR BEST OOS VARIANT
    # ======================================================================
    best_oos_idx = max(range(len(wf_results)),
                       key=lambda i: wf_results[i]['test_stats'].get('total_return_pct', 0))
    best_oos = wf_results[best_oos_idx]

    # Re-run best OOS to get per-coin trades
    best_v = variants[best_oos_idx]
    wf_test_best = test_weekly if best_v['weekly'] else None
    best_test_trades, _, _, _, _, _ = backtest_portfolio_filtered(
        test_data, PHASE3_PARAMS, test_bull,
        weekly_filters=wf_test_best,
        adx_threshold=best_v['adx_threshold'],
        pyramiding=True, label=best_v['label'],
    )
    best_coin_stats = compute_per_coin_stats(best_test_trades)

    if best_coin_stats:
        print(f"\n  {'='*80}")
        print(f"  PER-COIN OOS (2025-2026) — {best_oos['label']}:")
        print(f"  {'='*80}")
        print(f"  {'Coin':<12s} {'Trades':>7s} {'Wins':>5s} {'WR':>7s} {'PF':>7s} {'Sum P&L':>10s}")
        print(f"  {'-'*55}")
        for sym, cs in sorted(best_coin_stats.items(), key=lambda x: -x[1]['sum_pnl_pct']):
            print(f"  {sym:<12s} {cs['trades']:>7d} {cs['wins']:>5d} "
                  f"{cs['win_rate']*100:>6.1f}% {cs['profit_factor']:>7.2f} "
                  f"{cs['sum_pnl_pct']:>+9.2f}%")

    # ======================================================================
    # VERDICT
    # ======================================================================
    print_section("BATCH 1 VERDICT")

    baseline_full = results[0]
    baseline_oos = wf_results[0]
    best_full = max(results, key=lambda r: r['stats'].get('total_return_pct', 0))
    best_oos_r = wf_results[best_oos_idx]

    print(f"\n  FULL PERIOD:")
    print(f"    Baseline:    {baseline_full['stats'].get('total_return_pct',0):+.1f}% return, "
          f"PF {baseline_full['stats'].get('profit_factor',0):.2f}, "
          f"WR {baseline_full['stats'].get('win_rate',0)*100:.1f}%")
    print(f"    Best:        {best_full['stats'].get('total_return_pct',0):+.1f}% return, "
          f"PF {best_full['stats'].get('profit_factor',0):.2f}, "
          f"WR {best_full['stats'].get('win_rate',0)*100:.1f}% — {best_full['label']}")

    print(f"\n  OUT-OF-SAMPLE (2025-2026):")
    print(f"    Baseline:    {baseline_oos['test_stats'].get('total_return_pct',0):+.1f}% return, "
          f"PF {baseline_oos['test_stats'].get('profit_factor',0):.2f}, "
          f"WR {baseline_oos['test_stats'].get('win_rate',0)*100:.1f}%, "
          f"MaxDD {compute_max_drawdown(baseline_oos['test_eq'])[0]:.1f}%")
    print(f"    Best OOS:    {best_oos_r['test_stats'].get('total_return_pct',0):+.1f}% return, "
          f"PF {best_oos_r['test_stats'].get('profit_factor',0):.2f}, "
          f"WR {best_oos_r['test_stats'].get('win_rate',0)*100:.1f}%, "
          f"MaxDD {compute_max_drawdown(best_oos_r['test_eq'])[0]:.1f}% — {best_oos_r['label']}")

    # Improvement check
    baseline_oos_ret = baseline_oos['test_stats'].get('total_return_pct', 0)
    best_oos_ret = best_oos_r['test_stats'].get('total_return_pct', 0)
    improvement = best_oos_ret - baseline_oos_ret

    print(f"\n  OOS improvement over current production: {improvement:+.1f}%")

    if best_oos_ret > baseline_oos_ret and best_oos_r['test_stats'].get('profit_factor', 0) > baseline_oos['test_stats'].get('profit_factor', 0):
        print(f"  VERDICT: IMPROVEMENT — {best_oos_r['label']} beats baseline in OOS")
        print(f"  Consider deploying this filter to production.")
    elif best_oos_ret > baseline_oos_ret:
        print(f"  VERDICT: MARGINAL — Higher return but mixed metrics")
    else:
        print(f"  VERDICT: NO IMPROVEMENT — Current production config remains optimal")
        print(f"  The additional filters did not improve out-of-sample performance.")


if __name__ == "__main__":
    main()
