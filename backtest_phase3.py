"""Phase 3: Pyramiding & Exit Tuning — Backtest

Tests on top of the bull-filter baseline:
  1. Pyramiding: Add 1% risk tranche to winners at +15% if new 20-day high
  2. Wider trailing stop: 4x ATR instead of 3x
  3. Adjusted partials: 20%/20%/60% runner instead of 25%/25%/50%
  4. All combined

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
# PARAM VARIANTS
# ============================================================================

BASELINE = {
    **DEFAULT_PARAMS,
    'label': 'Baseline (bull filter)',
}

WIDER_STOP = {
    **DEFAULT_PARAMS,
    'label': '4x ATR trailing',
    'atr_mult': 4.0,
}

ADJUSTED_PARTIALS = {
    **DEFAULT_PARAMS,
    'label': 'Partials 20/20/60',
    'tp1_fraction': 0.20,
    'tp2_fraction': 0.20,
}

WIDER_AND_PARTIALS = {
    **DEFAULT_PARAMS,
    'label': '4x ATR + 20/20/60',
    'atr_mult': 4.0,
    'tp1_fraction': 0.20,
    'tp2_fraction': 0.20,
}

# Pyramiding params (used as flags, not in DEFAULT_PARAMS)
PYRAMID_GAIN_PCT = 15.0    # add to winner when up +15%
PYRAMID_RISK_PCT = 0.01    # 1% equity risk on the add-on tranche


# ============================================================================
# PORTFOLIO BACKTEST WITH PYRAMIDING + BULL FILTER
# ============================================================================

def backtest_portfolio_phase3(coin_data, params, bull_filter, pyramiding=False):
    """Portfolio backtest with bull filter, optional pyramiding, configurable exits."""
    label = params['label']
    if pyramiding:
        label += ' + pyramid'
    cost_per_side = (params['fee_pct'] + params['slippage_pct']) / 100
    max_positions = params['max_positions']
    risk_pct = params['risk_per_trade_pct'] / 100
    capital = params['starting_capital']

    # Pre-calculate indicators
    prepared = {}
    for symbol, df in coin_data.items():
        df_ind = calculate_indicators(df, params)
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
            prev_lookup[dates_list[j]] = lookups[symbol].get(dates_list[j-1])
        prev_lookups[symbol] = prev_lookup

    # State
    positions = {}  # symbol -> pos dict
    trades = []
    equity_curve = []
    pyramid_adds = 0

    for date in all_dates:
        is_bull = bull_filter.get(date, False)

        # === EXITS ===
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
                        'symbol': symbol,
                        'entry_time': pos['entry_time'],
                        'entry_price': pos['entry_price'],
                        'exit_time': row['time'],
                        'exit_price': partial_price,
                        'pnl_pct': partial_pnl,
                        'exit_reason': f'Partial TP1 (+{params["tp1_pct"]}%)',
                        'size_usd': partial_size,
                        'win': True,
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
                        'symbol': symbol,
                        'entry_time': pos['entry_time'],
                        'entry_price': pos['entry_price'],
                        'exit_time': row['time'],
                        'exit_price': partial_price,
                        'pnl_pct': partial_pnl,
                        'exit_reason': f'Partial TP2 (+{params["tp2_pct"]}%)',
                        'size_usd': partial_size,
                        'win': True,
                    })

            if exit_reason:
                exit_price_adj = current_close * (1 - cost_per_side)
                pnl_pct = ((exit_price_adj - pos['entry_price']) / pos['entry_price']) * 100
                pnl_usd = pos['size_usd'] * (pnl_pct / 100)
                capital += pos['size_usd'] + pnl_usd
                trades.append({
                    'symbol': symbol,
                    'entry_time': pos['entry_time'],
                    'entry_price': pos['entry_price'],
                    'exit_time': row['time'],
                    'exit_price': exit_price_adj,
                    'pnl_pct': pnl_pct,
                    'exit_reason': exit_reason,
                    'size_usd': pos['size_usd'],
                    'win': pnl_pct > 0,
                })
                symbols_to_close.append(symbol)

        for sym in symbols_to_close:
            del positions[sym]

        # === PYRAMIDING (add to existing winners) ===
        if pyramiding and is_bull:
            for symbol, pos in list(positions.items()):
                if pos.get('pyramided'):
                    continue  # only one add per position

                row = lookups[symbol].get(date)
                prev_row = prev_lookups[symbol].get(date)
                if row is None or prev_row is None:
                    continue

                current_close = float(row['close'])
                gain_pct = ((current_close - pos['entry_price']) / pos['entry_price']) * 100

                # Pyramid condition: up +15% AND making new 20-day high
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

                    add_size = min(add_size, capital * 0.50)  # don't use more than half remaining cash
                    if add_size >= 100:
                        capital -= add_size
                        pos['size_usd'] += add_size
                        pos['pyramided'] = True
                        pyramid_adds += 1

                        trades.append({
                            'symbol': symbol,
                            'entry_time': row['time'],
                            'entry_price': current_close * (1 + cost_per_side),
                            'exit_time': row['time'],
                            'exit_price': current_close,
                            'pnl_pct': 0,
                            'exit_reason': f'Pyramid add (+{gain_pct:.0f}%, new 20d high)',
                            'size_usd': add_size,
                            'win': True,
                        })

        # === NEW ENTRIES (gated by bull filter) ===
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

                breakout = current_close > float(prev_row['donchian_high'])
                if params['volume_mult'] > 0:
                    vol_sma = float(row['volume_sma']) if row['volume_sma'] > 0 else 1
                    volume_ok = float(row['volume']) > params['volume_mult'] * vol_sma
                else:
                    volume_ok = True
                trend_ok = current_close > float(row['ema_21'])

                if breakout and volume_ok and trend_ok:
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
            'symbol': symbol,
            'entry_time': pos['entry_time'],
            'entry_price': pos['entry_price'],
            'exit_time': last['time'],
            'exit_price': exit_price_adj,
            'pnl_pct': pnl_pct,
            'exit_reason': 'End of backtest',
            'size_usd': pos['size_usd'],
            'win': pnl_pct > 0,
        })

    return trades, equity_curve, capital, pyramid_adds


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 100)
    print("PHASE 3: PYRAMIDING & EXIT TUNING")
    print("  All variants include the bull filter from Phase 2")
    print("=" * 100)

    coin_data = fetch_all_coins()
    btc_df = coin_data.get('BTC-USD')
    if btc_df is None:
        print("ERROR: No BTC data")
        return

    print(f"\n  Computing BTC bull filter...")
    bull_filter = compute_btc_bull_filter(btc_df)

    # ======================================================================
    # FULL PERIOD COMPARISON
    # ======================================================================
    print_section("FULL PERIOD COMPARISON (2022-2026)")

    variants = [
        ('Baseline (bull filter)',    BASELINE,           False),
        ('4x ATR trailing',           WIDER_STOP,         False),
        ('Partials 20/20/60',         ADJUSTED_PARTIALS,  False),
        ('4x ATR + 20/20/60',         WIDER_AND_PARTIALS, False),
        ('Baseline + pyramid',        BASELINE,           True),
        ('4x ATR + pyramid',          WIDER_STOP,         True),
        ('4x + 20/20/60 + pyramid',   WIDER_AND_PARTIALS, True),
    ]

    results = []
    for name, params, pyramid in variants:
        trades, eq, cap, p_adds = backtest_portfolio_phase3(
            coin_data, params, bull_filter, pyramiding=pyramid)
        stats = compute_stats(trades, name)
        results.append((name, stats, eq, p_adds))

    print(f"\n\n  {'='*130}")
    print(f"  FULL PERIOD RESULTS")
    print(f"  {'='*130}")
    for name, stats, eq, p_adds in results:
        suffix = f"  (pyramid adds: {p_adds})" if p_adds > 0 else ""
        print_stats_row(name, stats, eq)
        if suffix:
            print(f"    {suffix}")

    # Per-coin for best variant
    best_idx = max(range(len(results)), key=lambda i: results[i][1].get('total_return_pct', 0))
    best_name, best_stats, best_eq, _ = results[best_idx]
    print(f"\n  Best variant: {best_name} ({best_stats.get('total_return_pct', 0):+.1f}% return)")

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

    print(f"\n  Running walk-forward on all variants...")

    wf_results = []
    for name, params, pyramid in variants:
        # Train
        train_trades, train_eq, _, _ = backtest_portfolio_phase3(
            train_data, params, train_bull, pyramiding=pyramid)
        train_stats = compute_stats(train_trades, f'{name} [train]')

        # Test (OOS)
        test_trades, test_eq, _, t_adds = backtest_portfolio_phase3(
            test_data, params, test_bull, pyramiding=pyramid)
        test_stats = compute_stats(test_trades, f'{name} [test]')

        wf_results.append((name, train_stats, train_eq, test_stats, test_eq, t_adds))

    print(f"\n\n  {'='*130}")
    print(f"  WALK-FORWARD: TRAIN (2022-2024)")
    print(f"  {'='*130}")
    for name, train_stats, train_eq, _, _, _ in wf_results:
        print_stats_row(name, train_stats, train_eq)

    print(f"\n  {'='*130}")
    print(f"  WALK-FORWARD: TEST / OUT-OF-SAMPLE (2025-2026)")
    print(f"  {'='*130}")
    for name, _, _, test_stats, test_eq, t_adds in wf_results:
        suffix = f"  (pyramid adds: {t_adds})" if t_adds > 0 else ""
        print_stats_row(name, test_stats, test_eq)
        if suffix:
            print(f"    {suffix}")

    # ======================================================================
    # VERDICT
    # ======================================================================
    print_section("PHASE 3 VERDICT")

    # Find best full-period variant
    best_full = max(results, key=lambda r: r[1].get('total_return_pct', 0))
    # Find best OOS variant
    best_oos = max(wf_results, key=lambda r: r[3].get('total_return_pct', 0))

    print(f"\n  Best full-period: {best_full[0]}")
    print(f"    Return: {best_full[1].get('total_return_pct', 0):+.1f}%  "
          f"PF: {best_full[1].get('profit_factor', 0):.2f}  "
          f"WR: {best_full[1].get('win_rate', 0)*100:.1f}%  "
          f"MaxDD: {compute_max_drawdown(best_full[2])[0]:.1f}%")

    print(f"\n  Best out-of-sample: {best_oos[0]}")
    print(f"    Return: {best_oos[3].get('total_return_pct', 0):+.1f}%  "
          f"PF: {best_oos[3].get('profit_factor', 0):.2f}  "
          f"WR: {best_oos[3].get('win_rate', 0)*100:.1f}%  "
          f"MaxDD: {compute_max_drawdown(best_oos[4])[0]:.1f}%")

    # Compare baseline OOS vs best OOS
    baseline_oos = wf_results[0][3]
    best_oos_stats = best_oos[3]
    baseline_ret = baseline_oos.get('total_return_pct', 0)
    best_ret = best_oos_stats.get('total_return_pct', 0)
    improvement = best_ret - baseline_ret

    print(f"\n  OOS improvement over baseline: {improvement:+.1f}%")

    if best_ret > baseline_ret and best_oos_stats.get('profit_factor', 0) > baseline_oos.get('profit_factor', 0):
        print(f"  VERDICT: IMPROVEMENT — {best_oos[0]} beats baseline in OOS")
    elif best_ret > baseline_ret:
        print(f"  VERDICT: MARGINAL — Higher return but mixed metrics")
    else:
        print(f"  VERDICT: NO IMPROVEMENT — Baseline remains best for OOS")

    print(f"\n  Recommendation: Deploy whichever variant has the best risk-adjusted OOS performance")
    print(f"  (prioritize PF and MaxDD over raw return)")


if __name__ == "__main__":
    main()
