"""Analyze hourly vs daily indicator distributions for regime recalibration.

Goal: The current RegimeClassifier uses ATR/close > 3% for VOLATILE and ADX > 25 for TRENDING.
These thresholds were set for daily candles. On hourly candles, ATR/close is much smaller,
so VOLATILE almost never triggers. This script analyzes the actual distributions to find
proper hourly thresholds that produce comparable regime splits.
"""
import pandas as pd
import numpy as np
import json
import os
from market_regime import RegimeClassifier


def load_data(mode):
    """Load cached data"""
    if mode == 'hourly':
        cache_file = 'btc_4year_hourly_cache.json'
    else:
        cache_file = 'btc_4year_cache.json'

    if not os.path.exists(cache_file):
        print(f"Cache not found: {cache_file}")
        return None

    with open(cache_file, 'r') as f:
        cache_data = json.load(f)
    df = pd.DataFrame(cache_data['data'])
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    print(f"Loaded {len(df):,} {mode} candles ({df['time'].min().date()} to {df['time'].max().date()})")
    return df


def compute_raw_indicators(df):
    """Compute indicators without classifying — just raw values"""
    df = df.copy()
    period = 14

    # ATR
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    # Volatility %
    df['volatility_pct'] = (df['atr'] / df['close']) * 100

    # +DI / -DI / ADX
    high_diff = df['high'].diff()
    low_diff = -df['low'].diff()
    plus_dm = pd.Series(
        np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0),
        index=df.index
    )
    minus_dm = pd.Series(
        np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0),
        index=df.index
    )
    smooth_plus_dm = plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    smooth_minus_dm = minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    df['plus_di'] = 100 * smooth_plus_dm / df['atr']
    df['minus_di'] = 100 * smooth_minus_dm / df['atr']
    dx = 100 * (df['plus_di'] - df['minus_di']).abs() / (df['plus_di'] + df['minus_di'])
    df['adx'] = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    # Trend direction
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['trend_direction'] = np.where(
        (df['close'] > df['sma_20']) & (df['sma_20'] > df['sma_50']), 'UPTREND',
        np.where(
            (df['close'] < df['sma_20']) & (df['sma_20'] < df['sma_50']), 'DOWNTREND',
            'SIDEWAYS'
        )
    )

    df = df.dropna().reset_index(drop=True)
    return df


def print_distribution(values, name, percentiles=[5, 10, 25, 50, 75, 90, 95]):
    """Print distribution stats for a series"""
    print(f"\n  {name}:")
    print(f"    Mean:   {values.mean():.4f}")
    print(f"    Std:    {values.std():.4f}")
    print(f"    Min:    {values.min():.4f}")
    print(f"    Max:    {values.max():.4f}")
    for p in percentiles:
        print(f"    P{p:<3d}:   {values.quantile(p/100):.4f}")


def analyze_regime_thresholds(df_daily, df_hourly):
    """Compare distributions and find equivalent thresholds"""

    print("=" * 100)
    print("INDICATOR DISTRIBUTION ANALYSIS — DAILY vs HOURLY")
    print("=" * 100)

    # =====================================================================
    # ATR/Close (Volatility %)
    # =====================================================================
    print("\n" + "=" * 100)
    print("VOLATILITY (ATR/Close %) — Current threshold: 3.0% (daily)")
    print("=" * 100)

    print("\n  DAILY candles:")
    print_distribution(df_daily['volatility_pct'], 'volatility_pct')
    daily_above_3 = (df_daily['volatility_pct'] > 3.0).sum()
    daily_total = len(df_daily)
    print(f"\n    Bars with volatility > 3.0%: {daily_above_3} / {daily_total} ({daily_above_3/daily_total*100:.1f}%)")

    print("\n  HOURLY candles:")
    print_distribution(df_hourly['volatility_pct'], 'volatility_pct')
    hourly_above_3 = (df_hourly['volatility_pct'] > 3.0).sum()
    hourly_total = len(df_hourly)
    print(f"\n    Bars with volatility > 3.0%: {hourly_above_3} / {hourly_total} ({hourly_above_3/hourly_total*100:.1f}%)")

    # Find hourly threshold that gives same % as daily 3%
    daily_volatile_pct = daily_above_3 / daily_total
    hourly_equiv_threshold = df_hourly['volatility_pct'].quantile(1 - daily_volatile_pct)
    print(f"\n  ** Daily 3.0% captures {daily_volatile_pct*100:.1f}% of bars as VOLATILE")
    print(f"  ** Equivalent hourly threshold (same %): {hourly_equiv_threshold:.4f}%")

    # Show what various thresholds would give on hourly
    print(f"\n  Hourly volatility threshold sweep:")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0]:
        count = (df_hourly['volatility_pct'] > thresh).sum()
        print(f"    > {thresh:.1f}%: {count:>6,} bars ({count/hourly_total*100:>5.1f}%)")

    # =====================================================================
    # ADX
    # =====================================================================
    print("\n" + "=" * 100)
    print("ADX — Current threshold: 25 (same for daily and hourly)")
    print("=" * 100)

    print("\n  DAILY candles:")
    print_distribution(df_daily['adx'], 'ADX')
    daily_above_25 = (df_daily['adx'] > 25).sum()
    print(f"\n    Bars with ADX > 25: {daily_above_25} / {daily_total} ({daily_above_25/daily_total*100:.1f}%)")

    print("\n  HOURLY candles:")
    print_distribution(df_hourly['adx'], 'ADX')
    hourly_above_25 = (df_hourly['adx'] > 25).sum()
    print(f"\n    Bars with ADX > 25: {hourly_above_25} / {hourly_total} ({hourly_above_25/hourly_total*100:.1f}%)")

    # ADX threshold sweep
    print(f"\n  Hourly ADX threshold sweep:")
    for thresh in [15, 18, 20, 22, 25, 28, 30, 35, 40]:
        count = (df_hourly['adx'] > thresh).sum()
        print(f"    > {thresh}: {count:>6,} bars ({count/hourly_total*100:>5.1f}%)")

    # =====================================================================
    # Combined regime simulation with various thresholds
    # =====================================================================
    print("\n" + "=" * 100)
    print("REGIME SIMULATION — Testing different hourly thresholds")
    print("=" * 100)

    # Current daily regime split for reference
    print(f"\n  Reference: Daily regime split (current thresholds: ADX>25, vol>3%)")
    df_daily_classified = RegimeClassifier.classify_dataframe(df_daily.copy(), min_warmup=50)
    df_daily_classified = df_daily_classified[df_daily_classified['regime'] != 'UNKNOWN']
    daily_regimes = df_daily_classified['regime'].value_counts()
    daily_n = len(df_daily_classified)
    for r in ['TRENDING', 'VOLATILE', 'RANGING']:
        c = daily_regimes.get(r, 0)
        print(f"    {r}: {c} ({c/daily_n*100:.1f}%)")

    print(f"\n  Simulated hourly regime splits:")
    print(f"  {'ADX thresh':>10s} {'Vol thresh':>10s} | {'TRENDING':>12s} {'VOLATILE':>12s} {'RANGING':>12s}")
    print(f"  {'-'*10} {'-'*10}-+-{'-'*12}-{'-'*12}-{'-'*12}")

    for adx_thresh in [20, 22, 25]:
        for vol_thresh in [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
            trending = 0
            volatile = 0
            ranging = 0
            for _, row in df_hourly.iterrows():
                has_direction = row['trend_direction'] in ('UPTREND', 'DOWNTREND')
                if row['adx'] > adx_thresh and has_direction:
                    trending += 1
                elif row['volatility_pct'] > vol_thresh:
                    volatile += 1
                else:
                    ranging += 1
            total = trending + volatile + ranging
            print(f"  {adx_thresh:>10d} {vol_thresh:>9.1f}% | "
                  f"{trending:>5d} ({trending/total*100:>4.1f}%) "
                  f"{volatile:>5d} ({volatile/total*100:>4.1f}%) "
                  f"{ranging:>5d} ({ranging/total*100:>4.1f}%)")

    # =====================================================================
    # Regime stability analysis (how long does each regime last?)
    # =====================================================================
    print("\n" + "=" * 100)
    print("REGIME DURATION ANALYSIS (current thresholds on hourly)")
    print("=" * 100)

    df_hourly_classified = RegimeClassifier.classify_dataframe(df_hourly.copy(), min_warmup=50)
    df_hourly_classified = df_hourly_classified[df_hourly_classified['regime'] != 'UNKNOWN']

    # Count consecutive regime runs
    regimes = df_hourly_classified['regime'].values
    runs = []
    current_regime = regimes[0]
    run_length = 1
    for i in range(1, len(regimes)):
        if regimes[i] == current_regime:
            run_length += 1
        else:
            runs.append((current_regime, run_length))
            current_regime = regimes[i]
            run_length = 1
    runs.append((current_regime, run_length))

    for regime in ['TRENDING', 'VOLATILE', 'RANGING']:
        regime_runs = [r[1] for r in runs if r[0] == regime]
        if regime_runs:
            print(f"\n  {regime} ({len(regime_runs)} episodes):")
            print(f"    Mean duration: {np.mean(regime_runs):.1f} bars ({np.mean(regime_runs):.1f} hours)")
            print(f"    Median:        {np.median(regime_runs):.0f} bars")
            print(f"    Min:           {min(regime_runs)} bars")
            print(f"    Max:           {max(regime_runs)} bars")
            print(f"    P25:           {np.percentile(regime_runs, 25):.0f} bars")
            print(f"    P75:           {np.percentile(regime_runs, 75):.0f} bars")
        else:
            print(f"\n  {regime}: No episodes found")


def main():
    df_daily = load_data('daily')
    df_hourly = load_data('hourly')

    if df_daily is None or df_hourly is None:
        return

    print("\nComputing indicators for daily data...")
    df_daily = compute_raw_indicators(df_daily)

    print("Computing indicators for hourly data...")
    df_hourly = compute_raw_indicators(df_hourly)

    analyze_regime_thresholds(df_daily, df_hourly)


if __name__ == "__main__":
    main()
