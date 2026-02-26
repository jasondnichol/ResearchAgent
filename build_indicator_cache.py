"""Build enriched indicator cache from hourly BTC data.

Loads btc_4year_hourly_cache.json, computes ALL indicators needed by all 3
strategies (Williams %R, BB Mean Reversion, ADX Momentum), classifies regimes,
and saves the enriched DataFrame to btc_4year_hourly_enriched.json.

Indicator formulas match EXACTLY:
  - market_regime.py  (RegimeClassifier.compute_indicators)
  - backtest_unified_4year.py  (calculate_all_indicators)

All smoothing uses Wilder's EWM (alpha=1/period, adjust=False).

Usage:
    python build_indicator_cache.py
    python build_indicator_cache.py --force   # ignore existing enriched cache
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime
from market_regime import RegimeClassifier


# ============================================================================
# LOAD RAW HOURLY CACHE
# ============================================================================

def load_hourly_cache(cache_file='btc_4year_hourly_cache.json'):
    """Load raw hourly OHLCV data from the JSON cache."""
    if not os.path.exists(cache_file):
        print(f"ERROR: {cache_file} not found. Run backtest_unified_4year.py first to fetch data.")
        sys.exit(1)

    print(f"Loading raw hourly cache from {cache_file}...")
    with open(cache_file, 'r') as f:
        cache_data = json.load(f)

    df = pd.DataFrame(cache_data['data'])
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)

    # Ensure numeric types
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    raw_meta = cache_data.get('metadata', {})
    print(f"  Loaded {len(df):,} hourly candles")
    print(f"  Range: {df['time'].min()} to {df['time'].max()}")
    return df, raw_meta


# ============================================================================
# INDICATOR COMPUTATION
# ============================================================================

def compute_all_indicators(df):
    """Compute every indicator used by all 3 strategies + regime classifier.

    Matches EXACTLY:
      - RegimeClassifier.compute_indicators() in market_regime.py
      - calculate_all_indicators() in backtest_unified_4year.py

    Columns added:
      Williams %R:   williams_r, sma_21
      Bollinger:     sma_20, bb_std, bb_lower_1_5, bb_upper_1_5, bb_lower_2_0, bb_upper_2_0
      RSI:           rsi
      ATR:           atr
      ADX/DI:        adx, plus_di, minus_di
      SMA:           sma_50
      Regime:        volatility_pct, trend_direction
    """
    df = df.copy()
    period = 14

    # ----- Williams %R(14) -----
    # Matches backtest_unified_4year.py calculate_all_indicators()
    high_roll = df['high'].rolling(window=period).max()
    low_roll = df['low'].rolling(window=period).min()
    df['williams_r'] = -100 * ((high_roll - df['close']) / (high_roll - low_roll))

    # ----- SMA(21) — used by Williams %R strategy -----
    df['sma_21'] = df['close'].rolling(window=21).mean()

    # ----- SMA(20) — used by BB + regime classifier -----
    # Matches both market_regime.py and backtest_unified_4year.py
    bb_period = 20
    df['sma_20'] = df['close'].rolling(window=bb_period).mean()

    # ----- Bollinger Bands -----
    # bb_std, bb_lower, bb_upper at both 1.5 sigma and 2.0 sigma
    # (backtest uses params['bb_sigma'] which can be 1.5 or 2.0)
    df['bb_std'] = df['close'].rolling(window=bb_period).std()
    df['bb_lower_1_5'] = df['sma_20'] - 1.5 * df['bb_std']
    df['bb_upper_1_5'] = df['sma_20'] + 1.5 * df['bb_std']
    df['bb_lower_2_0'] = df['sma_20'] - 2.0 * df['bb_std']
    df['bb_upper_2_0'] = df['sma_20'] + 2.0 * df['bb_std']

    # ----- RSI(14) using Wilder's EWM -----
    # Matches backtest_unified_4year.py exactly
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # ----- ATR(14) using Wilder's EWM -----
    # Matches both market_regime.py and backtest_unified_4year.py
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    # ----- +DI, -DI, ADX(14) using Wilder's EWM -----
    # Matches both market_regime.py and backtest_unified_4year.py
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

    # Wilder's smoothing for DM
    smooth_plus_dm = plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    smooth_minus_dm = minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    df['plus_di'] = 100 * smooth_plus_dm / df['atr']
    df['minus_di'] = 100 * smooth_minus_dm / df['atr']

    dx = 100 * (df['plus_di'] - df['minus_di']).abs() / (df['plus_di'] + df['minus_di'])
    df['adx'] = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    # ----- SMA(50) -----
    # Matches both market_regime.py and backtest_unified_4year.py
    df['sma_50'] = df['close'].rolling(window=50).mean()

    # ----- Volatility % = ATR / close * 100 -----
    # Matches market_regime.py RegimeClassifier.compute_indicators()
    df['volatility_pct'] = (df['atr'] / df['close']) * 100

    # ----- Trend Direction -----
    # Matches market_regime.py RegimeClassifier.compute_indicators()
    df['trend_direction'] = 'SIDEWAYS'
    df.loc[
        (df['close'] > df['sma_20']) & (df['sma_20'] > df['sma_50']),
        'trend_direction'
    ] = 'UPTREND'
    df.loc[
        (df['close'] < df['sma_20']) & (df['sma_20'] < df['sma_50']),
        'trend_direction'
    ] = 'DOWNTREND'

    return df


# ============================================================================
# REGIME CLASSIFICATION
# ============================================================================

def classify_regimes(df, timeframe='hourly', min_warmup=50):
    """Classify each bar's market regime using RegimeClassifier.

    Uses the canonical classify_bar() from market_regime.py with
    timeframe='hourly' (volatility threshold 0.5%).

    Bars before min_warmup are labelled 'UNKNOWN'.
    """
    df['regime'] = 'UNKNOWN'

    for i in range(min_warmup, len(df)):
        row = df.iloc[i]
        if pd.isna(row['adx']) or pd.isna(row['sma_20']):
            continue
        df.iloc[i, df.columns.get_loc('regime')] = RegimeClassifier.classify_bar(
            row['adx'], row['volatility_pct'], row['trend_direction'],
            timeframe=timeframe
        )

    return df


# ============================================================================
# SAVE ENRICHED CACHE
# ============================================================================

def save_enriched_cache(df, raw_meta, output_file='btc_4year_hourly_enriched.json'):
    """Save the enriched DataFrame with all indicator columns to JSON."""

    # List all indicator columns that were added
    ohlcv_cols = {'time', 'open', 'high', 'low', 'close', 'volume'}
    indicator_cols = sorted([c for c in df.columns if c not in ohlcv_cols])

    # Prepare export DataFrame
    df_export = df.copy()
    df_export['time'] = df_export['time'].astype(str)

    # Replace NaN/inf with None for clean JSON
    df_export = df_export.replace([np.inf, -np.inf], np.nan)

    # Count regime distribution (excluding UNKNOWN)
    regime_dist = df[df['regime'] != 'UNKNOWN']['regime'].value_counts().to_dict()
    total_classified = sum(regime_dist.values())

    metadata = {
        'description': 'Enriched hourly BTC cache with all strategy indicators and regime labels',
        'built_at': datetime.now().isoformat(),
        'source_file': 'btc_4year_hourly_cache.json',
        'source_metadata': raw_meta,
        'timeframe': 'hourly',
        'total_candles': len(df),
        'start_date': str(df['time'].min()),
        'end_date': str(df['time'].max()),
        'min_warmup': 50,
        'regime_timeframe': 'hourly',
        'regime_volatility_threshold': RegimeClassifier.VOLATILITY_THRESHOLDS['hourly'],
        'regime_adx_threshold': RegimeClassifier.ADX_THRESHOLD,
        'regime_distribution': {
            regime: {
                'count': int(count),
                'pct': round(count / total_classified * 100, 1)
            }
            for regime, count in regime_dist.items()
        },
        'indicator_columns': indicator_cols,
        'indicators_computed': {
            'williams_r': 'Williams %R(14) — highest high / lowest low over 14 bars',
            'sma_21': 'Simple Moving Average(21) — Williams %R strategy',
            'sma_20': 'Simple Moving Average(20) — BB middle band / regime classifier',
            'bb_std': 'Bollinger Band std dev (20-bar rolling)',
            'bb_lower_1_5': 'Lower BB at 1.5 sigma (loosened params)',
            'bb_upper_1_5': 'Upper BB at 1.5 sigma (loosened params)',
            'bb_lower_2_0': 'Lower BB at 2.0 sigma (strict/production params)',
            'bb_upper_2_0': 'Upper BB at 2.0 sigma (strict/production params)',
            'rsi': "RSI(14) — Wilder's EWM (alpha=1/14)",
            'atr': "ATR(14) — Wilder's EWM (alpha=1/14)",
            'plus_di': "+DI(14) — Wilder's EWM smoothed directional indicator",
            'minus_di': "-DI(14) — Wilder's EWM smoothed directional indicator",
            'adx': "ADX(14) — Wilder's EWM of DX",
            'sma_50': 'Simple Moving Average(50) — trend filter',
            'volatility_pct': 'ATR / close * 100 — volatility percentage',
            'trend_direction': 'UPTREND / DOWNTREND / SIDEWAYS based on close vs SMA20 vs SMA50',
            'regime': 'TRENDING / VOLATILE / RANGING / UNKNOWN — from RegimeClassifier',
        },
        'smoothing_method': "Wilder's EWM: ewm(alpha=1/period, min_periods=period, adjust=False)",
        'formula_source_files': [
            'market_regime.py (RegimeClassifier.compute_indicators)',
            'backtest_unified_4year.py (calculate_all_indicators)',
        ],
    }

    # Convert to records, handling NaN -> None via pandas
    records = json.loads(df_export.to_json(orient='records'))

    cache_data = {
        'metadata': metadata,
        'data': records,
    }

    print(f"\nSaving enriched cache to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(cache_data, f)

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  Saved {len(df):,} rows x {len(df.columns)} columns ({file_size_mb:.1f} MB)")
    print(f"  Indicator columns: {len(indicator_cols)}")
    print(f"  {indicator_cols}")

    return output_file


# ============================================================================
# VERIFICATION
# ============================================================================

def verify_against_originals(df):
    """Spot-check that our indicators match RegimeClassifier.compute_indicators()
    and that regime labels match RegimeClassifier.classify_dataframe().

    Raises AssertionError if there's a mismatch.
    """
    print("\nVerifying indicators match RegimeClassifier.compute_indicators()...")

    # Build a reference DataFrame from RegimeClassifier
    ref_df = df[['time', 'open', 'high', 'low', 'close', 'volume']].copy()
    ref_df = RegimeClassifier.compute_indicators(ref_df)

    # Compare key columns on rows where both have valid data
    check_cols = ['sma_20', 'sma_50', 'atr', 'adx', 'volatility_pct']
    for col in check_cols:
        mask = ref_df[col].notna() & df[col].notna()
        if mask.sum() == 0:
            continue
        max_diff = (ref_df.loc[mask, col] - df.loc[mask, col]).abs().max()
        assert max_diff < 1e-10, f"Mismatch in {col}: max diff = {max_diff}"
        print(f"  {col}: OK (max diff = {max_diff:.2e})")

    # Verify trend_direction
    mask = ref_df['sma_50'].notna()
    mismatches = (ref_df.loc[mask, 'trend_direction'] != df.loc[mask, 'trend_direction']).sum()
    assert mismatches == 0, f"trend_direction mismatch: {mismatches} rows differ"
    print(f"  trend_direction: OK (0 mismatches)")

    # Verify regime labels match classify_dataframe
    ref_regime = RegimeClassifier.classify_dataframe(
        df[['time', 'open', 'high', 'low', 'close', 'volume']].copy(),
        min_warmup=50,
        timeframe='hourly'
    )
    mask = (ref_regime['regime'] != 'UNKNOWN') & (df['regime'] != 'UNKNOWN')
    mismatches = (ref_regime.loc[mask, 'regime'] != df.loc[mask, 'regime']).sum()
    assert mismatches == 0, f"regime mismatch: {mismatches} rows differ"
    print(f"  regime: OK (0 mismatches)")

    print("  All checks passed.")


# ============================================================================
# MAIN
# ============================================================================

def main():
    force = '--force' in sys.argv
    output_file = 'btc_4year_hourly_enriched.json'

    # Check if enriched cache already exists (skip if recent and not forced)
    if not force and os.path.exists(output_file):
        file_time = datetime.fromtimestamp(os.path.getmtime(output_file))
        age_days = (datetime.now() - file_time).days
        if age_days <= 7:
            print(f"Enriched cache already exists ({age_days} days old). Use --force to rebuild.")
            return

    print("=" * 80)
    print("BUILD INDICATOR CACHE — Enriched Hourly BTC Data")
    print("=" * 80)

    # Step 1: Load raw data
    df, raw_meta = load_hourly_cache()

    # Step 2: Compute all indicators
    print("\nComputing all indicators (Wilder's EWM)...")
    df = compute_all_indicators(df)
    print(f"  Williams %R(14), SMA(21)")
    print(f"  BB: SMA(20), bb_std, bb_lower/upper at 1.5 and 2.0 sigma")
    print(f"  RSI(14), ATR(14), ADX(14), +DI, -DI")
    print(f"  SMA(50), volatility_pct, trend_direction")

    # Step 3: Classify regimes
    print("\nClassifying regimes (timeframe='hourly', vol threshold=0.5%)...")
    df = classify_regimes(df, timeframe='hourly', min_warmup=50)

    # Print regime distribution
    valid = df[df['regime'] != 'UNKNOWN']
    regime_counts = valid['regime'].value_counts()
    total = len(valid)
    print(f"  Regime distribution ({total:,} classified bars):")
    for regime in ['TRENDING', 'VOLATILE', 'RANGING']:
        count = regime_counts.get(regime, 0)
        print(f"    {regime}: {count:,} bars ({count/total*100:.1f}%)")

    # Step 4: Verify against originals
    verify_against_originals(df)

    # Step 5: Save
    save_enriched_cache(df, raw_meta, output_file)

    print(f"\nDone. Future backtests can load {output_file} and skip indicator computation.")


if __name__ == "__main__":
    main()
