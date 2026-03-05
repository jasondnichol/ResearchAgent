"""UAlgo Trend Strategy — Signal Generation & Indicator Library

Core: Supertrend (ATR 14, mult 2.0) for trend detection.
Filters: HMA momentum, CMO, RSI(9), pivot detection.

Entry logic uses a lookback window: the supertrend must have flipped within
the last FLIP_WINDOW bars, and the RSI/CMO/pivot conditions must be met on
the current bar.

BUY (Long): ST flipped bull within FLIP_WINDOW + RSI(9) < threshold + CMO > threshold + low pivot
SELL (Short): ST flipped bear within FLIP_WINDOW + RSI(9) > threshold + CMO < threshold + high pivot
SL: Fixed % from entry (configurable)
TP: R-multiple based (1R@25%, 2R@25%, 3R@50%)

Indicator functions extracted from backtest_new_indicators.py.
Used by ualgo_bot.py for live signal generation.
"""

import requests
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

logger = logging.getLogger("tradingbot")

COINBASE_API = "https://api.exchange.coinbase.com"

# Default UAlgo parameters (overridden by Supabase strategy_params)
DEFAULT_PARAMS = {
    "supertrend_atr_period": 14,
    "supertrend_multiplier": 2.0,
    "sl_pct": 2.0,           # stop loss as %, e.g. 2.0 means 2%
    "flip_window": 5,
    "rsi_period": 9,
    "hma_open_period": 5,
    "hma_close_period": 12,
    "cmo_period": 14,
    "rsi_buy_threshold": 60,
    "rsi_sell_threshold": 40,
    "cmo_buy_threshold": -10,
    "cmo_sell_threshold": 10,
    "tp1_fraction": 0.25,
    "tp2_fraction": 0.25,
    "tp3_fraction": 0.50,
    "short_sl_pct": 2.0,
    "short_rsi_buy_threshold": 40,
    "short_rsi_sell_threshold": 60,
    "short_cmo_buy_threshold": 10,
    "short_cmo_sell_threshold": -10,
    "max_hold_days": 30,
    "lookback_candles": 250,  # candles needed for indicators (200 SMA + buffer)
}

# Bull/bear filter constants (same as Donchian)
BULL_SMA_FAST = 50
BULL_SMA_SLOW = 200
BULL_LOOKBACK = 220


# ============================================================================
# INDICATOR LIBRARY
# ============================================================================

def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR using Wilder's smoothing (EWM with alpha=1/period)."""
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI using Wilder's smoothing."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def calc_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def calc_sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period).mean()


def calc_hma(series: pd.Series, period: int) -> pd.Series:
    """Hull Moving Average: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))"""
    half_period = max(1, int(period / 2))
    sqrt_period = max(1, int(np.sqrt(period)))
    wma_half = series.rolling(window=half_period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)),
        raw=True
    )
    wma_full = series.rolling(window=period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)),
        raw=True
    )
    diff = 2 * wma_half - wma_full
    hma = diff.rolling(window=sqrt_period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)),
        raw=True
    )
    return hma


def calc_cmo(series: pd.Series, period: int = 14) -> pd.Series:
    """Chande Momentum Oscillator: CMO = 100 * (sum_up - sum_dn) / (sum_up + sum_dn)"""
    delta = series.diff()
    up = delta.clip(lower=0)
    dn = (-delta).clip(lower=0)
    sum_up = up.rolling(window=period).sum()
    sum_dn = dn.rolling(window=period).sum()
    total = sum_up + sum_dn
    cmo = np.where(total > 0, 100.0 * (sum_up - sum_dn) / total, 0.0)
    return pd.Series(cmo, index=series.index)


def calc_supertrend(df: pd.DataFrame, atr_period: int = 14, multiplier: float = 3.0):
    """
    Supertrend indicator following Pine Script logic exactly.

    Returns: (trend, supertrend_line, up_band, dn_band) where
      trend = +1 (bull) or -1 (bear)
      supertrend_line = the relevant band value (up when bull, dn when bear)
    """
    atr = calc_atr(df, atr_period)
    hl2 = (df['high'] + df['low']) / 2.0

    up_basic = hl2 - multiplier * atr
    dn_basic = hl2 + multiplier * atr

    n = len(df)
    close = df['close'].values
    up = np.full(n, np.nan)
    dn = np.full(n, np.nan)
    trend = np.full(n, 1)  # 1 = bull, -1 = bear
    st_line = np.full(n, np.nan)

    up_basic_arr = up_basic.values
    dn_basic_arr = dn_basic.values

    first_valid = atr_period
    if first_valid >= n:
        return (pd.Series(trend, index=df.index), pd.Series(st_line, index=df.index),
                pd.Series(up, index=df.index), pd.Series(dn, index=df.index))

    up[first_valid] = up_basic_arr[first_valid]
    dn[first_valid] = dn_basic_arr[first_valid]
    trend[first_valid] = 1
    st_line[first_valid] = up[first_valid]

    for i in range(first_valid + 1, n):
        # Ratchet up band: only moves up, never down
        if np.isnan(up_basic_arr[i]):
            up[i] = up[i - 1] if not np.isnan(up[i - 1]) else 0
        else:
            if close[i - 1] > up[i - 1] and not np.isnan(up[i - 1]):
                up[i] = max(up_basic_arr[i], up[i - 1])
            else:
                up[i] = up_basic_arr[i]

        # Ratchet down band: only moves down, never up
        if np.isnan(dn_basic_arr[i]):
            dn[i] = dn[i - 1] if not np.isnan(dn[i - 1]) else 0
        else:
            if close[i - 1] < dn[i - 1] and not np.isnan(dn[i - 1]):
                dn[i] = min(dn_basic_arr[i], dn[i - 1])
            else:
                dn[i] = dn_basic_arr[i]

        # Trend determination
        prev_trend = trend[i - 1]
        if prev_trend == -1 and close[i] > dn[i - 1]:
            trend[i] = 1
        elif prev_trend == 1 and close[i] < up[i - 1]:
            trend[i] = -1
        else:
            trend[i] = prev_trend

        # Supertrend line value
        if trend[i] == 1:
            st_line[i] = up[i]
        else:
            st_line[i] = dn[i]

    return (
        pd.Series(trend, index=df.index),
        pd.Series(st_line, index=df.index),
        pd.Series(up, index=df.index),
        pd.Series(dn, index=df.index),
    )


# ============================================================================
# CANDLE FETCHING
# ============================================================================

def fetch_daily_candles(symbol: str, limit: int = 250) -> pd.DataFrame:
    """Fetch daily candles from Coinbase public API."""
    url = f"{COINBASE_API}/products/{symbol}/candles"
    params = {'granularity': 86400}
    response = requests.get(url, params=params, timeout=15)
    data = response.json()

    df = pd.DataFrame(data[:limit], columns=['time', 'low', 'high', 'open', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.sort_values('time').reset_index(drop=True)
    return df


def fetch_current_price(symbol: str) -> float:
    """Fetch current price from Coinbase ticker."""
    url = f"{COINBASE_API}/products/{symbol}/ticker"
    response = requests.get(url, timeout=10)
    data = response.json()
    return float(data['price'])


# ============================================================================
# INDICATOR COMPUTATION (applies all UAlgo indicators to a candle DataFrame)
# ============================================================================

def compute_ualgo_indicators(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Compute all UAlgo indicators on a candle DataFrame.

    Returns DataFrame with added columns:
      trend, st_line, rsi9, cmo, is_low_pivot, is_high_pivot,
      bars_since_bull_flip, bars_since_bear_flip
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    df = df.copy()

    # Core supertrend
    trend, st_line, _, _ = calc_supertrend(
        df,
        atr_period=p['supertrend_atr_period'],
        multiplier=p['supertrend_multiplier']
    )
    df['trend'] = trend
    df['st_line'] = st_line

    # HMA momentum
    hma_open = calc_hma(df['open'], p['hma_open_period'])
    hma_close = calc_hma(df['close'], p['hma_close_period'])
    hma_open_shifted = hma_open.shift(1)
    momentum_diff = hma_open_shifted - hma_close

    # CMO of momentum diff
    df['cmo'] = calc_cmo(momentum_diff, p['cmo_period'])

    # RSI
    df['rsi9'] = calc_rsi(df['close'], p['rsi_period'])

    # Pivot detection: low pivot if lowest low in 3-bar window, high pivot if highest high
    df['is_low_pivot'] = (df['low'] <= df['low'].shift(1)) & (df['low'] <= df['low'].shift(2))
    df['is_high_pivot'] = (df['high'] >= df['high'].shift(1)) & (df['high'] >= df['high'].shift(2))

    # Track bars since last supertrend flip
    trend_arr = df['trend'].values
    bars_since_bull_flip = np.full(len(df), 999)
    bars_since_bear_flip = np.full(len(df), 999)
    for i in range(1, len(df)):
        if trend_arr[i] == 1 and trend_arr[i - 1] == -1:
            bars_since_bull_flip[i] = 0
        else:
            bars_since_bull_flip[i] = bars_since_bull_flip[i - 1] + 1
        if trend_arr[i] == -1 and trend_arr[i - 1] == 1:
            bars_since_bear_flip[i] = 0
        else:
            bars_since_bear_flip[i] = bars_since_bear_flip[i - 1] + 1

    df['bars_since_bull_flip'] = bars_since_bull_flip
    df['bars_since_bear_flip'] = bars_since_bear_flip

    # ATR for reference (used in position sizing)
    df['atr'] = calc_atr(df, p['supertrend_atr_period'])

    return df


# ============================================================================
# SIGNAL GENERATION
# ============================================================================

def check_ualgo_long_entry(df: pd.DataFrame, params: dict = None) -> dict:
    """Check for UAlgo LONG entry signal on the latest bar.

    Args:
        df: DataFrame with UAlgo indicators already computed (via compute_ualgo_indicators)
        params: strategy_params dict from bot_config

    Returns:
        dict with keys: signal (bool), entry_price, stop_price, tp_levels, tp_fractions,
                        indicators (rsi, cmo, bars_since_flip, trend)
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    cur = df.iloc[-1]

    rsi_val = float(cur['rsi9']) if not np.isnan(cur['rsi9']) else 999
    cmo_val = float(cur['cmo']) if not np.isnan(cur['cmo']) else -999
    cur_trend = int(cur['trend'])
    bull_flip_recent = int(cur['bars_since_bull_flip']) <= p['flip_window']
    is_low_piv = bool(cur['is_low_pivot'])

    indicators = {
        'rsi9': round(rsi_val, 2),
        'cmo': round(cmo_val, 2),
        'trend': cur_trend,
        'bars_since_bull_flip': int(cur['bars_since_bull_flip']),
        'is_low_pivot': is_low_piv,
        'st_line': round(float(cur['st_line']), 2) if not np.isnan(cur['st_line']) else None,
    }

    # All 5 conditions must be true
    signal = (
        cur_trend == 1 and
        bull_flip_recent and
        rsi_val < p['rsi_buy_threshold'] and
        cmo_val > p['cmo_buy_threshold'] and
        is_low_piv
    )

    result = {'signal': signal, 'indicators': indicators}

    if signal:
        entry_price = float(cur['close'])
        sl_pct = p['sl_pct'] / 100.0  # convert from % to decimal
        stop_price = entry_price * (1.0 - sl_pct)
        atr_val = float(cur['atr']) if not np.isnan(cur.get('atr', float('nan'))) else 0

        # ATR-Fibonacci take profit levels
        tp1 = entry_price + 1.618 * atr_val
        tp2 = entry_price + 2.618 * atr_val
        tp3 = entry_price + 3.618 * atr_val

        result.update({
            'entry_price': entry_price,
            'stop_price': stop_price,
            'tp_levels': [tp1, tp2, tp3],
            'tp_fractions': [p['tp1_fraction'], p['tp2_fraction'], p['tp3_fraction']],
        })

    return result


def check_ualgo_short_entry(df: pd.DataFrame, params: dict = None) -> dict:
    """Check for UAlgo SHORT entry signal on the latest bar.

    Args:
        df: DataFrame with UAlgo indicators already computed (via compute_ualgo_indicators)
        params: strategy_params dict from bot_config

    Returns:
        dict with keys: signal (bool), entry_price, stop_price, tp_levels, tp_fractions,
                        indicators (rsi, cmo, bars_since_flip, trend)
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    cur = df.iloc[-1]

    rsi_val = float(cur['rsi9']) if not np.isnan(cur['rsi9']) else -999
    cmo_val = float(cur['cmo']) if not np.isnan(cur['cmo']) else 999
    cur_trend = int(cur['trend'])
    bear_flip_recent = int(cur['bars_since_bear_flip']) <= p['flip_window']
    is_high_piv = bool(cur['is_high_pivot'])

    indicators = {
        'rsi9': round(rsi_val, 2),
        'cmo': round(cmo_val, 2),
        'trend': cur_trend,
        'bars_since_bear_flip': int(cur['bars_since_bear_flip']),
        'is_high_pivot': is_high_piv,
        'st_line': round(float(cur['st_line']), 2) if not np.isnan(cur['st_line']) else None,
    }

    # All 5 conditions must be true
    signal = (
        cur_trend == -1 and
        bear_flip_recent and
        rsi_val > p['short_rsi_sell_threshold'] and
        cmo_val < p['short_cmo_sell_threshold'] and
        is_high_piv
    )

    result = {'signal': signal, 'indicators': indicators}

    if signal:
        entry_price = float(cur['close'])
        sl_pct = p['short_sl_pct'] / 100.0
        stop_price = entry_price * (1.0 + sl_pct)
        atr_val = float(cur['atr']) if not np.isnan(cur.get('atr', float('nan'))) else 0

        # ATR-Fibonacci take profit levels (inverted for short)
        tp1 = entry_price - 1.618 * atr_val
        tp2 = entry_price - 2.618 * atr_val
        tp3 = entry_price - 3.618 * atr_val

        result.update({
            'entry_price': entry_price,
            'stop_price': stop_price,
            'tp_levels': [tp1, tp2, tp3],
            'tp_fractions': [p['tp1_fraction'], p['tp2_fraction'], p['tp3_fraction']],
        })

    return result


# ============================================================================
# EXIT CHECKS (called during trailing stop loop)
# ============================================================================

def check_ualgo_stop_loss(position: dict, current_price: float) -> dict:
    """Check if stop loss is hit for a UAlgo position.

    Args:
        position: dict with side, stop_price
        current_price: current market price

    Returns:
        dict with hit (bool), exit_reason
    """
    side = position.get('side', 'LONG')
    stop = position.get('stop_price')

    if stop is None:
        return {'hit': False}

    if side in ('LONG', 'FUTURES_LONG'):
        hit = current_price <= stop
    else:  # SHORT
        hit = current_price >= stop

    return {
        'hit': hit,
        'exit_reason': 'Stop loss' if hit else None,
    }


def check_ualgo_partial_tp(position: dict, current_price: float) -> dict:
    """Check if a partial TP level is hit.

    Args:
        position: dict with side, entry_price, stop_price, partials_taken,
                  remaining_fraction, tp_levels (computed from entry + stop at open)
        current_price: current high (for longs) or low (for shorts)

    Returns:
        dict with hit (bool), tp_index, tp_price, fraction, exit_reason
    """
    side = position.get('side', 'LONG')
    entry = position['entry_price']
    stop = position['stop_price']
    partials_taken = position.get('partials_taken', 0)

    # Compute TP levels from entry + stop distance
    if side in ('LONG', 'FUTURES_LONG'):
        risk_dist = entry - stop
        tp_levels = [
            entry + 1.0 * risk_dist,
            entry + 2.0 * risk_dist,
            entry + 3.0 * risk_dist,
        ]
    else:  # SHORT
        risk_dist = stop - entry
        tp_levels = [
            entry - 1.0 * risk_dist,
            entry - 2.0 * risk_dist,
            entry - 3.0 * risk_dist,
        ]

    tp_fractions = [0.25, 0.25, 0.50]

    if partials_taken >= len(tp_levels):
        return {'hit': False}

    tp_idx = partials_taken
    tp_price = tp_levels[tp_idx]

    if side in ('LONG', 'FUTURES_LONG'):
        hit = current_price >= tp_price
    else:
        hit = current_price <= tp_price

    if hit:
        return {
            'hit': True,
            'tp_index': tp_idx,
            'tp_price': tp_price,
            'fraction': tp_fractions[tp_idx],
            'exit_reason': f'TP{tp_idx + 1} ({tp_idx + 1}R)',
        }
    return {'hit': False}


def check_ualgo_opposing_signal(position: dict, df: pd.DataFrame) -> dict:
    """Check if an opposing supertrend flip closes the position.

    Args:
        position: dict with side
        df: DataFrame with UAlgo indicators computed

    Returns:
        dict with hit (bool), exit_reason
    """
    if len(df) < 2:
        return {'hit': False}

    cur_trend = int(df['trend'].iloc[-1])
    prev_trend = int(df['trend'].iloc[-2])
    side = position.get('side', 'LONG')

    if side in ('LONG', 'FUTURES_LONG'):
        hit = cur_trend == -1 and prev_trend == 1
    else:
        hit = cur_trend == 1 and prev_trend == -1

    return {
        'hit': hit,
        'exit_reason': 'Opposing signal' if hit else None,
    }


def check_ualgo_max_hold(position: dict, max_hold_days: int = 30) -> dict:
    """Check if max hold period exceeded.

    Args:
        position: dict with entry_time (datetime or ISO string)
        max_hold_days: max days to hold (default 30)

    Returns:
        dict with hit (bool), hold_days, exit_reason
    """
    entry_time = position.get('entry_time')
    if entry_time is None:
        return {'hit': False, 'hold_days': 0}

    if isinstance(entry_time, str):
        entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))

    from datetime import timezone
    now = datetime.now(timezone.utc)
    hold_days = (now - entry_time).total_seconds() / 86400

    return {
        'hit': hold_days >= max_hold_days,
        'hold_days': round(hold_days, 1),
        'exit_reason': f'Max hold ({max_hold_days}d)' if hold_days >= max_hold_days else None,
    }


# ============================================================================
# BULL/BEAR MARKET FILTERS (same logic as Donchian bot)
# ============================================================================

def check_bull_filter(enabled: bool = True) -> tuple:
    """Check if BTC is in a bull market. Bull = BTC close > SMA(200).
    Returns (is_bull, details_dict).
    """
    if not enabled:
        return True, {'status': 'DISABLED'}

    try:
        url = f"{COINBASE_API}/products/BTC-USD/candles"
        params = {'granularity': 86400}
        response = requests.get(url, params=params, timeout=15)
        data = response.json()

        df = pd.DataFrame(data[:BULL_LOOKBACK], columns=['time', 'low', 'high', 'open', 'close', 'volume'])
        df = df.sort_values('time').reset_index(drop=True)

        if len(df) < BULL_SMA_SLOW:
            logger.warning(f"Bull filter: only {len(df)} candles, need {BULL_SMA_SLOW}")
            return True, {'status': 'INSUFFICIENT_DATA'}

        sma_fast = df['close'].rolling(window=BULL_SMA_FAST).mean().iloc[-1]
        sma_slow = df['close'].rolling(window=BULL_SMA_SLOW).mean().iloc[-1]
        btc_close = float(df['close'].iloc[-1])

        above_200 = btc_close > sma_slow
        is_bull = above_200

        details = {
            'status': 'BULL' if is_bull else 'BEAR',
            'btc_close': btc_close,
            'sma_50': round(float(sma_fast), 2),
            'sma_200': round(float(sma_slow), 2),
            'above_200': above_200,
        }
        return is_bull, details

    except Exception as e:
        logger.error(f"Bull filter check failed: {e}")
        return True, {'status': 'ERROR', 'error': str(e)}


def check_bear_filter(enabled: bool = True) -> tuple:
    """Check if BTC is in a bear market (death cross).
    Bear = BTC close < SMA(200) AND SMA(50) < SMA(200).
    Returns (is_bear, details_dict).
    """
    if not enabled:
        return False, {'status': 'DISABLED'}

    try:
        url = f"{COINBASE_API}/products/BTC-USD/candles"
        params = {'granularity': 86400}
        response = requests.get(url, params=params, timeout=15)
        data = response.json()

        df = pd.DataFrame(data[:BULL_LOOKBACK], columns=['time', 'low', 'high', 'open', 'close', 'volume'])
        df = df.sort_values('time').reset_index(drop=True)

        if len(df) < BULL_SMA_SLOW:
            logger.warning(f"Bear filter: only {len(df)} candles, need {BULL_SMA_SLOW}")
            return False, {'status': 'INSUFFICIENT_DATA'}

        sma_fast = df['close'].rolling(window=BULL_SMA_FAST).mean().iloc[-1]
        sma_slow = df['close'].rolling(window=BULL_SMA_SLOW).mean().iloc[-1]
        btc_close = float(df['close'].iloc[-1])

        below_200 = btc_close < sma_slow
        death_cross = sma_fast < sma_slow
        is_bear = below_200 and death_cross

        details = {
            'status': 'DEATH_CROSS' if is_bear else ('BELOW_200' if below_200 else 'NEUTRAL'),
            'btc_close': btc_close,
            'sma_50': round(float(sma_fast), 2),
            'sma_200': round(float(sma_slow), 2),
            'below_200': below_200,
            'death_cross': death_cross,
        }
        return is_bear, details

    except Exception as e:
        logger.error(f"Bear filter check failed: {e}")
        return False, {'status': 'ERROR', 'error': str(e)}
