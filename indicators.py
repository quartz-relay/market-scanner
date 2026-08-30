import pandas as pd
import numpy as np

# Pure math only. No tickers, no constants, no I/O.

def compute_daily_indicators(historicals_dict_list):
    """
    Computes all daily-bar-derived indicators. Output is safe to cache for
    the full trading day -- only live price changes intraday. See
    compose_live_metrics() for combining this with a fresh live price.

    Added vs original:
      - RSI-2   : 2-period RSI for sensitive mean-reversion dip detection
      - ADX-14  : Average Directional Index for trend-vs-range regime
      - TRIX-15 : Triple-smoothed EMA rate-of-change for momentum confirmation
      - TRIX-Signal : 9-period EMA of TRIX (signal line crossover)
      - Volume-20avg: 20-day average volume for breakout volume filter
      - Volume-ratio: today's volume / 20-day average (>1.1 = above-avg breakout)
    """
    if not historicals_dict_list:
        return None

    df = pd.DataFrame(historicals_dict_list)
    df['close_price'] = pd.to_numeric(df['close_price'])
    df['high_price']  = pd.to_numeric(df['high_price'])
    df['low_price']   = pd.to_numeric(df['low_price'])

    # Volume is optional — not all historicals include it
    has_volume = 'volume' in df.columns and df['volume'].notna().any()
    if has_volume:
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)

    if not (len(df['close_price']) == len(df['high_price']) == len(df['low_price'])):
        raise ValueError("close/high/low arrays are mismatched in length")

    # ── SMA ──────────────────────────────────────────────────────────────────
    df['sma20'] = df['close_price'].rolling(window=20).mean()
    df['sma50'] = df['close_price'].rolling(window=50).mean()

    # ── RSI-14 (Wilder's EWM) ────────────────────────────────────────────────
    delta14 = df['close_price'].diff()
    gain14  = (delta14.where(delta14 > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss14  = (-delta14.where(delta14 < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    df['rsi14'] = 100 - (100 / (1 + gain14 / loss14))

    # ── RSI-2 (ultra-sensitive mean-reversion trigger) ────────────────────────
    delta2 = df['close_price'].diff()
    gain2  = (delta2.where(delta2 > 0, 0)).ewm(alpha=1/2, adjust=False).mean()
    loss2  = (-delta2.where(delta2 < 0, 0)).ewm(alpha=1/2, adjust=False).mean()
    df['rsi2'] = 100 - (100 / (1 + gain2 / loss2))

    # ── Bollinger Bands (20/2) ────────────────────────────────────────────────
    std20 = df['close_price'].rolling(window=20).std()
    df['upper_band'] = df['sma20'] + (2 * std20)
    df['lower_band'] = df['sma20'] - (2 * std20)

    # ── ATR-14 ───────────────────────────────────────────────────────────────
    prev_close = df['close_price'].shift(1)
    tr = pd.concat([
        df['high_price'] - df['low_price'],
        (df['high_price'] - prev_close).abs(),
        (df['low_price']  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df['atr14'] = tr.ewm(alpha=1/14, adjust=False).mean()

    # ── EMA-10 ───────────────────────────────────────────────────────────────
    df['ema10'] = df['close_price'].ewm(span=10, adjust=False).mean()

    # ── EMA-20 / EMA-50 / EMA-200 (for ADX trend pillars) ───────────────────
    df['ema20']  = df['close_price'].ewm(span=20,  adjust=False).mean()
    df['ema50']  = df['close_price'].ewm(span=50,  adjust=False).mean()
    df['ema200'] = df['close_price'].ewm(span=200, adjust=False).mean()

    # ── 20d closing high ─────────────────────────────────────────────────────
    df['high20'] = df['close_price'].rolling(window=20).max()

    # ── 10-day % return ──────────────────────────────────────────────────────
    ten_days_ago = df.iloc[-11] if len(df) >= 11 else None
    latest = df.iloc[-1]
    return_10d = (
        (latest['close_price'] - ten_days_ago['close_price'])
        / ten_days_ago['close_price'] * 100
        if ten_days_ago is not None else None
    )

    # ── ADX-14 ───────────────────────────────────────────────────────────────
    # +DM / -DM
    high_diff = df['high_price'].diff()
    low_diff  = df['low_price'].diff().mul(-1)
    plus_dm   = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
    minus_dm  = low_diff.where((low_diff > high_diff)  & (low_diff  > 0), 0.0)

    atr14_raw  = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di14  = 100 * plus_dm.ewm(alpha=1/14,  adjust=False).mean() / atr14_raw
    minus_di14 = 100 * minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr14_raw

    dx = (100 * (plus_di14 - minus_di14).abs()
          / (plus_di14 + minus_di14).replace(0, np.nan)).fillna(0)
    _adx14 = dx.ewm(alpha=1/14, adjust=False).mean()

    # ── TRIX-15 + Signal-9 ───────────────────────────────────────────────────
    ema1 = df['close_price'].ewm(span=15, adjust=False).mean()
    ema2 = ema1.ewm(span=15, adjust=False).mean()
    ema3 = ema2.ewm(span=15, adjust=False).mean()
    _trix15 = ema3.pct_change() * 100         # rate of change of triple EWM
    _trix_signal9 = _trix15.ewm(span=9, adjust=False).mean()

    # ── Volume indicators ─────────────────────────────────────────────────────
    if has_volume:
        vol_avg20     = df['volume'].rolling(window=20).mean()
        volume_ratio  = (df['volume'] / vol_avg20).fillna(1.0)
        vol_avg20_val = round(float(vol_avg20.iloc[-1]), 0) if not pd.isna(vol_avg20.iloc[-1]) else None
        vol_ratio_val = round(float(volume_ratio.iloc[-1]), 2) if not pd.isna(volume_ratio.iloc[-1]) else 1.0
    else:
        vol_avg20_val = None
        vol_ratio_val = None

    def _safe(val):
        return round(float(val), 4) if val is not None and not pd.isna(val) else 0.0

    def _safe2(val, decimals=2):
        return round(float(val), decimals) if val is not None and not pd.isna(val) else 0.0

    return {
        "RSI-14":        _safe2(latest['rsi14']),
        "RSI-2":         _safe2(latest['rsi2']),
        "SMA20":         _safe2(latest['sma20']),
        "SMA50":         _safe2(latest['sma50']),
        "EMA-10":        _safe2(latest['ema10']),
        "EMA-20":        _safe2(latest['ema20']),
        "EMA-50":        _safe2(latest['ema50']),
        "EMA-200":       _safe2(latest['ema200']),
        "UpperBand":     _safe2(latest['upper_band']),
        "LowerBand":     _safe2(latest['lower_band']),
        "ATR-14":        _safe(latest['atr14']),
        "ADX-14":        _safe2(_adx14.iloc[-1]),
        "TRIX-15":       _safe2(_trix15.iloc[-1], 4),
        "TRIX-Signal-9": _safe2(_trix_signal9.iloc[-1], 4),
        "20d-High":      _safe2(latest['high20']),
        "Return-10d-pct": round(return_10d, 2) if return_10d is not None else None,
        "Volume-20avg":  vol_avg20_val,
        "Volume-ratio":  vol_ratio_val,
    }


def compose_live_metrics(cached_daily_indicators, live_price):
    """
    Combines cached daily indicators with a fresh live price each cycle.
    Adds %B, Is-New-20d-High, and score components. Never cache this output.
    """
    d = cached_daily_indicators
    upper, lower = d["UpperBand"], d["LowerBand"]
    pct_b = (live_price - lower) / (upper - lower) if upper != lower else 0.0

    # ── Three-pillar score (-6 to +6) ────────────────────────────────────────
    # Trend pillar (-2 to +2)
    trend_score = 0
    if live_price > d["EMA-20"]:   trend_score += 1
    if d["EMA-20"] > d["EMA-50"]:  trend_score += 0.5
    if d["EMA-50"] > d["EMA-200"]: trend_score += 0.5
    ema200 = d["EMA-200"]
    if ema200 > 0:
        # EMA-200 slope proxy: compare to SMA50 as a stand-in
        if d["SMA50"] > ema200: trend_score += 0.5
        else:                   trend_score -= 0.5
    trend_score = max(-2, min(2, round(trend_score)))

    # Momentum pillar (-2 to +2)
    rsi14 = d["RSI-14"]
    mom_score = 0
    if rsi14 > 55:   mom_score += 1
    elif rsi14 < 45: mom_score -= 1
    trix  = d["TRIX-15"]
    tsig  = d["TRIX-Signal-9"]
    if trix > 0:            mom_score += 0.5
    else:                   mom_score -= 0.5
    if trix > tsig:         mom_score += 0.5
    else:                   mom_score -= 0.5
    mom_score = max(-2, min(2, round(mom_score)))

    # Exhaustion / Bollinger pillar (-2 to +2)
    if pct_b > 1.0:        exh_score = 2    # above upper band = overbought exhaustion
    elif pct_b > 0.8:      exh_score = 1
    elif pct_b < 0.0:      exh_score = -2   # below lower band = oversold exhaustion
    elif pct_b < 0.2:      exh_score = -1
    else:                  exh_score = 0

    composite_score = trend_score + mom_score + exh_score  # -6 to +6

    return {
        "Live":             round(live_price, 2),
        "%B":               round(pct_b, 2),
        "Is-New-20d-High":  bool(live_price >= d["20d-High"]),
        "TrendScore":       trend_score,
        "MomentumScore":    mom_score,
        "ExhaustionScore":  exh_score,
        "CompositeScore":   composite_score,
        **d,
    }


def calculate_indicators(historicals_dict_list, live_price):
    """One-shot wrapper (no caching). Use compute_daily_indicators +
    compose_live_metrics separately when caching is available."""
    daily = compute_daily_indicators(historicals_dict_list)
    if daily is None:
        return None
    return compose_live_metrics(daily, live_price)


def update_trailing_peak(cached_peak, current_price, entry_price=None):
    """Ratchets the momentum trailing-stop peak upward only."""
    if cached_peak is None:
        base = entry_price if entry_price is not None else current_price
        return max(base, current_price)
    return max(cached_peak, current_price)


def evaluate_trend_breakdown(cached_state, price_below_sma20, sector_return_negative, today_date):
    """
    Advances the mean-reversion 2-close breakdown state machine at most
    once per calendar day. Returns (new_state_dict, is_exit_signal).
    """
    if cached_state is None:
        cached_state = {"state": "NONE", "last_updated_date": None}

    if cached_state.get("last_updated_date") == today_date:
        return cached_state, False

    condition_today = price_below_sma20 and sector_return_negative
    current = cached_state.get("state", "NONE")

    if current == "NONE":
        new_state = "DAY1_PENDING" if condition_today else "NONE"
        is_exit   = False
    elif current == "DAY1_PENDING":
        if condition_today:
            new_state = "EXIT_CONFIRMED"
            is_exit   = True
        else:
            new_state = "NONE"
            is_exit   = False
    elif current == "EXIT_CONFIRMED":
        # Reset after one cycle so position can re-enter the machine if still held
        new_state = "NONE"
        is_exit   = False
    else:
        new_state = "NONE"
        is_exit   = False

    return {"state": new_state, "last_updated_date": today_date}, is_exit


def calculate_rs(stock_return_10d_pct, sector_return_10d_pct):
    """Relative Strength = stock 10d return minus sector ETF 10d return (%)."""
    return round(stock_return_10d_pct - sector_return_10d_pct, 2)


def check_safety_mode(pnl_history, total_portfolio_value):
    """Circuit breaker: trailing 24h realized losses > 5% of current portfolio."""
    realized_24h = sum(t['realized_pnl'] for t in pnl_history if t.get('is_last_24h'))
    return realized_24h < -0.05 * total_portfolio_value


def adx_regime(adx_value):
    """
    Returns the market regime label for a given ADX-14 value.
      < 20  : ranging  → mean-reversion signals more reliable
      20-25 : weak trend
      > 25  : trending → momentum signals more reliable
    """
    if adx_value < 20:
        return "ranging"
    elif adx_value < 25:
        return "weak_trend"
    else:
        return "trending"

