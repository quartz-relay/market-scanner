"""
run_scan.py  —  Trade signal engine. Accepts pre-fetched Robinhood data from
Claude (the orchestrator), runs all calculations, posts to Slack, and logs to
Google Sheets. Claude does NOT post signals or write logs — this script does.

Usage:
    python run_scan.py --input payload.json

Required env vars (same as before):
    WORKER_URL                  Cloudflare Worker base URL
    WORKER_SHARED_SECRET        Auth secret for Worker endpoints
    SHEET_LOG_WEBAPP_URL        Google Apps Script Web App URL
    SHEET_LOG_SHARED_SECRET     Auth secret for Sheet endpoint

Input JSON schema (Claude builds this from Robinhood MCP calls):
{
  "run":             "A" | "B" | "C",
  "entry_tickers":   ["NVDA", "TSM", ...],        # this run's 15 tickers
  "portfolio_balance": 312.50,                     # get_portfolio total value
  "available_cash":    45.00,                      # buying power
  "open_positions": [
    {
      "ticker":             "NVDA",
      "average_buy_price":  "118.40",
      "strategy_type":      "MEAN_REVERSION",      # from Sheet lookup; default MEAN_REVERSION
      "entry_timestamp":    "2026-08-28T14:00:00Z",
      "dca_count":          0                      # DCA adds so far (0 = initial only)
    }
  ],
  "historicals": {                                 # daily OHLC list keyed by ticker
    "NVDA": [
      {"close_price":"118.40","high_price":"120.10","low_price":"117.20"}, ...
    ]
  },
  "quotes": {                                      # live quotes keyed by ticker
    "NVDA": {"last_trade_price": "119.55"}
  },
  "today_date":               "2026-08-30",
  "market_close_in_minutes":  240,                 # minutes until close; 9999 if unknown
  "position_states": {                             # from Worker /position-state-check
    "NVDA": {
      "peak_price": 122.10,                        # momentum only
      "breakdown": {"state":"NONE","last_updated_date":"2026-08-29"}
    }
  },
  "open_position_counts": {
    "mean_reversion": 2,
    "momentum":       1,
    "by_sector":      {"Semiconductors": 1}
  },
  "cooldown_history":   [{"ticker":"AMD","timestamp":"2026-08-29T10:00:00Z"}],
  "frequency_history":  [{"ticker":"NVDA","timestamp":"2026-08-20T14:00:00Z"}],
  "pnl_history":        [{"realized_pnl":-5.20,"is_last_24h":true}]
}
"""

import os
import sys
import json
import argparse
from datetime import datetime, timezone, timedelta

import requests

from indicators import (
    calculate_indicators,
    update_trailing_peak,
    evaluate_trend_breakdown,
    calculate_rs,
    check_safety_mode,
    adx_regime,
)

TICKER_SECTOR_MAP = {
    'NVDA':  ('Semiconductors',        'SMH'),
    'TSM':   ('Semiconductors',        'SMH'),
    'AVGO':  ('Semiconductors',        'SMH'),
    'AMD':   ('Semiconductors',        'SMH'),
    'MU':    ('Semiconductors',        'SMH'),
    'LLY':   ('Healthcare',            'XLV'),
    'JNJ':   ('Healthcare',            'XLV'),
    'ABBV':  ('Healthcare',            'XLV'),
    'UNH':   ('Healthcare',            'XLV'),
    'MRK':   ('Healthcare',            'XLV'),
    'BRK-B': ('Financials',            'XLF'),
    'JPM':   ('Financials',            'XLF'),
    'V':     ('Financials',            'XLF'),
    'MA':    ('Financials',            'XLF'),
    'BAC':   ('Financials',            'XLF'),
    'AAPL':  ('Technology',            'XLK'),
    'MSFT':  ('Technology',            'XLK'),
    'ORCL':  ('Technology',            'XLK'),
    'PLTR':  ('Technology',            'XLK'),
    'CSCO':  ('Technology',            'XLK'),
    'XOM':   ('Energy',                'XLE'),
    'CVX':   ('Energy',                'XLE'),
    'COP':   ('Energy',                'XLE'),
    'MPC':   ('Energy',                'XLE'),
    'PSX':   ('Energy',                'XLE'),
    'NEE':   ('Utilities',             'XLU'),
    'SO':    ('Utilities',             'XLU'),
    'CEG':   ('Utilities',             'XLU'),
    'DUK':   ('Utilities',             'XLU'),
    'VST':   ('Utilities',             'XLU'),
    'META':  ('Communication Services','XLC'),
    'GOOGL': ('Communication Services','XLC'),
    'NFLX':  ('Communication Services','XLC'),
    'TMUS':  ('Communication Services','XLC'),
    'T':     ('Communication Services','XLC'),
    'AMZN':  ('Consumer Discretionary','XLY'),
    'TSLA':  ('Consumer Discretionary','XLY'),
    'HD':    ('Consumer Discretionary','XLY'),
    'MCD':   ('Consumer Discretionary','XLY'),
    'TJX':   ('Consumer Discretionary','XLY'),
    'WMT':   ('Consumer Staples',      'XLP'),
    'COST':  ('Consumer Staples',      'XLP'),
    'PG':    ('Consumer Staples',      'XLP'),
    'KO':    ('Consumer Staples',      'XLP'),
    'PM':    ('Consumer Staples',      'XLP'),
}


# ── I/O helpers ──────────────────────────────────────────────────────────────

def get_live_price(quotes, ticker):
    q = quotes.get(ticker, {})
    for field in ('last_trade_price', 'last_extended_hours_trade_price', 'ask_price', 'bid_price'):
        v = q.get(field)
        if v and float(v) > 0:
            return float(v)
    return None


def compute_metrics(ticker, payload):
    hist = payload['historicals'].get(ticker)
    if not hist or len(hist) < 20:
        return None, f"insufficient historicals ({len(hist) if hist else 0} bars)"
    live = get_live_price(payload['quotes'], ticker)
    if live is None:
        return None, "no live price in quotes"
    try:
        return calculate_indicators(hist, live), None
    except Exception as e:
        return None, str(e)


def check_dedupe(worker_url, secret, ticker, signal_type):
    try:
        res = requests.post(
            f"{worker_url.rstrip('/')}/dedupe-check",
            json={"shared_secret": secret, "ticker": ticker,
                  "signal_type": signal_type, "lookback_hours": 20},
            timeout=5,
        )
        if res.status_code == 200:
            return res.json().get("already_reported", False)
    except Exception as e:
        print(f"  Warning: dedupe-check failed for {ticker}: {e}")
    return False


def update_dedupe(worker_url, secret, ticker, signal_type, daily_bar_date):
    try:
        requests.post(
            f"{worker_url.rstrip('/')}/dedupe-update",
            json={"shared_secret": secret, "ticker": ticker,
                  "signal_type": signal_type, "daily_bar_date": daily_bar_date},
            timeout=5,
        )
    except Exception as e:
        print(f"  Warning: dedupe-update failed for {ticker}: {e}")


def post_to_slack(worker_url, secret, ticker, action, live_price, stop,
                  dollar_amount, rs_value, notes, channel="#trade-signals"):
    payload = {
        "shared_secret": secret,
        "channel": channel,
        "ticker": ticker,
        "action": action,
        "live_price": live_price,
        "dollar_amount": dollar_amount,
        "rs_value": rs_value,
        "notes": notes,
    }
    if stop is not None:
        payload["stop"] = stop
    try:
        res = requests.post(f"{worker_url.rstrip('/')}/post-signal", json=payload, timeout=10)
        if res.status_code == 200 and res.json().get("posted"):
            print(f"  -> Slack: posted {action} signal for {ticker}")
        else:
            print(f"  -> Slack error for {ticker}: {res.status_code} {res.text}")
    except Exception as e:
        print(f"  -> Slack exception for {ticker}: {e}")


def log_to_sheet(sheet_url, secret, record):
    try:
        res = requests.post(sheet_url, json={"shared_secret": secret, **record},
                            timeout=10, allow_redirects=True)
        data = res.json() if res.headers.get('content-type', '').startswith('application/json') else {}
        if data.get("appended"):
            print(f"  -> Sheet: logged {record.get('recordType')} for {record.get('ticker')}")
        else:
            # Non-JSON on redirect hop = echo-hop pattern; write likely succeeded
            print(f"  -> Sheet: confirmation unclear (echo-hop); write likely succeeded for {record.get('ticker')}")
    except Exception as e:
        print(f"  -> Sheet exception for {record.get('ticker')}: {e}")


def update_position_state(worker_url, secret, ticker, state):
    try:
        requests.post(
            f"{worker_url.rstrip('/')}/position-state-update",
            json={"shared_secret": secret, "ticker": ticker, "state": state},
            timeout=5,
        )
    except Exception as e:
        print(f"  Warning: position-state-update failed for {ticker}: {e}")


# ── Guardrail helpers ─────────────────────────────────────────────────────────

def check_cooldown(ticker, cooldown_history):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
    for rec in cooldown_history:
        try:
            ts = datetime.fromisoformat(rec['timestamp'].replace('Z', '+00:00'))
            if rec['ticker'] == ticker and ts > cutoff:
                return True
        except Exception:
            pass
    return False


def check_frequency_cap(ticker, frequency_history):
    cutoff = datetime.now(timezone.utc) - timedelta(days=14)
    count = sum(
        1 for rec in frequency_history
        if rec.get('ticker') == ticker and
        _parse_ts(rec.get('timestamp', '')) > cutoff
    )
    return count >= 2


def _parse_ts(ts_str):
    try:
        return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)


# ── Entry evaluation ──────────────────────────────────────────────────────────

def evaluate_entries(payload, worker_url, worker_secret, sheet_url, sheet_secret):
    run        = payload['run']
    balance    = float(payload['portfolio_balance'])
    today      = payload['today_date']
    close_min  = payload.get('market_close_in_minutes', 9999)
    counts     = payload['open_position_counts']
    mr_count   = counts.get('mean_reversion', 0)
    mo_count   = counts.get('momentum', 0)
    by_sector  = counts.get('by_sector', {})
    cooldown   = payload.get('cooldown_history', [])
    freq       = payload.get('frequency_history', [])
    pnl_hist   = payload.get('pnl_history', [])

    safety_mode = check_safety_mode(pnl_hist, balance) if pnl_hist else False
    position_size = round(balance * 0.05, 2)

    if position_size < 5:
        print(f"Position size ${position_size} < $5 minimum — skipping all entries this cycle")
        return

    for ticker in payload['entry_tickers']:
        sector, etf = TICKER_SECTOR_MAP.get(ticker, (None, None))
        if not sector:
            print(f"{ticker}: not in sector map — skipping")
            continue

        metrics, err = compute_metrics(ticker, payload)
        if err:
            print(f"{ticker}: metrics error — {err}")
            continue

        etf_metrics, _ = compute_metrics(etf, payload)
        etf_return_10d  = etf_metrics.get('Return-10d-pct') if etf_metrics else None

        live          = metrics['Live']
        rsi           = metrics['RSI-14']
        pct_b         = metrics['%B']
        sma20         = metrics['SMA20']
        sma50         = metrics['SMA50']
        ema10         = metrics['EMA-10']
        atr           = metrics['ATR-14']
        is_new_20d    = metrics.get('Is-New-20d-High', False)
        stock_ret_10d = metrics.get('Return-10d-pct')
        rs = (calculate_rs(stock_ret_10d, etf_return_10d)
              if stock_ret_10d is not None and etf_return_10d is not None
              else None)

        # New indicators
        rsi2          = metrics.get('RSI-2', rsi)
        adx           = metrics.get('ADX-14', 0)
        trix          = metrics.get('TRIX-15', 0)
        trix_sig      = metrics.get('TRIX-Signal-9', 0)
        vol_ratio     = metrics.get('Volume-ratio')   # None if volume not in historicals
        regime        = adx_regime(adx)
        score         = metrics.get('CompositeScore', 0)

        confirming_str = (f"RSI-14={rsi}, RSI-2={rsi2:.1f}, %B={pct_b:.2f}, "
                          f"ADX={adx:.1f}({regime}), TRIX={trix:.4f}, Score={score}")

        def _block(reason, signal_type):
            print(f"  {ticker} BLOCKED ({signal_type}): {reason}")
            log_to_sheet(sheet_url, sheet_secret, {
                "timestamp":      datetime.now(timezone.utc).isoformat(),
                "run":            run,
                "recordType":     "BLOCKED",
                "signalType":     signal_type,
                "ticker":         ticker,
                "sector":         sector,
                "etfProxy":       etf,
                "action":         "",
                "price":          live,
                "size":           "",
                "signalReason":   "",
                "positionCount":  "",
                "exitReason":     "",
                "confirmingData": confirming_str,
                "costBasis":      "",
                "realizedPnLDollar":  "",
                "realizedPnLPercent": "",
                "holdDuration":   "",
                "blockReason":    reason,
            })

        def _fire_entry(signal_type, signal_reason, stop_loss, pending=False, pending_reason=""):
            label = "PENDING" if pending else "BUY"
            print(f"  🔥 {ticker} {label} ({signal_type}): {signal_reason}")

            if pending:
                log_to_sheet(sheet_url, sheet_secret, {
                    "timestamp":      datetime.now(timezone.utc).isoformat(),
                    "run":            run,
                    "recordType":     "BLOCKED",
                    "signalType":     signal_type,
                    "ticker":         ticker,
                    "sector":         sector,
                    "etfProxy":       etf,
                    "action":         "PENDING",
                    "price":          live,
                    "size":           position_size,
                    "signalReason":   signal_reason,
                    "positionCount":  "",
                    "exitReason":     "",
                    "confirmingData": confirming_str,
                    "costBasis":      "",
                    "realizedPnLDollar":  "",
                    "realizedPnLPercent": "",
                    "holdDuration":   "",
                    "blockReason":    pending_reason,
                })
                return

            already = check_dedupe(worker_url, worker_secret, ticker, signal_type)
            if already:
                print(f"  -> ALREADY REPORTED ({signal_type}); logging BLOCKED, skipping Slack")
                _block(f"already reported (dedupe)", signal_type)
                return

            log_to_sheet(sheet_url, sheet_secret, {
                "timestamp":      datetime.now(timezone.utc).isoformat(),
                "run":            run,
                "recordType":     "ENTRY",
                "signalType":     signal_type,
                "ticker":         ticker,
                "sector":         sector,
                "etfProxy":       etf,
                "action":         "INITIAL ENTRY",
                "price":          live,
                "size":           position_size,
                "signalReason":   signal_reason,
                "positionCount":  "1",
                "exitReason":     "",
                "confirmingData": confirming_str,
                "costBasis":      "",
                "realizedPnLDollar":  "",
                "realizedPnLPercent": "",
                "holdDuration":   "",
                "blockReason":    "",
            })

            post_to_slack(
                worker_url, worker_secret,
                ticker=ticker, action="BUY",
                live_price=live, stop=stop_loss,
                dollar_amount=position_size,
                rs_value=rs,
                notes=f"{signal_type} — {signal_reason}",
            )
            update_dedupe(worker_url, worker_secret, ticker, signal_type, today)

        # ── MEAN REVERSION ────────────────────────────────────────────────────
        # ADX regime: ranging (<20) favours MR; trending (>25) weakens MR thesis
        mr_triggers = []
        if rsi < 40:
            mr_triggers.append(f"RSI-14={rsi:.1f}<40")
        if rsi2 < 10:
            mr_triggers.append(f"RSI-2={rsi2:.1f}<10 (strong dip)")
        if pct_b < 0:
            mr_triggers.append(f"%B={pct_b:.2f}<0")
        touching_ema = ema10 and abs(live - ema10) / ema10 < 0.01
        if live > sma20 and touching_ema and rsi < 45:
            mr_triggers.append(f"trend-pullback(price>{sma20:.2f},EMA10={ema10:.2f},RSI={rsi:.1f})")

        if mr_triggers:
            trigger_str = "; ".join(mr_triggers)
            uptrend_a = live > sma20
            uptrend_b = etf_return_10d is not None and etf_return_10d > 0

            if not uptrend_a and not uptrend_b:
                _block(f"no uptrend context (price vs SMA20={sma20:.2f}; ETF 10d={etf_return_10d})", "MEAN_REVERSION")
            elif rs is not None and rs < -6:
                _block(f"RS={rs:.1f}% < -6% (RS filter)", "MEAN_REVERSION")
            elif regime == "trending" and adx > 30:
                # Strong trend (ADX>30) undercuts mean-reversion thesis — skip unless RSI-2 is extreme
                if rsi2 >= 10:
                    _block(f"ADX={adx:.1f} (strong trend) weakens MR thesis; RSI-2={rsi2:.1f} not extreme enough", "MEAN_REVERSION")
            elif score < -2:
                _block(f"composite score={score} too weak for MR entry (min -2)", "MEAN_REVERSION")
            elif safety_mode:
                _block("safety mode (24h losses >5% of account)", "MEAN_REVERSION")
            elif mr_count >= 4:
                _block("max concurrent mean-reversion positions (4)", "MEAN_REVERSION")
            elif by_sector.get(sector, 0) >= 2:
                _block(f"max per-sector positions (2) in {sector}", "MEAN_REVERSION")
            elif live < sma50 * 0.90:
                _block(f"trend filter: price {live:.2f} >10% below SMA50={sma50:.2f}", "MEAN_REVERSION")
            elif check_cooldown(ticker, cooldown):
                _block("cooldown: closed within last 48h", "MEAN_REVERSION")
            elif check_frequency_cap(ticker, freq):
                _block("frequency cap: 2+ entries in trailing 14 days", "MEAN_REVERSION")
            else:
                ctx = "a (price>SMA20)" if uptrend_a else "b (sector ETF 10d positive)"
                reason = f"{trigger_str}; uptrend context {ctx}; RS={rs:.1f}%"
                stop   = round(live - 1.75 * atr, 2)
                if close_min <= 120:
                    _fire_entry("MEAN_REVERSION", reason, stop, pending=True,
                                pending_reason="within 2h of close — reverify at open")
                else:
                    _fire_entry("MEAN_REVERSION", reason, stop)
        else:
            print(f"  {ticker}: no MR trigger (RSI={rsi:.1f}, %B={pct_b:.2f})")
            log_to_sheet(sheet_url, sheet_secret, {
                "timestamp":      datetime.now(timezone.utc).isoformat(),
                "run":            run,
                "recordType":     "BLOCKED",
                "signalType":     "MEAN_REVERSION",
                "ticker":         ticker,
                "sector":         sector,
                "etfProxy":       etf,
                "action":         "",
                "price":          live,
                "size":           "",
                "signalReason":   "",
                "positionCount":  "",
                "exitReason":     "",
                "confirmingData": confirming_str,
                "costBasis":      "",
                "realizedPnLDollar":  "",
                "realizedPnLPercent": "",
                "holdDuration":   "",
                "blockReason":    f"no raw trigger (RSI={rsi:.1f}, %B={pct_b:.2f})",
            })

        # ── MOMENTUM ──────────────────────────────────────────────────────────
        mo_triggers = []
        if rsi > 65:
            mo_triggers.append(f"RSI-14={rsi:.1f}>65")
        if pct_b > 1.0:
            mo_triggers.append(f"%B={pct_b:.2f}>1.0")
        if is_new_20d:
            mo_triggers.append(f"new 20d closing high at {live:.2f}")

        if mo_triggers:
            trigger_str = "; ".join(mo_triggers)
            if live <= sma20 or live <= sma50:
                _block(f"momentum context not confirmed (price {live:.2f} vs SMA20={sma20:.2f}, SMA50={sma50:.2f})", "MOMENTUM")
            elif trix <= 0 or trix <= trix_sig:
                # TRIX must be positive AND above its signal line to confirm momentum
                _block(f"TRIX not confirming momentum (TRIX={trix:.4f}, signal={trix_sig:.4f})", "MOMENTUM")
            elif is_new_20d and vol_ratio is not None and vol_ratio < 1.1:
                # New 20d high on below-average volume = weak breakout
                _block(f"new 20d high on low volume (vol ratio={vol_ratio:.2f} < 1.1)", "MOMENTUM")
            elif score < 2:
                _block(f"composite score={score} too weak for momentum entry (min +2)", "MOMENTUM")
            elif rs is None:
                _block("RS unavailable — cannot verify sector confirmation", "MOMENTUM")
            elif rs <= 4:
                _block(f"RS={rs:.1f}% <= +4% (insufficient RS)", "MOMENTUM")
            elif etf_return_10d is not None and etf_return_10d < 0 and rs <= 10:
                _block(f"sector not participating (ETF 10d={etf_return_10d:.1f}%) and RS={rs:.1f}% not >+10%", "MOMENTUM")
            elif safety_mode:
                _block("safety mode (24h losses >5% of account)", "MOMENTUM")
            elif mo_count >= 4:
                _block("max concurrent momentum positions (4)", "MOMENTUM")
            elif by_sector.get(sector, 0) >= 2:
                _block(f"max per-sector positions (2) in {sector}", "MOMENTUM")
            elif live < sma50 * 0.90:
                _block(f"trend filter: price {live:.2f} >10% below SMA50={sma50:.2f}", "MOMENTUM")
            elif check_cooldown(ticker, cooldown):
                _block("cooldown: closed within last 48h", "MOMENTUM")
            elif check_frequency_cap(ticker, freq):
                _block("frequency cap: 2+ entries in trailing 14 days", "MOMENTUM")
            else:
                waiver = ""
                if etf_return_10d is not None and etf_return_10d < 0 and rs > 10:
                    waiver = " (sector-participation waived: RS>+10%)"
                reason = f"{trigger_str}; RS={rs:.1f}%{waiver}"
                stop   = round(live - 1.75 * atr, 2)
                if close_min <= 120:
                    _fire_entry("MOMENTUM", reason, stop, pending=True,
                                pending_reason="within 2h of close — reverify at open")
                else:
                    _fire_entry("MOMENTUM", reason, stop)


# ── Exit evaluation ───────────────────────────────────────────────────────────

def evaluate_exits(payload, worker_url, worker_secret, sheet_url, sheet_secret):
    run             = payload['run']
    today           = payload['today_date']
    position_states = payload.get('position_states', {})

    for pos in payload.get('open_positions', []):
        ticker   = pos['ticker']
        strategy = pos.get('strategy_type', 'MEAN_REVERSION')
        entry_px = float(pos.get('average_buy_price', 0))
        sector, etf = TICKER_SECTOR_MAP.get(ticker, (None, None))
        if not sector:
            print(f"EXIT {ticker}: not in sector map — skipping")
            continue

        metrics, err = compute_metrics(ticker, payload)
        if err:
            print(f"EXIT {ticker}: metrics error — {err}")
            continue

        etf_metrics, _ = compute_metrics(etf, payload) if etf else (None, None)

        live          = metrics['Live']
        sma20         = metrics['SMA20']
        atr           = metrics['ATR-14']
        etf_return_10d = etf_metrics.get('Return-10d-pct') if etf_metrics else None
        cached        = position_states.get(ticker, {})
        new_state     = {}

        confirming_str = f"ATR={atr:.4f}, SMA20={sma20:.2f}"

        def _fire_exit(exit_type, label, confirming, cost_basis=None, realized_pnl=None,
                       realized_pct=None, hold_duration=None):
            print(f"  🚨 {ticker} {label} ({strategy})")
            log_to_sheet(sheet_url, sheet_secret, {
                "timestamp":      datetime.now(timezone.utc).isoformat(),
                "run":            run,
                "recordType":     "EXIT",
                "signalType":     strategy,
                "ticker":         ticker,
                "sector":         sector,
                "etfProxy":       etf,
                "action":         "SELL",
                "price":          live,
                "size":           "",
                "signalReason":   label,
                "positionCount":  "",
                "exitReason":     exit_type,
                "confirmingData": confirming,
                "costBasis":      cost_basis or "",
                "realizedPnLDollar":  realized_pnl or "",
                "realizedPnLPercent": realized_pct or "",
                "holdDuration":   hold_duration or "",
                "blockReason":    "",
            })
            already = check_dedupe(worker_url, worker_secret, ticker, f"exit-{exit_type}")
            if not already:
                post_to_slack(
                    worker_url, worker_secret,
                    ticker=ticker, action="SELL",
                    live_price=live, stop=None,
                    dollar_amount=None, rs_value=None,
                    notes=f"{label} | {confirming}",
                )
                update_dedupe(worker_url, worker_secret, ticker, f"exit-{exit_type}", today)
            else:
                print(f"  -> Exit for {ticker} already reported; skipping Slack re-post")

        if strategy == 'MOMENTUM':
            cached_peak = cached.get('peak_price')
            new_peak    = update_trailing_peak(cached_peak, live, entry_px)
            trail_stop  = round(new_peak - 1.75 * atr, 2)
            new_state['peak_price'] = new_peak

            if live <= trail_stop:
                severe    = live <= (trail_stop - atr)
                label     = 'EXIT SIGNAL (trailing-stop)' if severe else 'STOP WATCH (intraday)'
                confirming = f"peak={new_peak:.2f}, trailing_stop={trail_stop:.2f}, ATR={atr:.4f}"
                if severe:
                    _fire_exit('trailing-stop', label, confirming)
                else:
                    print(f"  {ticker} {label}: live={live:.2f}, stop={trail_stop:.2f}")
            else:
                print(f"  {ticker} MOMENTUM hold — peak={new_peak:.2f}, stop={trail_stop:.2f}, live={live:.2f}")

        else:  # MEAN_REVERSION
            stop_level = round(entry_px - 1.75 * atr, 2)
            if live <= stop_level:
                severe    = live <= (stop_level - atr)
                label     = 'EXIT SIGNAL (stop-loss)' if severe else 'STOP WATCH (intraday)'
                confirming = f"entry={entry_px:.2f}, stop={stop_level:.2f}, ATR={atr:.4f}"
                if severe:
                    _fire_exit('stop-loss', label, confirming)
                else:
                    print(f"  {ticker} {label}: live={live:.2f}, stop={stop_level:.2f}")

            price_below_sma20 = live < sma20
            sector_neg        = etf_return_10d is not None and etf_return_10d < 0
            cached_bd         = cached.get('breakdown')
            new_bd, is_exit   = evaluate_trend_breakdown(cached_bd, price_below_sma20, sector_neg, today)
            new_state['breakdown'] = new_bd

            if is_exit:
                confirming = (f"price<SMA20={sma20:.2f}: {price_below_sma20}; "
                              f"ETF 10d return={etf_return_10d}: {sector_neg}")
                _fire_exit('trend-breakdown (2-close confirmed)',
                           'EXIT SIGNAL (trend-breakdown)', confirming)
            elif new_bd.get('state') == 'DAY1_PENDING':
                print(f"  {ticker} TREND WATCH (day 1 of 2) — "
                      f"price<SMA20:{price_below_sma20}, sector neg:{sector_neg}")
            else:
                print(f"  {ticker} MEAN_REVERSION hold — stop={stop_level:.2f}, live={live:.2f}")

        if new_state:
            update_position_state(worker_url, worker_secret, ticker, new_state)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to JSON payload from Claude')
    args = parser.parse_args()

    worker_url    = os.environ.get('WORKER_URL', '')
    worker_secret = os.environ.get('WORKER_SHARED_SECRET', '')
    sheet_url     = os.environ.get('SHEET_LOG_WEBAPP_URL', '')
    sheet_secret  = os.environ.get('SHEET_LOG_SHARED_SECRET', '')

    if not worker_url:
        print("Warning: WORKER_URL not set — Slack calls will fail")
    if not sheet_url:
        print("Warning: SHEET_LOG_WEBAPP_URL not set — Sheet logging will fail")

    with open(args.input, 'r') as f:
        payload = json.load(f)

    run = payload.get('run', '?')
    print(f"\n{'='*60}")
    print(f"Run {run} — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Balance: ${payload.get('portfolio_balance', 0):.2f}  "
          f"Cash: ${payload.get('available_cash', 0):.2f}")
    print(f"{'='*60}")

    print("\n── EXIT EVALUATION (all open positions) ──")
    evaluate_exits(payload, worker_url, worker_secret, sheet_url, sheet_secret)

    print(f"\n── ENTRY SCAN (Run {run} tickers) ──")
    evaluate_entries(payload, worker_url, worker_secret, sheet_url, sheet_secret)

    print("\nScan complete.")


if __name__ == '__main__':
    main()
