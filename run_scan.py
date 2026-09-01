"""
run_scan.py  —  Trade signal engine. Accepts account state from Claude
(balance, positions, guardrail history), fetches market data via yfinance,
runs all calculations, posts to Slack, and logs to Google Sheets.

Claude fetches ONLY account-specific data from Robinhood (no historicals,
no quotes). GitHub fetches all market data here at runtime — free, fast,
no Claude tokens spent on price data.

Usage:
    python run_scan.py --input payload.json

Required env vars:
    WORKER_URL                  Cloudflare Worker base URL
    WORKER_SHARED_SECRET        Auth secret for Worker endpoints
    SHEET_LOG_WEBAPP_URL        Google Apps Script Web App URL
    SHEET_LOG_SHARED_SECRET     Auth secret for Sheet endpoint
    TICKER_CONFIG               Comma-separated TICKER:ETF pairs (repo variable)
                                e.g. NVDA:SMH,TSM:SMH,...,PM:XLP
"""

import os
import sys
import json
import argparse
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests
import yfinance as yf

from indicators import (
    calculate_indicators,
    update_trailing_peak,
    evaluate_trend_breakdown,
    calculate_rs,
    check_safety_mode,
    adx_regime,
)

ETF_SECTOR_NAMES = {
    'SMH': 'Semiconductors',
    'XLV': 'Healthcare',
    'XLF': 'Financials',
    'XLK': 'Technology',
    'XLE': 'Energy',
    'XLU': 'Utilities',
    'XLC': 'Communication Services',
    'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples',
}


def build_ticker_sector_map():
    """Builds {TICKER: (SectorName, ETF)} from TICKER_CONFIG env var at runtime."""
    config = os.environ.get('TICKER_CONFIG', '')
    result = {}
    for pair in config.split(','):
        pair = pair.strip()
        if ':' not in pair:
            continue
        ticker, etf = pair.split(':', 1)
        ticker, etf = ticker.strip().upper(), etf.strip().upper()
        sector = ETF_SECTOR_NAMES.get(etf, etf)
        result[ticker] = (sector, etf)
    return result


def fetch_market_data(tickers):
    """
    Batch-fetches 1y of daily OHLCV + latest price for all tickers via yfinance.
    Returns (historicals_dict, quotes_dict) in the same shape the rest of the
    script expects — no Robinhood calls needed.
    """
    historicals = {}
    quotes = {}
    if not tickers:
        return historicals, quotes

    print(f"Fetching market data for {len(tickers)} tickers via yfinance...")
    try:
        raw = yf.download(
            tickers=' '.join(tickers),
            period='1y',
            interval='1d',
            progress=False,
            auto_adjust=True,
            group_by='ticker',
        )
    except Exception as e:
        print(f"  yfinance batch download failed: {e}")
        return historicals, quotes

    for ticker in tickers:
        try:
            if len(tickers) > 1:
                if isinstance(raw.columns, pd.MultiIndex):
                    # Try (field, ticker) order first (group_by='column' style)
                    if ('Close', ticker) in raw.columns:
                        df = raw.xs(ticker, axis=1, level=1)
                    # Then try (ticker, field) order (group_by='ticker' style)
                    elif (ticker, 'Close') in raw.columns:
                        df = raw.xs(ticker, axis=1, level=0)
                    else:
                        print(f"  {ticker}: no data in batch response")
                        continue
                else:
                    print(f"  {ticker}: unexpected column format")
                    continue
            else:
                df = raw.copy()
            df = df.dropna(subset=['Close'])
            if df.empty or len(df) < 20:
                print(f"  {ticker}: insufficient yfinance data ({len(df)} bars)")
                continue
            historicals[ticker] = [
                {
                    'close_price': str(round(float(row['Close']), 4)),
                    'high_price':  str(round(float(row['High']),  4)),
                    'low_price':   str(round(float(row['Low']),   4)),
                    'volume':      str(int(row['Volume'])) if 'Volume' in row and row['Volume'] == row['Volume'] else '0',
                }
                for _, row in df.iterrows()
            ]
            # Use fast_info for a fresher intraday price; fall back to last close
            try:
                live = yf.Ticker(ticker).fast_info.last_price
                if not live or live != live:
                    raise ValueError("no fast_info price")
            except Exception:
                live = float(df['Close'].iloc[-1])
            quotes[ticker] = {'last_trade_price': str(round(live, 4))}
        except Exception as e:
            print(f"  {ticker}: yfinance parse error — {e}")

    print(f"  Market data ready: {len(historicals)} tickers with historicals, {len(quotes)} with quotes")
    return historicals, quotes


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


def evaluate_entries(payload, worker_url, worker_secret, sheet_url, sheet_secret):
    TICKER_SECTOR_MAP = build_ticker_sector_map()
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

        rsi2      = metrics.get('RSI-2', rsi)
        adx       = metrics.get('ADX-14', 0)
        trix      = metrics.get('TRIX-15', 0)
        trix_sig  = metrics.get('TRIX-Signal-9', 0)
        vol_ratio = metrics.get('Volume-ratio')
        regime    = adx_regime(adx)
        score     = metrics.get('CompositeScore', 0)

        confirming_str = (f"RSI-14={rsi}, RSI-2={rsi2:.1f}, %B={pct_b:.2f}, "
                          f"ADX={adx:.1f}({regime}), TRIX={trix:.4f}, Score={score}")

        def _block(reason, signal_type):
            print(f"  {ticker} BLOCKED ({signal_type}): {reason}")
            log_to_sheet(sheet_url, sheet_secret, {
                "timestamp": datetime.now(timezone.utc).isoformat(), "run": run,
                "recordType": "BLOCKED", "signalType": signal_type, "ticker": ticker,
                "sector": sector, "etfProxy": etf, "action": "", "price": live,
                "size": "", "signalReason": "", "positionCount": "", "exitReason": "",
                "confirmingData": confirming_str, "costBasis": "",
                "realizedPnLDollar": "", "realizedPnLPercent": "",
                "holdDuration": "", "blockReason": reason,
            })

        def _fire_entry(signal_type, signal_reason, stop_loss, pending=False, pending_reason=""):
            label = "PENDING" if pending else "BUY"
            print(f"  {ticker} {label} ({signal_type}): {signal_reason}")
            if pending:
                log_to_sheet(sheet_url, sheet_secret, {
                    "timestamp": datetime.now(timezone.utc).isoformat(), "run": run,
                    "recordType": "BLOCKED", "signalType": signal_type, "ticker": ticker,
                    "sector": sector, "etfProxy": etf, "action": "PENDING", "price": live,
                    "size": position_size, "signalReason": signal_reason, "positionCount": "",
                    "exitReason": "", "confirmingData": confirming_str, "costBasis": "",
                    "realizedPnLDollar": "", "realizedPnLPercent": "",
                    "holdDuration": "", "blockReason": pending_reason,
                })
                return
            already = check_dedupe(worker_url, worker_secret, ticker, signal_type)
            if already:
                print(f"  -> ALREADY REPORTED ({signal_type}); logging BLOCKED, skipping Slack")
                _block(f"already reported (dedupe)", signal_type)
                return
            log_to_sheet(sheet_url, sheet_secret, {
                "timestamp": datetime.now(timezone.utc).isoformat(), "run": run,
                "recordType": "ENTRY", "signalType": signal_type, "ticker": ticker,
                "sector": sector, "etfProxy": etf, "action": "INITIAL ENTRY", "price": live,
                "size": position_size, "signalReason": signal_reason, "positionCount": "1",
                "exitReason": "", "confirmingData": confirming_str, "costBasis": "",
                "realizedPnLDollar": "", "realizedPnLPercent": "", "holdDuration": "", "blockReason": "",
            })
            post_to_slack(worker_url, worker_secret, ticker=ticker, action="BUY",
                          live_price=live, stop=stop_loss, dollar_amount=position_size,
                          rs_value=rs, notes=f"{signal_type} — {signal_reason}")
            update_dedupe(worker_url, worker_secret, ticker, signal_type, today)

        # MEAN REVERSION
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
            elif regime == "trending" and adx > 30 and rsi2 >= 10:
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
                "timestamp": datetime.now(timezone.utc).isoformat(), "run": run,
                "recordType": "BLOCKED", "signalType": "MEAN_REVERSION", "ticker": ticker,
                "sector": sector, "etfProxy": etf, "action": "", "price": live,
                "size": "", "signalReason": "", "positionCount": "", "exitReason": "",
                "confirmingData": confirming_str, "costBasis": "",
                "realizedPnLDollar": "", "realizedPnLPercent": "", "holdDuration": "",
                "blockReason": f"no raw trigger (RSI={rsi:.1f}, %B={pct_b:.2f})",
            })

        # MOMENTUM
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
                _block(f"TRIX not confirming momentum (TRIX={trix:.4f}, signal={trix_sig:.4f})", "MOMENTUM")
            elif is_new_20d and vol_ratio is not None and vol_ratio < 1.1:
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


def evaluate_exits(payload, worker_url, worker_secret, sheet_url, sheet_secret):
    TICKER_SECTOR_MAP = build_ticker_sector_map()
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

        live           = metrics['Live']
        sma20          = metrics['SMA20']
        atr            = metrics['ATR-14']
        etf_return_10d = etf_metrics.get('Return-10d-pct') if etf_metrics else None
        cached         = position_states.get(ticker, {})
        new_state      = {}

        confirming_str = f"ATR={atr:.4f}, SMA20={sma20:.2f}"

        def _fire_exit(exit_type, label, confirming, cost_basis=None, realized_pnl=None,
                       realized_pct=None, hold_duration=None):
            print(f"  {ticker} {label} ({strategy})")
            log_to_sheet(sheet_url, sheet_secret, {
                "timestamp": datetime.now(timezone.utc).isoformat(), "run": run,
                "recordType": "EXIT", "signalType": strategy, "ticker": ticker,
                "sector": sector, "etfProxy": etf, "action": "SELL", "price": live,
                "size": "", "signalReason": label, "positionCount": "",
                "exitReason": exit_type, "confirmingData": confirming,
                "costBasis": cost_basis or "", "realizedPnLDollar": realized_pnl or "",
                "realizedPnLPercent": realized_pct or "", "holdDuration": hold_duration or "",
                "blockReason": "",
            })
            # Always post exit signals to Slack — position is still open so signal is still valid.
            # Dedupe is intentionally skipped for exits.
            qty = float(pos.get('quantity', 0) or 0)
            dollar_val = round(qty * live, 2) if qty and live else None
            post_to_slack(worker_url, worker_secret, ticker=ticker, action="SELL",
                          live_price=live, stop=None, dollar_amount=dollar_val, rs_value=None,
                          notes=f"{label} | {confirming}")

        if strategy == 'MOMENTUM':
            cached_peak = cached.get('peak_price')
            new_peak    = update_trailing_peak(cached_peak, live, entry_px)
            trail_stop  = round(new_peak - 1.75 * atr, 2)
            new_state['peak_price'] = new_peak
            if live <= trail_stop:
                severe     = live <= (trail_stop - atr)
                label      = 'EXIT SIGNAL (trailing-stop)' if severe else 'STOP WATCH (intraday)'
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
                severe     = live <= (stop_level - atr)
                label      = 'EXIT SIGNAL (stop-loss)' if severe else 'STOP WATCH (intraday)'
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
                _fire_exit('trend-breakdown (2-close confirmed)', 'EXIT SIGNAL (trend-breakdown)', confirming)
            elif new_bd.get('state') == 'DAY1_PENDING':
                print(f"  {ticker} TREND WATCH (day 1 of 2) — "
                      f"price<SMA20:{price_below_sma20}, sector neg:{sector_neg}")
            else:
                print(f"  {ticker} MEAN_REVERSION hold — stop={stop_level:.2f}, live={live:.2f}")

        if new_state:
            update_position_state(worker_url, worker_secret, ticker, new_state)


def normalize_payload(payload, worker_url, worker_secret):
    """
    Translate the simplified cloud-routine payload format into the shape
    the rest of the script expects. The simplified format has:
      run_id, account{buying_power, portfolio_value, cash}, positions[], open_orders[]
    The legacy format (and what the script reads internally) has:
      run, portfolio_balance, available_cash, open_positions[], position_states{}, etc.
    """
    if 'run' in payload:
        return payload  # already in legacy format

    run = payload.get('run_id', '?')
    account = payload.get('account', {})
    positions_raw = payload.get('positions', [])

    # Fetch per-ticker position state from Worker to infer MOMENTUM vs MEAN_REVERSION
    position_states = {}
    if worker_url and worker_secret:
        tickers = [p.get('symbol', p.get('ticker', '')) for p in positions_raw if p.get('symbol') or p.get('ticker')]
        for ticker in tickers:
            try:
                res = requests.post(
                    f"{worker_url.rstrip('/')}/position-state-check",
                    json={"shared_secret": worker_secret, "ticker": ticker},
                    timeout=5,
                )
                if res.status_code == 200:
                    data = res.json()
                    if data.get('found'):
                        position_states[ticker] = data.get('state', {})
            except Exception as e:
                print(f"  Warning: could not fetch position state for {ticker}: {e}")

    def infer_strategy(ticker):
        state = position_states.get(ticker, {})
        if 'peak_price' in state:
            return 'MOMENTUM'
        if 'breakdown' in state:
            return 'MEAN_REVERSION'
        return 'MEAN_REVERSION'

    TICKER_SECTOR_MAP = build_ticker_sector_map()

    open_positions = []
    by_sector = {}
    mr_count = 0
    mo_count = 0
    for p in positions_raw:
        ticker = p.get('symbol', p.get('ticker', ''))
        if not ticker:
            continue
        strategy = infer_strategy(ticker)
        open_positions.append({
            'ticker': ticker,
            'average_buy_price': str(p.get('avg_cost', p.get('average_buy_price', 0))),
            'quantity': str(p.get('qty', p.get('quantity', 0))),
            'strategy_type': strategy,
            'entry_timestamp': '',
            'dca_count': 0,
        })
        if strategy == 'MEAN_REVERSION':
            mr_count += 1
        else:
            mo_count += 1
        sector, _ = TICKER_SECTOR_MAP.get(ticker, (None, None))
        if sector:
            by_sector[sector] = by_sector.get(sector, 0) + 1

    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    market_close_utc = datetime.now(timezone.utc).replace(
        hour=20, minute=0, second=0, microsecond=0)
    close_min = max(0, int((market_close_utc - datetime.now(timezone.utc)).total_seconds() / 60))

    return {
        'run': run,
        'portfolio_balance': float(account.get('portfolio_value', 0)),
        'available_cash': float(account.get('buying_power', account.get('cash', 0))),
        'open_positions': open_positions,
        'open_orders': payload.get('open_orders', []),
        'historicals': payload.get('historicals', {}),
        'quotes': payload.get('quotes', {}),
        'entry_tickers': payload.get('entry_tickers', []),
        'position_states': position_states,
        'open_position_counts': {
            'mean_reversion': mr_count,
            'momentum': mo_count,
            'by_sector': by_sector,
        },
        'cooldown_history': [],
        'frequency_history': [],
        'pnl_history': [],
        'today_date': today,
        'market_close_in_minutes': close_min,
    }


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

    payload = normalize_payload(payload, worker_url, worker_secret)

    run = payload.get('run', '?')
    print(f"\n{'='*60}")
    print(f"Run {run} — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Balance: ${payload.get('portfolio_balance', 0):.2f}  "
          f"Cash: ${payload.get('available_cash', 0):.2f}")
    print(f"{'='*60}")

    # Build ticker list from TICKER_CONFIG repo variable
    ticker_config = os.environ.get('TICKER_CONFIG', '')
    if ticker_config:
        entry_tickers = [pair.split(':')[0].strip().upper()
                         for pair in ticker_config.split(',') if ':' in pair]
        etfs = list(dict.fromkeys(
            pair.split(':')[1].strip().upper()
            for pair in ticker_config.split(',') if ':' in pair
        ))
        payload['entry_tickers'] = entry_tickers
        print(f"Tickers from TICKER_CONFIG: {len(entry_tickers)} stocks + {len(etfs)} ETFs")
    else:
        entry_tickers = payload.get('entry_tickers', [])
        etfs = []
        print("Warning: TICKER_CONFIG not set — using entry_tickers from payload")

    # Add any open position tickers not already covered
    open_tickers = [p['ticker'] for p in payload.get('open_positions', [])]
    all_fetch = list(dict.fromkeys(entry_tickers + etfs + open_tickers))

    # Fetch all market data via yfinance (no Robinhood historicals needed)
    historicals, quotes = fetch_market_data(all_fetch)
    payload['historicals'] = historicals
    payload['quotes'] = quotes

    print("\n── EXIT EVALUATION (all open positions) ──")
    evaluate_exits(payload, worker_url, worker_secret, sheet_url, sheet_secret)

    print(f"\n── ENTRY SCAN (Run {run} — {len(entry_tickers)} tickers) ──")
    evaluate_entries(payload, worker_url, worker_secret, sheet_url, sheet_secret)

    print("\nScan complete.")


if __name__ == '__main__':
    main()
