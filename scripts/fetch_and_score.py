"""抓資料、計分、輸出 docs/data.json。

執行：FRED_API_KEY=... python scripts/fetch_and_score.py

安全與授權守則（對應 spec D4、第 8 節）：
- 金鑰只從環境變數讀取，不寫入任何檔案、不印出。
- 公開倉庫的 Actions 日誌任何人都看得到，所以日誌只印「序列代號、成功或失敗、最新觀測日期」，
  絕不印原始數值，也不印含金鑰的網址或完整的例外訊息。
- 原始資料不存進倉庫，只輸出計算後的分數與燈號。
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, time as dtime, timezone
from typing import Any, Optional

import pandas as pd
import yaml

import pipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "config", "indicators.yml")
OUTPUT_PATH = os.path.join(ROOT, "docs", "data.json")

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"

# pipeline 的序列代號 -> FRED 代號
FRED_SERIES = {
    "hy_oas": "BAMLH0A0HYM2",
    "vix": "VIXCLS",
    "vix3m": "VXVCLS",
    "nfci": "NFCI",
    "stlfsi4": "STLFSI4",
    "walcl": "WALCL",
    "dff": "DFF",
    "dollar": "DTWEXBGS",
    "t10y3m": "T10Y3M",
    "sahm": "SAHMREALTIME",
    "icsa": "ICSA",
}

# pipeline 的序列代號 -> Yahoo 代號（依序嘗試，第一個有資料的為準）
YAHOO_SERIES = {
    "spx": ["^GSPC"],
    "twii": ["^TWII"],
    "gold": ["GC=F", "GLD"],
}

# 各市場「當日資料定稿」的時間（UTC，保留緩衝）。在此之前，當天的那一筆是盤中資料，必須丟掉。
#   S&P 500：美東 16:00 收盤 = UTC 21:00（冬令）；台股：13:30 收盤 = UTC 05:30；
#   黃金期貨（GC=F）：美東 17:00 結束 = UTC 22:00。
MARKET_FINAL_UTC = {"spx": (21, 30), "twii": (6, 0), "gold": (22, 15)}


def drop_unfinished_last_bar(s: Optional[pd.Series], now_utc: datetime, final_hm: tuple[int, int]) -> Optional[pd.Series]:
    """若最後一筆是「今天」且還沒到定稿時間，就是盤中資料，丟掉；日期在未來的也丟掉。"""
    if s is None or s.empty:
        return s
    last = s.index[-1].date()
    today = now_utc.date()
    final_at = datetime.combine(today, dtime(*final_hm), tzinfo=timezone.utc)
    if last > today or (last == today and now_utc < final_at):
        return s.iloc[:-1] if len(s) > 1 else None
    return s


def log(message: str) -> None:
    """日誌只允許安全的文字：代號、狀態、日期。不得放數值或網址。"""
    print(message, flush=True)


# ---------------------------------------------------------------------------
# FRED
# ---------------------------------------------------------------------------


def parse_fred_observations(payload: dict[str, Any]) -> Optional[pd.Series]:
    """把 FRED API 的 JSON 轉成 Series；'.' 代表缺值。"""
    rows = payload.get("observations")
    if not rows:
        return None
    idx, vals = [], []
    for r in rows:
        try:
            v = float(r["value"])
        except (ValueError, KeyError, TypeError):
            continue  # '.' 或格式不對的列直接略過
        idx.append(pd.Timestamp(r["date"]))
        vals.append(v)
    if not vals:
        return None
    return pd.Series(vals, index=pd.DatetimeIndex(idx)).sort_index()


def fetch_fred(series_id: str, api_key: str, start: str, retries: int = 2) -> Optional[pd.Series]:
    query = urllib.parse.urlencode(
        {"series_id": series_id, "api_key": api_key, "file_type": "json", "observation_start": start}
    )
    url = f"{FRED_URL}?{query}"
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310（固定的 https 網址）
                payload = json.loads(resp.read().decode("utf-8"))
            return parse_fred_observations(payload)
        except Exception as exc:  # 只記錄例外類型，避免訊息中帶出含金鑰的網址
            if attempt == retries:
                log(f"  FRED {series_id}: 失敗（{type(exc).__name__}）")
                return None
            time.sleep(2 * (attempt + 1))
    return None


# ---------------------------------------------------------------------------
# Yahoo（yfinance 為社群套件，可能失效；失敗時回傳 None，由缺值規則處理）
# ---------------------------------------------------------------------------


def fetch_yahoo(symbols: list[str], period: str = "3y") -> tuple[Optional[pd.Series], Optional[str]]:
    try:
        import yfinance as yf
    except Exception as exc:
        log(f"  yfinance 無法載入（{type(exc).__name__}）")
        return None, None
    for sym in symbols:
        try:
            hist = yf.Ticker(sym).history(period=period, auto_adjust=False)
            close = hist["Close"].dropna()
            if close.empty:
                continue
            idx = pd.DatetimeIndex(close.index)
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            s = pd.Series(close.to_numpy(dtype=float), index=idx.normalize())
            s = s[s > 0]  # 價格必須為正，否則視為壞資料
            if len(s) >= 30:
                return s, sym
        except Exception as exc:
            log(f"  Yahoo {sym}: 失敗（{type(exc).__name__}）")
    return None, None


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def load_all(api_key: str, today: datetime, now_utc: Optional[datetime] = None) -> dict[str, Optional[pd.Series]]:
    now_utc = now_utc or datetime.now(timezone.utc)
    start = (today - pd.Timedelta(days=500)).strftime("%Y-%m-%d")
    series: dict[str, Optional[pd.Series]] = {}
    for key, fred_id in FRED_SERIES.items():
        s = fetch_fred(fred_id, api_key, start)
        series[key] = s
        if s is not None:
            log(f"  FRED {fred_id}: 成功，最新觀測日期 {s.index[-1].strftime('%Y-%m-%d')}")
    for key, symbols in YAHOO_SERIES.items():
        s, used = fetch_yahoo(symbols)
        if s is not None:
            trimmed = drop_unfinished_last_bar(s, now_utc, MARKET_FINAL_UTC[key])
            if trimmed is None or len(trimmed) != len(s):
                log(f"  Yahoo {used}: 最後一筆尚未收盤，已丟棄")
            s = trimmed
        series[key] = s
        if s is not None:
            log(f"  Yahoo {used}: 成功，最新觀測日期 {s.index[-1].strftime('%Y-%m-%d')}")
        else:
            log(f"  Yahoo {'/'.join(symbols)}: 無資料")
    return series


def write_atomic(path: str, data: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def main() -> int:
    api_key = os.environ.get("FRED_API_KEY", "").strip()
    if not api_key:
        log("錯誤：沒有設定環境變數 FRED_API_KEY（請在 GitHub Secrets 設定）。")
        return 1

    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    now = datetime.now(timezone.utc)
    log(f"開始抓資料：{now.strftime('%Y-%m-%d %H:%M')} UTC")
    series = load_all(api_key, pd.Timestamp(now.date()), now)

    public = pipeline.build(series, cfg, now.date(), now.strftime("%Y-%m-%dT%H:%M:%SZ"))
    log(f"A 狀態：{public['regime']['status']}，完整度 {public['regime']['completeness']}")
    log(f"B 狀態：{public['stress']['status']}，完整度 {public['stress']['completeness']}")

    # 兩個分數都算不出來，視為抓取失敗：保留上一份 data.json，不寫入空資料
    if public["regime"]["status"] == "insufficient" and public["stress"]["status"] == "insufficient":
        log("兩個分數的資料都不足，保留上一份 data.json，不覆寫。")
        return 1

    write_atomic(OUTPUT_PATH, public)
    log(f"已寫入 {os.path.relpath(OUTPUT_PATH, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
