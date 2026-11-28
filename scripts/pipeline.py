"""資料處理管線：原始時間序列 -> 特徵 -> 分數與旗標 -> 公開輸出。

不碰網路與檔案（抓取在 fetch_and_score.py），因此可以用模擬資料完整測試。

輸入 series 是 dict，key 為下列代號，value 為以日期為索引的 pandas.Series（缺資料給 None）：
  spx, twii, gold, hy_oas, vix, vix3m, nfci, stlfsi4, walcl, dff, dollar, t10y3m, sahm, icsa

隱私與授權（D4）：這個模組的任何輸出（含 print／log）都不得包含第三方原始數值。
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

import numpy as np
import pandas as pd

import scoring

# 各序列的更新頻率，用於判斷資料是否過舊（對應 config 的 freshness_days）
FREQ = {
    "spx": "daily", "twii": "daily", "gold": "daily", "hy_oas": "daily", "vix": "daily",
    "vix3m": "daily", "dff": "daily", "dollar": "daily", "t10y3m": "daily",
    "nfci": "weekly", "stlfsi4": "weekly", "walcl": "weekly", "icsa": "weekly",
    "sahm": "monthly",
}

DAYS_3M = 91  # 3 個月，以日曆天計


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def clean(s: Optional[pd.Series]) -> Optional[pd.Series]:
    """去除缺值、重複日期，依日期排序；空的回傳 None。"""
    if s is None:
        return None
    s = pd.Series(s).dropna()
    if s.empty:
        return None
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.astype(float)


def is_fresh(s: Optional[pd.Series], key: str, cfg: dict[str, Any], today: date, at: Optional[pd.Timestamp] = None) -> bool:
    """最新一筆的觀測日期是否在有效期限內（自觀測日期起算）。"""
    if s is None or s.empty:
        return False
    ref = pd.Timestamp(at if at is not None else today).normalize()
    age = (ref - s.index[-1].normalize()).days
    return 0 <= age <= cfg["freshness_days"][FREQ[key]]


def asof_value(s: Optional[pd.Series], ts: pd.Timestamp) -> Optional[tuple[pd.Timestamp, float]]:
    """取得不晚於 ts 的最近一筆 (日期, 數值)。"""
    if s is None:
        return None
    sub = s.loc[:ts]
    if sub.empty:
        return None
    return sub.index[-1], float(sub.iloc[-1])


def _all_indicator_ids(cfg: dict[str, Any]) -> list[str]:
    ids = []
    for g in cfg["regime"]["dimensions"] + cfg["stress"]["blocks"]:
        ids.extend(i["id"] for i in g["indicators"])
    return ids


def _pct(now: float, before: float) -> Optional[float]:
    if before == 0:
        return None
    return (now / before - 1.0) * 100.0


# ---------------------------------------------------------------------------
# 特徵
# ---------------------------------------------------------------------------


def compute_features(S: dict[str, Optional[pd.Series]], cfg: dict[str, Any], today: date):
    """回傳 (特徵 dict, 各序列最新觀測日期 dict)。資料缺或過舊的特徵為 None。"""
    S = {k: clean(v) for k, v in S.items()}
    f: dict[str, Optional[float]] = {i: None for i in _all_indicator_ids(cfg)}
    fresh = lambda k: is_fresh(S.get(k), k, cfg, today)  # noqa: E731

    # --- 股市（S&P 500）---
    spx = S.get("spx")
    if fresh("spx"):
        if len(spx) >= 221:
            sma200 = spx.rolling(200).mean()
            f["spx_vs_200dma"] = _pct(spx.iloc[-1], sma200.iloc[-1])
            f["spx_200dma_slope"] = _pct(sma200.iloc[-1], sma200.iloc[-21])
        if len(spx) >= 240:
            peak = spx.iloc[-252:].max()
            f["spx_drawdown_52w"] = (1.0 - spx.iloc[-1] / peak) * 100.0

    # --- 台股 ---
    twii = S.get("twii")
    if fresh("twii") and len(twii) >= 60:
        sma60 = twii.rolling(60).mean()
        f["twii_vs_60dma"] = _pct(twii.iloc[-1], sma60.iloc[-1])

    # --- 流動性與政策 ---
    walcl = S.get("walcl")
    if fresh("walcl") and len(walcl) >= 14:
        f["fed_balance_sheet_13w"] = _pct(walcl.iloc[-1], walcl.iloc[-14])

    dff = S.get("dff")
    if fresh("dff"):
        prev = asof_value(dff, dff.index[-1] - pd.Timedelta(days=DAYS_3M))
        if prev is not None:
            f["fed_funds_3m_change"] = (dff.iloc[-1] - prev[1]) * 100.0  # 單位：bp

    dollar = S.get("dollar")
    if fresh("dollar"):
        prev = asof_value(dollar, dollar.index[-1] - pd.Timedelta(days=DAYS_3M))
        if prev is not None:
            f["broad_dollar_3m"] = _pct(dollar.iloc[-1], prev[1])

    # --- 景氣 ---
    if fresh("t10y3m"):
        f["yield_curve_10y3m"] = float(S["t10y3m"].iloc[-1])
    if fresh("sahm"):
        f["sahm_rule"] = float(S["sahm"].iloc[-1])
    icsa = S.get("icsa")
    if fresh("icsa") and len(icsa) >= 17:
        ma4 = icsa.rolling(4).mean()
        f["initial_claims_13w"] = _pct(ma4.iloc[-1], ma4.iloc[-14])

    # --- 避險情緒：VIX / VIX3M（取兩者共同的最新日期）---
    vix, vix3m = S.get("vix"), S.get("vix3m")
    if vix is not None and vix3m is not None:
        both = pd.concat([vix, vix3m], axis=1, join="inner").dropna()
        if not both.empty:
            ts = both.index[-1]
            age = (pd.Timestamp(today).normalize() - ts.normalize()).days
            if 0 <= age <= cfg["freshness_days"]["daily"] and both.iloc[-1, 1] != 0:
                f["vix_term_structure"] = float(both.iloc[-1, 0] / both.iloc[-1, 1])

    # --- 金融壓力（B）---
    for feat, key in (("hy_oas", "hy_oas"), ("nfci", "nfci"), ("stlfsi4", "stlfsi4"), ("vix", "vix")):
        if fresh(key):
            f[feat] = float(S[key].iloc[-1])

    as_of = {k: (None if S.get(k) is None else S[k].index[-1].strftime("%Y-%m-%d")) for k in FREQ}
    return f, as_of


# ---------------------------------------------------------------------------
# B 分數的歷史（供「轉折確認」使用）
# ---------------------------------------------------------------------------


def stress_history(S: dict[str, Optional[pd.Series]], cfg: dict[str, Any], today: date, n: int = 60) -> list[dict[str, Any]]:
    """重算過去 n 個交易日的 B 分數。注意：NFCI、STLFSI4 的歷史值會被修正，這裡用的是目前版本。"""
    S = {k: clean(v) for k, v in S.items()}
    spx = S.get("spx")
    if spx is not None:
        days = list(spx.index[-n:])
    else:
        days = list(pd.bdate_range(end=pd.Timestamp(today), periods=n))
    out = []
    for d in days:
        feats: dict[str, Optional[float]] = {}
        for feat, key in (("hy_oas", "hy_oas"), ("nfci", "nfci"), ("stlfsi4", "stlfsi4"), ("vix", "vix")):
            hit = asof_value(S.get(key), d)
            ok = hit is not None and (d.normalize() - hit[0].normalize()).days <= cfg["freshness_days"][FREQ[key]]
            feats[feat] = hit[1] if ok else None
        res = scoring.score_stress(cfg, feats)
        out.append(
            {
                "date": d,
                "total": res["score"],
                "block": res["blocks"][0]["score"] if res["blocks"][0].get("included") else None,
            }
        )
    return out


# ---------------------------------------------------------------------------
# 旗標
# ---------------------------------------------------------------------------


def _ret_trading_days(s: Optional[pd.Series], key: str, cfg: dict[str, Any], today: date, n: int) -> Optional[float]:
    if s is None or not is_fresh(s, key, cfg, today) or len(s) < n + 1:
        return None
    return _pct(s.iloc[-1], s.iloc[-1 - n])


def compute_flags_detailed(S, cfg, today, features, stress, hist):
    """回傳 (旗標, 內部細節)。細節含第三方資料衍生的數值，只供回測診斷，不得寫入公開輸出或日誌。"""
    S = {k: clean(v) for k, v in S.items()}
    fg = cfg["flags"]["gold_equity_drop"]
    tp = cfg["flags"]["turning_point"]

    # --- 股金同跌 ---
    spx_ret = _ret_trading_days(S.get("spx"), "spx", cfg, today, fg["window_trading_days"])
    gold_ret = _ret_trading_days(S.get("gold"), "gold", cfg, today, fg["window_trading_days"])
    hy = S.get("hy_oas")
    hy_chg = None
    if hy is not None and is_fresh(hy, "hy_oas", cfg, today):
        prev = asof_value(hy, hy.index[-1] - pd.Timedelta(days=7))
        if prev is not None:
            hy_chg = float(hy.iloc[-1]) - prev[1]
    gold_flag = scoring.gold_equity_flag(cfg, spx_ret, gold_ret, stress["score"], features.get("vix"), hy_chg)

    # --- 轉折確認 ---
    totals = [h["total"] for h in hist if h["total"] is not None]
    thr = tp["precondition_stress_min"]
    if any(t >= thr for t in totals):
        had_high: Optional[bool] = True
    elif len(totals) >= 30:  # 至少有一半以上的天數有分數，才能斷定「沒有出現過高壓力」
        had_high = False
    else:
        had_high = None

    # 條件 1：Fed 資產負債表 4 週內擴張，或聯邦基金利率 4 週內降幅 >= 50bp
    c1_parts: list[Optional[bool]] = []
    walcl = S.get("walcl")
    if walcl is not None and is_fresh(walcl, "walcl", cfg, today) and len(walcl) >= 5:
        c1_parts.append(bool(walcl.iloc[-1] > walcl.iloc[-1 - tp["fed_balance_sheet_weeks"]]))
    dff = S.get("dff")
    if dff is not None and is_fresh(dff, "dff", cfg, today):
        prev = asof_value(dff, dff.index[-1] - pd.Timedelta(days=7 * tp["fed_balance_sheet_weeks"]))
        if prev is not None:
            c1_parts.append(bool((dff.iloc[-1] - prev[1]) * 100.0 <= -tp["emergency_cut_bp"]))
    c1: Optional[bool] = None if not c1_parts else any(c1_parts)

    # 條件 2：壓力指標區塊分數自 60 日高點回落 >= 15 分
    blocks = [h["block"] for h in hist if h["block"] is not None]
    now_block = stress["blocks"][0]["score"] if stress["blocks"][0].get("included") else None
    c2: Optional[bool] = None
    if now_block is not None and len(blocks) >= 10:
        c2 = bool(now_block <= max(blocks) - tp["stress_block_drop_points"])

    # 條件 3：VIX 自 20 日峰值回落 >= 30%
    vix = S.get("vix")
    c3: Optional[bool] = None
    if vix is not None and is_fresh(vix, "vix", cfg, today) and len(vix) >= 20:
        peak = vix.iloc[-20:].max()
        c3 = bool(vix.iloc[-1] <= peak * (1 - tp["vix_drop_from_peak_pct"] / 100.0))

    # 條件 4：S&P 500 的 60 日最低收盤價發生在 20 個交易日以前
    spx = S.get("spx")
    c4: Optional[bool] = None
    if spx is not None and is_fresh(spx, "spx", cfg, today) and len(spx) >= tp["lookback_trading_days"]:
        w = spx.iloc[-tp["lookback_trading_days"]:].to_numpy()
        days_since_low = len(w) - 1 - int(np.argmin(w))
        c4 = bool(days_since_low >= tp["no_new_low_days"])

    turning = scoring.turning_point_flag(cfg, had_high, [c1, c2, c3, c4])
    flags = {"gold_equity_drop": gold_flag, "turning_point": turning}
    triggered = (
        spx_ret is not None and gold_ret is not None
        and spx_ret <= fg["spx_return_max_pct"] and gold_ret <= fg["gold_return_max_pct"]
    )
    details = {"gold_triggered": bool(triggered), "had_high": had_high, "conditions": [c1, c2, c3, c4]}
    return flags, details


def compute_flags(S, cfg, today, features, stress, hist) -> dict[str, Any]:
    return compute_flags_detailed(S, cfg, today, features, stress, hist)[0]


# ---------------------------------------------------------------------------
# 總流程
# ---------------------------------------------------------------------------


def build_detailed(S: dict[str, Optional[pd.Series]], cfg: dict[str, Any], today: date, generated_at: str):
    """回傳 (公開輸出, 內部細節)。內部細節含原始特徵值，只供回測診斷，絕不可寫入檔案或印出。"""
    features, as_of = compute_features(S, cfg, today)
    regime = scoring.score_regime(cfg, features)
    stress = scoring.score_stress(cfg, features)
    hist = stress_history(S, cfg, today)
    flags, flag_details = compute_flags_detailed(S, cfg, today, features, stress, hist)
    public = scoring.to_public(regime, stress, flags, generated_at, as_of)
    details = {"features": features, "stress_score": stress["score"], **flag_details}
    return public, details


def build(S: dict[str, Optional[pd.Series]], cfg: dict[str, Any], today: date, generated_at: str) -> dict[str, Any]:
    """回傳公開輸出（可直接寫成 data.json）。"""
    return build_detailed(S, cfg, today, generated_at)[0]
