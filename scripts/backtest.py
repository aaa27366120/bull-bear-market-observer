"""歷史回測：逐日重播 2008、2020、2022 與一段平靜期，檢查分數與旗標的表現（對應 spec 第 10 節）。

執行：FRED_API_KEY=... python scripts/backtest.py
輸出：backtest/report.md（摘要）、backtest/daily.csv（逐日的分數與狀態）

重要限制（報告裡也會寫出來）：
- 這是「用目前版本的歷史資料」重播。NFCI 與 STLFSI4 的歷史值會被修正，所以結果帶有後見之明；
  要貼近當時真實情況需使用 ALFRED 的歷史版本，本程式沒有做。
- HY OAS 在 FRED 只有近 3 年，歷史期間 B 只用 NFCI、STLFSI4、VIX，其餘依缺值規則重分配權重。
- 已套用「公布落後」：週資料與月資料要等公布後才能使用，避免偷看未來。
- 回測只用來校準初始設定，不代表未來，也不構成投資建議。
- 輸出只含計算後的分數、狀態與日期，不含第三方原始數值。
"""

from __future__ import annotations

import csv
import os
import sys
from typing import Any, Optional

import pandas as pd
import yaml

import fetch_and_score as fs
import pipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "backtest")
HISTORY_ROWS = 700  # 每個時間點只取最近這麼多筆，避免重複運算整段歷史

# 各序列「觀測日期到實際能取得」的落後天數（日曆天），用來避免回測偷看未來
PUBLICATION_LAG_DAYS = {
    "hy_oas": 1, "vix": 1, "vix3m": 1, "dff": 1, "dollar": 1, "t10y3m": 1, "walcl": 1,
    "nfci": 5, "stlfsi4": 5, "icsa": 5,
    "sahm": 35,  # 觀測日期為當月 1 日，約隔月第一個週五公布
    "spx": 0, "twii": 0, "gold": 0,
}

# peak / trough 是我依記憶填的參考日期，報告會用資料算出的跌幅供你對照，請自行核對
EPISODES = [
    {"id": "2008", "name": "2008 金融海嘯", "start": "2007-10-01", "end": "2009-12-31",
     "peak": "2007-10-09", "trough": "2009-03-09", "events": {"雷曼兄弟破產": "2008-09-15"}},
    {"id": "2020", "name": "2020 年 3 月（疫情）", "start": "2019-10-01", "end": "2020-12-31",
     "peak": "2020-02-19", "trough": "2020-03-23", "events": {}},
    {"id": "2022", "name": "2022 升息與美元走強", "start": "2021-10-01", "end": "2023-03-31",
     "peak": "2022-01-03", "trough": "2022-10-12", "events": {}},
    {"id": "2017", "name": "2017 平靜期（對照組）", "start": "2016-10-01", "end": "2017-12-31",
     "peak": None, "trough": None, "events": {}},
]


# ---------------------------------------------------------------------------
# 重播
# ---------------------------------------------------------------------------


def apply_publication_lag(S: dict[str, Optional[pd.Series]]) -> dict[str, Optional[pd.Series]]:
    out: dict[str, Optional[pd.Series]] = {}
    for k, v in S.items():
        if v is None:
            out[k] = None
            continue
        v = pipeline.clean(v)
        lag = PUBLICATION_LAG_DAYS.get(k, 0)
        out[k] = None if v is None else pd.Series(v.to_numpy(), index=v.index + pd.Timedelta(days=lag))
    return out


def run_episode(S: dict[str, Optional[pd.Series]], cfg: dict[str, Any], start: str, end: str) -> list[dict[str, Any]]:
    """S 必須已套用公布落後。逐一交易日重算並回傳每天的結果。"""
    spx = S["spx"]
    days = spx.loc[start:end].index
    rows = []
    for d in days:
        sl = {k: (None if v is None else v.loc[:d].iloc[-HISTORY_ROWS:]) for k, v in S.items()}
        out = pipeline.build(sl, cfg, d.date(), d.strftime("%Y-%m-%dT00:00:00Z"))
        rows.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "a_score": out["regime"]["score"],
                "a_state": out["regime"]["state"],
                "a_status": out["regime"]["status"],
                "a_completeness": out["regime"]["completeness"],
                "b_score": out["stress"]["score"],
                "b_label": (out["stress"]["light"] or {}).get("label"),
                "b_status": out["stress"]["status"],
                "b_completeness": out["stress"]["completeness"],
                "gold_equity": out["flags"]["gold_equity_drop"]["state"],
                "turning_point": out["flags"]["turning_point"]["state"],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 摘要
# ---------------------------------------------------------------------------


def _first(df: pd.DataFrame, mask: pd.Series) -> Optional[str]:
    hit = df.loc[mask, "date"]
    return None if hit.empty else str(hit.iloc[0])


def _ranges(df: pd.DataFrame, mask: pd.Series, limit: int = 8) -> list[str]:
    """把連續符合條件的交易日壓縮成「起～迄」區間。"""
    out, start, prev = [], None, None
    for date, flag in zip(df["date"], mask):
        if flag and start is None:
            start = date
        if not flag and start is not None:
            out.append((start, prev))
            start = None
        prev = date
    if start is not None:
        out.append((start, prev))
    text = [a if a == b else f"{a}～{b}" for a, b in out]
    return text[:limit] + ([f"…（共 {len(text)} 段）"] if len(text) > limit else [])


def _day_gap(a: Optional[str], b: Optional[str]) -> Optional[int]:
    if not a or not b:
        return None
    return (pd.Timestamp(a) - pd.Timestamp(b)).days


def summarize(rows: list[dict[str, Any]], ep: dict[str, Any], spx: pd.Series) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    n = len(df)
    s: dict[str, Any] = {"id": ep["id"], "name": ep["name"], "days": n}
    s["a_ok_pct"] = round(100 * (df["a_status"] == "ok").mean(), 1)
    s["b_ok_pct"] = round(100 * (df["b_status"] == "ok").mean(), 1)
    s["b_insufficient_pct"] = round(100 * (df["b_status"] == "insufficient").mean(), 1)
    s["a_states"] = df["a_state"].fillna("資料不足").value_counts().to_dict()

    bear = df["a_state"].isin(["偏空", "熊市"])
    s["a_first_bearish"] = _first(df, bear)
    s["a_first_bear_market"] = _first(df, df["a_state"] == "熊市")
    if df["a_score"].notna().any():
        i = df["a_score"].idxmin()
        s["a_min"] = (round(float(df.loc[i, "a_score"]), 1), df.loc[i, "date"])

    if df["b_score"].notna().any():
        i = df["b_score"].idxmax()
        s["b_max"] = (round(float(df.loc[i, "b_score"]), 1), df.loc[i, "date"])
    s["b_first_60"] = _first(df, df["b_score"] >= 60)
    s["b_first_80"] = _first(df, df["b_score"] >= 80)
    s["b_days_60"] = int((df["b_score"] >= 60).sum())

    s["gold_red_days"] = int((df["gold_equity"] == "red").sum())
    s["gold_yellow_days"] = int((df["gold_equity"] == "yellow").sum())
    s["gold_first_red"] = _first(df, df["gold_equity"] == "red")
    s["gold_red_ranges"] = _ranges(df, df["gold_equity"] == "red")
    s["gold_yellow_ranges"] = _ranges(df, df["gold_equity"] == "yellow")

    s["tp_first_met"] = _first(df, df["turning_point"] == "met")
    s["tp_met_days"] = int((df["turning_point"] == "met").sum())
    s["tp_states"] = df["turning_point"].value_counts().to_dict()

    # 與參考日期對照
    for key in ("peak", "trough"):
        ref = ep.get(key)
        if ref:
            hit = spx.loc[ref:].index
            s[key] = hit[0].strftime("%Y-%m-%d") if len(hit) else None
    if s.get("peak") and s.get("trough"):
        p, t = float(spx.loc[s["peak"]]), float(spx.loc[s["trough"]])
        s["max_drawdown_pct"] = round((1 - t / p) * 100, 1)
        s["a_bearish_after_peak_days"] = _day_gap(s["a_first_bearish"], s["peak"])
        s["b60_vs_trough_days"] = _day_gap(s["b_first_60"], s["trough"])
        s["tp_vs_trough_days"] = _day_gap(s["tp_first_met"], s["trough"])
        tr = df.loc[df["date"] == s["trough"]]
        if not tr.empty and pd.notna(tr["b_score"].iloc[0]):
            s["b_at_trough"] = round(float(tr["b_score"].iloc[0]), 1)
        if not tr.empty and pd.notna(tr["a_score"].iloc[0]):
            s["a_at_trough"] = round(float(tr["a_score"].iloc[0]), 1)
    return s


# ---------------------------------------------------------------------------
# 報告
# ---------------------------------------------------------------------------


def _fmt(v: Any, suffix: str = "") -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{v}{suffix}"


def _signed_days(v: Optional[int]) -> str:
    if v is None:
        return "—"
    return f"{v:+d} 天"


def render_report(results: list[dict[str, Any]], generated: str) -> str:
    L: list[str] = []
    L.append("# 歷史回測報告\n")
    L.append(f"產生時間：{generated}\n")
    L.append("> 這份報告只用來校準初始設定，不代表未來，也不構成投資建議。所有日期與天數都是用資料算出的結果，")
    L.append("> 參考的高低點日期是依記憶填寫，請自行核對。\n")
    L.append("## 方法與限制\n")
    L.append("- 逐一交易日重播，每天只使用「當時已公布」的資料（週資料落後 5 天、Sahm Rule 落後 35 天）。")
    L.append("- **後見之明：** NFCI 與 STLFSI4 的歷史值會被修正，這裡用的是目前版本，結果會比當時真實情況樂觀。")
    L.append("- **HY OAS 不可用：** FRED 只剩近 3 年資料，所以歷史期間 B 只用 NFCI、STLFSI4、VIX，其餘依缺值規則重分配。")
    L.append("- 輸出只含分數、狀態與日期，不含第三方原始數值。\n")

    L.append("## 總覽\n")
    L.append("| 期間 | 交易日 | A 完整 | B 完整 | S&P 500 最大回檔 | A 首次偏空 | B 首次 ≥60 | 轉折確認首次成立 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        L.append(
            f"| {r['name']} | {r['days']} | {r['a_ok_pct']}% | {r['b_ok_pct']}% | "
            f"{_fmt(r.get('max_drawdown_pct'), '%')} | {_fmt(r['a_first_bearish'])} | "
            f"{_fmt(r['b_first_60'])} | {_fmt(r['tp_first_met'])} |"
        )
    L.append("")

    for r in results:
        L.append(f"## {r['name']}\n")
        if r.get("peak"):
            L.append(f"參考高點 {r['peak']}，參考低點 {r['trough']}，S&P 500 區間最大回檔 {_fmt(r.get('max_drawdown_pct'), '%')}。\n")
        L.append("**A 市場環境**\n")
        L.append("- 狀態分布（天數）：" + "、".join(f"{k} {v}" for k, v in r["a_states"].items()))
        L.append(f"- 首次偏空：{_fmt(r['a_first_bearish'])}"
                 + (f"（高點後 {_signed_days(r.get('a_bearish_after_peak_days'))}）" if r.get("peak") else ""))
        L.append(f"- 首次熊市：{_fmt(r['a_first_bear_market'])}")
        if r.get("a_min"):
            L.append(f"- 最低分：{r['a_min'][0]}（{r['a_min'][1]}）")
        if r.get("a_at_trough") is not None:
            L.append(f"- 低點當天 A 分數：{r['a_at_trough']}")
        L.append("")
        L.append("**B 金融壓力**\n")
        if r.get("b_max"):
            L.append(f"- 最高分：{r['b_max'][0]}（{r['b_max'][1]}）")
        L.append(f"- 首次 ≥60（高壓力）：{_fmt(r['b_first_60'])}"
                 + (f"（相對低點 {_signed_days(r.get('b60_vs_trough_days'))}）" if r.get("trough") else ""))
        L.append(f"- 首次 ≥80（危機級）：{_fmt(r['b_first_80'])}；≥60 的天數：{r['b_days_60']}")
        if r.get("b_at_trough") is not None:
            L.append(f"- 低點當天 B 分數：{r['b_at_trough']}")
        L.append(f"- 資料不足的天數比例：{r['b_insufficient_pct']}%")
        L.append("")
        L.append("**警示旗標**\n")
        L.append(f"- 股金同跌：紅色 {r['gold_red_days']} 天、黃色 {r['gold_yellow_days']} 天")
        if r["gold_red_ranges"]:
            L.append("  - 紅色區間：" + "；".join(r["gold_red_ranges"]))
        if r["gold_yellow_ranges"]:
            L.append("  - 黃色區間：" + "；".join(r["gold_yellow_ranges"]))
        L.append(f"- 轉折確認：首次成立 {_fmt(r['tp_first_met'])}"
                 + (f"（相對低點 {_signed_days(r.get('tp_vs_trough_days'))}）" if r.get("trough") else "")
                 + f"，成立天數 {r['tp_met_days']}")
        L.append("- 轉折確認狀態分布：" + "、".join(f"{k} {v}" for k, v in r["tp_states"].items()))
        L.append("")

    L.append("## 怎麼讀這份報告\n")
    L.append("1. **2008、2020：** B 應該在低點前明顯升高；A 應該轉為偏空；股金同跌若發生在危機中應為紅色。")
    L.append("2. **2022：** 這是升息造成的下跌，不是流動性危機。股金同跌理想上以黃色為主，B 不應長期處於危機級。")
    L.append("3. **2017 對照組：** 平靜期不應出現紅色旗標，B 應長期偏低。出現就是誤報，需要檢討門檻。")
    L.append("4. **轉折確認：** 相對低點的天數為負代表在低點之前就成立（太早），為正代表之後才成立（較保守）。")
    L.append("5. 以上是用修正後的歷史值重播的結果，真實使用時的表現通常不會更好。")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def load_history(api_key: str) -> dict[str, Optional[pd.Series]]:
    S: dict[str, Optional[pd.Series]] = {}
    for key, fred_id in fs.FRED_SERIES.items():
        s = fs.fetch_fred(fred_id, api_key, "1990-01-01")
        S[key] = s
        if s is not None:
            fs.log(f"  FRED {fred_id}: 成功，資料 {s.index[0].strftime('%Y-%m-%d')} ～ {s.index[-1].strftime('%Y-%m-%d')}")
    for key, symbols in fs.YAHOO_SERIES.items():
        s, used = fs.fetch_yahoo(symbols, period="max")
        S[key] = s
        if s is not None:
            fs.log(f"  Yahoo {used}: 成功，資料 {s.index[0].strftime('%Y-%m-%d')} ～ {s.index[-1].strftime('%Y-%m-%d')}")
        else:
            fs.log(f"  Yahoo {'/'.join(symbols)}: 無資料")
    return S


def write_outputs(all_rows: list[dict[str, Any]], report: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    cols = list(all_rows[0].keys())
    with open(os.path.join(OUT_DIR, "daily.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)


def main() -> int:
    api_key = os.environ.get("FRED_API_KEY", "").strip()
    if not api_key:
        fs.log("錯誤：沒有設定環境變數 FRED_API_KEY。")
        return 1
    with open(fs.CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    fs.log("下載歷史資料…")
    raw = load_history(api_key)
    if raw.get("spx") is None:
        fs.log("S&P 500 歷史資料取得失敗，無法回測。")
        return 1
    S = apply_publication_lag(raw)

    results, all_rows = [], []
    for ep in EPISODES:
        fs.log(f"重播：{ep['name']}（{ep['start']} ～ {ep['end']}）")
        rows = run_episode(S, cfg, ep["start"], ep["end"])
        if not rows:
            fs.log("  這段期間沒有資料，略過。")
            continue
        results.append(summarize(rows, ep, S["spx"]))
        all_rows.extend({"episode": ep["id"], **r} for r in rows)

    if not results:
        fs.log("沒有任何期間可以回測。")
        return 1
    generated = pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC")
    write_outputs(all_rows, render_report(results, generated))
    fs.log("已寫入 backtest/report.md 與 backtest/daily.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
