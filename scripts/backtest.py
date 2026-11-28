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


def run_episode_detailed(S: dict[str, Optional[pd.Series]], cfg: dict[str, Any], start: str, end: str):
    """S 必須已套用公布落後。逐一交易日重算，回傳 (每天的公開結果, 每天的內部細節)。

    內部細節含第三方資料衍生的數值（如 VIX），只用於診斷與敏感度分析，不會寫入任何輸出檔。
    """
    spx = S["spx"]
    days = spx.loc[start:end].index
    red_cfg = cfg["flags"]["gold_equity_drop"]["red_if_any"]
    rows, internals = [], []
    for d in days:
        sl = {k: (None if v is None else v.loc[:d].iloc[-HISTORY_ROWS:]) for k, v in S.items()}
        out, det = pipeline.build_detailed(sl, cfg, d.date(), d.strftime("%Y-%m-%dT00:00:00Z"))
        gold_state = out["flags"]["gold_equity_drop"]["state"]
        b, vix = det["stress_score"], det["features"].get("vix")
        reason = ""
        if gold_state == "red":
            parts = []
            if b is not None and b >= red_cfg["stress_score_min"]:
                parts.append("B≥%d" % red_cfg["stress_score_min"])
            if vix is not None and vix > red_cfg["vix_min"]:
                parts.append("VIX>%d" % red_cfg["vix_min"])
            reason = "+".join(parts) or "HY 利差"
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
                "gold_equity": gold_state,
                "red_reason": reason,
                "turning_point": out["flags"]["turning_point"]["state"],
            }
        )
        internals.append(
            {"triggered": det["gold_triggered"], "b_score": b, "vix": vix,
             "had_high": det["had_high"], "conditions": det["conditions"]}
        )
    return rows, internals


def run_episode(S: dict[str, Optional[pd.Series]], cfg: dict[str, Any], start: str, end: str) -> list[dict[str, Any]]:
    return run_episode_detailed(S, cfg, start, end)[0]


# ---------------------------------------------------------------------------
# 敏感度：股金同跌的紅色條件換成不同的 VIX 門檻，三次下跌各會有幾天紅色
# ---------------------------------------------------------------------------

# (名稱, B 門檻, VIX 門檻；None 代表不使用 VIX 條件)。歷史上沒有 HY OAS，所以這裡不含利差條件。
VARIANTS = [
    ("基準：B≥60 或 VIX>30", 60, 30),
    ("B≥60 或 VIX>35", 60, 35),
    ("B≥60 或 VIX>40", 60, 40),
    ("只看 B≥60（不用 VIX）", 60, None),
]


def classify_gold_flag(it: dict[str, Any], b_thr: float, vix_thr: Optional[float]) -> str:
    if not it["triggered"]:
        return "none"
    b_hit = it["b_score"] is not None and it["b_score"] >= b_thr
    v_hit = vix_thr is not None and it["vix"] is not None and it["vix"] > vix_thr
    return "red" if (b_hit or v_hit) else "yellow"


def sensitivity(internals_by_ep: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    table = {}
    for name, b_thr, vix_thr in VARIANTS:
        table[name] = {}
        for ep_id, its in internals_by_ep.items():
            states = [classify_gold_flag(it, b_thr, vix_thr) for it in its]
            table[name][ep_id] = (states.count("red"), states.count("yellow"))
    return {"variants": [v[0] for v in VARIANTS], "table": table}


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


def summarize(rows: list[dict[str, Any]], ep: dict[str, Any], spx: pd.Series,
              internals: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    n = len(df)
    s: dict[str, Any] = {"id": ep["id"], "name": ep["name"], "days": n}
    s["a_ok_pct"] = round(100 * (df["a_status"] == "ok").mean(), 1)
    s["b_ok_pct"] = round(100 * (df["b_status"] == "ok").mean(), 1)
    s["b_insufficient_pct"] = round(100 * (df["b_status"] == "insufficient").mean(), 1)
    s["b_ref_pct"] = round(100 * (df["b_status"] == "reference_only").mean(), 1)
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
    s["tp_met_ranges"] = _ranges(df, df["turning_point"] == "met")
    s["red_reasons"] = df.loc[df["gold_equity"] == "red", "red_reason"].value_counts().to_dict()

    # 轉折確認的條件診斷（需要內部細節）
    if internals:
        evaluated = [it for it in internals if it["had_high"]]
        met_idx = [i for i, st in enumerate(df["turning_point"]) if st == "met"]

        def rate(items, k):
            known = [it["conditions"][k] for it in items if it["conditions"][k] is not None]
            return None if not known else round(100 * sum(known) / len(known), 1)

        s["cond_rate_evaluated"] = [rate(evaluated, k) for k in range(4)]
        s["cond_rate_met"] = [rate([internals[i] for i in met_idx], k) for k in range(4)]
        s["tp_first_met_conditions"] = internals[met_idx[0]]["conditions"] if met_idx else None

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


COND_NAMES = [
    "Fed 資產負債表擴張，或降息 ≥50bp",
    "壓力指標區塊自高點回落 ≥15 分",
    "VIX 自 20 日峰值回落 ≥30%",
    "S&P 500 的 60 日低點在 20 日以前",
]


def render_report(results: list[dict[str, Any]], generated: str, sens: Optional[dict[str, Any]] = None) -> str:
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
    L.append("| 期間 | 交易日 | A 狀態正常 | B 正常／僅供參考 | S&P 500 最大回檔 | A 首次偏空 | B 首次 ≥60 | 轉折確認首次成立 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        L.append(
            f"| {r['name']} | {r['days']} | {r['a_ok_pct']}% | {r['b_ok_pct']}%／{r['b_ref_pct']}% | "
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
        if r["tp_met_ranges"]:
            L.append("- 轉折確認成立區間：" + "；".join(r["tp_met_ranges"]))
        if r["red_reasons"]:
            L.append("- 股金同跌紅色的成因（天數）：" + "、".join(f"{k} {v}" for k, v in r["red_reasons"].items()))
        if r.get("cond_rate_evaluated"):
            L.append("")
            L.append("**轉折確認的四項條件**（只統計「已評估」的日子，即近 60 日內 B 曾 ≥60）\n")
            L.append("| 條件 | 已評估日子中成立比例 | 轉折確認成立日中成立比例 |")
            L.append("|---|---|---|")
            for k, name in enumerate(COND_NAMES):
                L.append(f"| {name} | {_fmt(r['cond_rate_evaluated'][k], '%')} | {_fmt(r['cond_rate_met'][k], '%')} |")
            if r.get("tp_first_met_conditions"):
                flags = r["tp_first_met_conditions"]
                text = "、".join(f"{COND_NAMES[k]}：{'是' if v else ('否' if v is False else '未知')}" for k, v in enumerate(flags))
                L.append("")
                L.append("- 首次成立當天各條件：" + text)
        L.append("")

    if sens:
        L.append("## 敏感度：股金同跌的紅色條件\n")
        L.append("把紅色的確認條件換成不同的 VIX 門檻，各期間會有幾天紅色／黃色（格式：紅／黃）。歷史上沒有 HY OAS，所以不含利差條件。\n")
        ids = [r["id"] for r in results]
        names = {r["id"]: r["name"] for r in results}
        L.append("| 規則 | " + " | ".join(names[i] for i in ids) + " |")
        L.append("|---|" + "---|" * len(ids))
        for v in sens["variants"]:
            cells = []
            for i in ids:
                red, yellow = sens["table"][v].get(i, (0, 0))
                cells.append(f"{red}／{yellow}")
            L.append(f"| {v} | " + " | ".join(cells) + " |")
        L.append("")
        L.append("> 用這三段歷史調整門檻等於「用答案調參數」，結果只能當參考，不能當作門檻已經最佳化的證明。\n")

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

    results, all_rows, internals_by_ep = [], [], {}
    for ep in EPISODES:
        fs.log(f"重播：{ep['name']}（{ep['start']} ～ {ep['end']}）")
        rows, internals = run_episode_detailed(S, cfg, ep["start"], ep["end"])
        if not rows:
            fs.log("  這段期間沒有資料，略過。")
            continue
        results.append(summarize(rows, ep, S["spx"], internals))
        internals_by_ep[ep["id"]] = internals
        all_rows.extend({"episode": ep["id"], **r} for r in rows)

    if not results:
        fs.log("沒有任何期間可以回測。")
        return 1
    generated = pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC")
    write_outputs(all_rows, render_report(results, generated, sensitivity(internals_by_ep)))
    fs.log("已寫入 backtest/report.md 與 backtest/daily.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
