"""牛市熊市觀察儀表板：計分引擎（對應 spec.md v0.3）。

只做純計算，不碰網路與檔案，方便單元測試。
輸入是「已算好的特徵值」（例如 S&P 500 相對 200 日均線的乖離百分比），
缺值一律用 None 表示，絕不以 0 代入。
"""

from __future__ import annotations

from typing import Any, Optional

# ---------------------------------------------------------------------------
# 基本工具
# ---------------------------------------------------------------------------


def interpolate(value: Optional[float], anchors: list[list[float]]) -> Optional[float]:
    """依錨點線性內插成 0～100 分；超出範圍取端點值；value 為 None 回傳 None。"""
    if value is None:
        return None
    pts = sorted((float(x), float(y)) for x, y in anchors)
    xs = [p[0] for p in pts]
    if len(set(xs)) != len(xs):
        raise ValueError("錨點的原始值不可重複")
    if value <= pts[0][0]:
        return pts[0][1]
    if value >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= value <= x1:
            ratio = (value - x0) / (x1 - x0)
            return y0 + ratio * (y1 - y0)
    raise AssertionError("不應該走到這裡")


def weighted_mean(items: list[tuple[Optional[float], float]]) -> tuple[Optional[float], float]:
    """加權平均；缺值（None）不計入並依剩餘權重重新分配。

    回傳 (平均分數或 None, 完整度)。完整度 = 有效權重 / 名目權重。
    """
    total = sum(w for _, w in items)
    valid = [(v, w) for v, w in items if v is not None]
    valid_weight = sum(w for _, w in valid)
    if total <= 0 or valid_weight <= 0:
        return None, 0.0
    mean = sum(v * w for v, w in valid) / valid_weight
    return mean, valid_weight / total


def light_index(score: float) -> int:
    """每 20 分一格；剛好落在邊界時歸入較高的一段；100 分歸入最後一段。"""
    return max(0, min(int(score // 20), 4))


def make_light(score: Optional[float], palette: list[dict[str, str]]) -> Optional[dict[str, str]]:
    if score is None:
        return None
    entry = palette[light_index(score)]
    return {"color": entry["color"], "label": entry["label"]}


def completeness_status(completeness: float, cfg: dict[str, float]) -> str:
    """ok / reference_only / insufficient。"""
    if completeness >= cfg["ok"]:
        return "ok"
    if completeness >= cfg["reference_only"]:
        return "reference_only"
    return "insufficient"


# ---------------------------------------------------------------------------
# 構面與總分
# ---------------------------------------------------------------------------


def score_group(group: dict[str, Any], features: dict[str, Optional[float]]) -> dict[str, Any]:
    """計算一個構面（或區塊）：每個指標內插、構面內加權平均。"""
    indicators = []
    items = []
    for ind in group["indicators"]:
        raw = features.get(ind["id"])
        score = interpolate(raw, ind["anchors"])
        items.append((score, ind["weight"]))
        indicators.append({"id": ind["id"], "name": ind["name"], "score": score})
    mean, comp = weighted_mean(items)
    return {
        "id": group["id"],
        "name": group["name"],
        "weight": group["weight"],
        "score": mean,
        "completeness": comp,
        "indicators": indicators,
    }


def combine_groups(groups: list[dict[str, Any]], completeness_cfg: dict[str, float]) -> dict[str, Any]:
    """把多個構面合成總體 0～100 分。

    構面完整度低於 dimension_min 時，該構面不計入總分（視為缺值）。
    總體完整度 = 有效構面的（構面權重 x 構面完整度）總和 / 構面權重總和。
    """
    items = []
    valid_weight = 0.0
    total_weight = sum(g["weight"] for g in groups)
    for g in groups:
        usable = g["score"] is not None and g["completeness"] >= completeness_cfg["dimension_min"]
        g["included"] = usable
        items.append((g["score"] if usable else None, g["weight"]))
        if usable:
            valid_weight += g["weight"] * g["completeness"]
    mean, _ = weighted_mean(items)
    overall_comp = valid_weight / total_weight if total_weight > 0 else 0.0
    return {
        "score_0_100": mean,
        "completeness": overall_comp,
        "status": completeness_status(overall_comp, completeness_cfg),
    }


# ---------------------------------------------------------------------------
# 分數 A：市場環境
# ---------------------------------------------------------------------------

_REGIME_ORDER = ["熊市", "偏空", "盤整", "偏多", "牛市"]


def regime_state(total: float) -> str:
    """-100～+100 總分對應狀態。邊界：> +40 牛市；(+10, +40] 偏多；[-10, +10] 盤整；[-40, -10) 偏空；< -40 熊市。"""
    if total > 40:
        return "牛市"
    if total > 10:
        return "偏多"
    if total >= -10:
        return "盤整"
    if total >= -40:
        return "偏空"
    return "熊市"


def apply_drawdown_cap(state: str, drawdown_pct: Optional[float], cap_pct: float) -> tuple[str, bool]:
    """S&P 500 回檔超過門檻時，狀態最多只能是「盤整」。只使用 A 自己的輸入。"""
    if drawdown_pct is not None and drawdown_pct > cap_pct:
        if _REGIME_ORDER.index(state) > _REGIME_ORDER.index("盤整"):
            return "盤整", True
    return state, False


def score_regime(cfg: dict[str, Any], features: dict[str, Optional[float]]) -> dict[str, Any]:
    groups = [score_group(d, features) for d in cfg["regime"]["dimensions"]]
    combined = combine_groups(groups, cfg["completeness"])
    palette = cfg["lights"]["regime"]

    base = combined["score_0_100"]
    result: dict[str, Any] = {
        "dimensions": groups,
        "completeness": combined["completeness"],
        "status": combined["status"],
        "score": None,
        "state": None,
        "capped_by_drawdown": False,
        "light": None,
    }
    if base is None or combined["status"] == "insufficient":
        result["status"] = "insufficient"
        return result

    total = 2 * (base - 50)
    state = regime_state(total)
    state, capped = apply_drawdown_cap(
        state, features.get("spx_drawdown_52w"), cfg["regime"]["state_overlay"]["drawdown_cap_pct"]
    )
    result.update(
        {
            "score": total,
            "state": state,
            "capped_by_drawdown": capped,
            # 燈號用 0～100 換算（(總分 + 100) / 2），與構面燈號同一套五段
            "light": make_light((total + 100) / 2, palette),
        }
    )
    for g in groups:
        g["light"] = make_light(g["score"], palette)
        for ind in g["indicators"]:
            ind["light"] = make_light(ind["score"], palette)
    return result


# ---------------------------------------------------------------------------
# 分數 B：金融壓力
# ---------------------------------------------------------------------------


def score_stress(cfg: dict[str, Any], features: dict[str, Optional[float]]) -> dict[str, Any]:
    groups = [score_group(b, features) for b in cfg["stress"]["blocks"]]
    combined = combine_groups(groups, cfg["completeness"])
    palette = cfg["lights"]["stress"]

    result: dict[str, Any] = {
        "blocks": groups,
        "completeness": combined["completeness"],
        "status": combined["status"],
        "score": None,
        "light": None,
    }
    for g in groups:
        g["light"] = make_light(g["score"], palette)
        for ind in g["indicators"]:
            ind["light"] = make_light(ind["score"], palette)
    if combined["score_0_100"] is None or combined["status"] == "insufficient":
        result["status"] = "insufficient"
        return result
    result["score"] = combined["score_0_100"]
    result["light"] = make_light(result["score"], palette)
    return result


# ---------------------------------------------------------------------------
# 警示旗標
# ---------------------------------------------------------------------------


def gold_equity_flag(
    cfg: dict[str, Any],
    spx_return_pct: Optional[float],
    gold_return_pct: Optional[float],
    stress_score: Optional[float],
    vix: Optional[float],
    hy_oas_week_change_pp: Optional[float],
) -> dict[str, Any]:
    """股金同跌：none / yellow / red / unknown。

    觸發需 S&P 500 與黃金近 10 個交易日報酬都在門檻以下。
    紅：觸發且符合任一確認條件；黃：觸發但確認條件皆未成立。
    """
    c = cfg["flags"]["gold_equity_drop"]
    if spx_return_pct is None or gold_return_pct is None:
        return {"state": "unknown"}
    triggered = spx_return_pct <= c["spx_return_max_pct"] and gold_return_pct <= c["gold_return_max_pct"]
    if not triggered:
        return {"state": "none"}
    r = c["red_if_any"]
    checks = [
        None if stress_score is None else stress_score >= r["stress_score_min"],
        None if vix is None else vix > r["vix_min"],
        None if hy_oas_week_change_pp is None else hy_oas_week_change_pp >= r["hy_oas_week_change_min_pp"],
    ]
    if any(ch is True for ch in checks):
        return {"state": "red"}
    return {"state": "yellow", "confirmation_unknown": all(ch is None for ch in checks)}


def turning_point_flag(
    cfg: dict[str, Any],
    had_high_stress: Optional[bool],
    conditions: list[Optional[bool]],
) -> dict[str, Any]:
    """轉折確認。

    前提：過去 60 個交易日內 B 分數曾達門檻，否則 not_evaluated。
    4 項條件（True/False/None=未知）至少 3 項成立才是 met；
    有效（非 None）項目少於 3 項時回報 insufficient。
    """
    required = cfg["flags"]["turning_point"]["required_conditions"]
    if had_high_stress is None:
        return {"state": "insufficient"}
    if not had_high_stress:
        return {"state": "not_evaluated"}
    known = [c for c in conditions if c is not None]
    if len(known) < required:
        return {"state": "insufficient", "met_count": sum(known)}
    met = sum(1 for c in known if c)
    return {"state": "met" if met >= required else "not_met", "met_count": met}


# ---------------------------------------------------------------------------
# 公開輸出（D4、D10）：只放分數與燈號，指標層級只放燈號，不放原始值與精確指標分數
# ---------------------------------------------------------------------------


def _public_indicator(ind: dict[str, Any]) -> dict[str, Any]:
    return {"id": ind["id"], "name": ind["name"], "light": ind.get("light")}


def _public_group(g: dict[str, Any], publish_score: bool = True) -> dict[str, Any]:
    """構面／區塊的公開輸出。

    公開分數不得讓任何單一指標可以被反推出來（錨點在公開倉庫裡）：
    1. 只有一個指標的構面，其分數就是該指標的分數，不公開分數，只公開燈號。
    2. 若總分是由幾個區塊加權而成，而其中一個區塊只含單一指標，
       公開其餘區塊的分數與總分，兩者相減就能算出該指標的分數。
       所以這種情況下所有區塊都只公開燈號（publish_score=False）。
    """
    single = len(g["indicators"]) == 1
    score = g["score"]
    hide = (not publish_score) or single or score is None
    return {
        "id": g["id"],
        "name": g["name"],
        "weight": g["weight"],
        "included": g.get("included", False),
        "score": None if hide else round(score, 1),
        "light": g.get("light"),
        "completeness": round(g["completeness"], 1),
        "indicators": [_public_indicator(i) for i in g["indicators"]],
    }


def to_public(
    regime: dict[str, Any],
    stress: dict[str, Any],
    flags: dict[str, Any],
    generated_at: str,
    data_dates: dict[str, Optional[str]],
) -> dict[str, Any]:
    def r(x: Optional[float]) -> Optional[float]:
        return None if x is None else round(x, 1)

    return {
        "schema_version": 1,
        "generated_at": generated_at,
        "data_dates": data_dates,
        "regime": {
            "status": regime["status"],
            "score": r(regime["score"]),
            "state": regime["state"],
            "light": regime["light"],
            "capped_by_drawdown": regime["capped_by_drawdown"],
            "completeness": r(regime["completeness"]),
            "dimensions": [_public_group(g) for g in regime["dimensions"]],
        },
        "stress": {
            "status": stress["status"],
            "score": r(stress["score"]),
            "light": stress["light"],
            "completeness": r(stress["completeness"]),
            "blocks": [_public_group(g, publish_score=False) for g in stress["blocks"]],
        },
        "flags": flags,
    }
