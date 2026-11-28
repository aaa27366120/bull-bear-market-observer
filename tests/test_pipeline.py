"""pipeline.py 的單元測試（全部使用模擬資料，不需要網路）。

執行：python -m unittest discover -s tests -v
"""

import json
import os
import sys
import unittest
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import pipeline  # noqa: E402

with open(os.path.join(ROOT, "config", "indicators.yml"), encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

END = pd.Timestamp("2026-11-26")
TODAY = date(2026, 11, 27)
N = 400


def idx(n=N, end=END):
    return pd.bdate_range(end=end, periods=n)


def series(values, end=END):
    values = np.asarray(values, dtype=float)
    return pd.Series(values, index=idx(len(values), end))


def make(kind="calm", end=END):
    """建立一組完整的模擬資料。calm=平靜；crisis=危機高峰（t=340）後復原到 t=399。"""
    t = np.arange(N, dtype=float)
    if kind == "calm":
        spx = 4000 * (1.0004 ** t)
        data = {
            "spx": spx, "twii": spx * 0.2, "gold": np.full(N, 2000.0),
            "hy_oas": np.full(N, 3.2137), "vix": np.full(N, 14.3719), "vix3m": np.full(N, 17.0),
            "nfci": np.full(N, -0.5), "stlfsi4": np.full(N, -0.8),
        }
    elif kind == "crisis":
        spx = np.interp(t, [0, 340, 399], [5000, 3500, 4300])
        data = {
            "spx": spx, "twii": spx * 0.2, "gold": np.full(N, 2000.0),
            "hy_oas": np.interp(t, [0, 300, 340, 399], [3.5, 3.5, 9, 5]),
            "vix": np.interp(t, [0, 300, 340, 399], [15, 15, 50, 18]),
            "vix3m": np.full(N, 30.0),
            "nfci": np.interp(t, [0, 300, 340, 399], [-0.5, -0.5, 1.0, -0.1]),
            "stlfsi4": np.interp(t, [0, 300, 340, 399], [-0.8, -0.8, 5, 0.2]),
        }
    else:
        raise ValueError(kind)
    data.update(
        {
            "walcl": 7_000_000 + t * 1000, "dff": np.full(N, 3.5), "dollar": np.full(N, 120.0),
            "t10y3m": np.full(N, 0.5), "sahm": np.full(N, 0.1), "icsa": np.full(N, 210000.0),
        }
    )
    return {k: series(v, end) for k, v in data.items()}


def truncate(S, n):
    return {k: v.iloc[:n] for k, v in S.items()}


class FeatureTests(unittest.TestCase):
    def feats(self, S):
        return pipeline.compute_features(S, CFG, TODAY)[0]

    def test_spx_linear_series(self):
        arr = np.arange(300) + 100.0
        f = self.feats({"spx": series(arr)})
        self.assertAlmostEqual(f["spx_vs_200dma"], (arr[-1] / arr[-200:].mean() - 1) * 100)
        self.assertAlmostEqual(f["spx_200dma_slope"], (arr[-200:].mean() / arr[-220:-20].mean() - 1) * 100)
        self.assertAlmostEqual(f["spx_drawdown_52w"], 0.0)  # 一路創高，回檔為 0

    def test_drawdown(self):
        arr = np.concatenate([np.full(250, 100.0), np.full(10, 80.0)])
        f = self.feats({"spx": series(arr)})
        self.assertAlmostEqual(f["spx_drawdown_52w"], 20.0)

    def test_too_short_history_gives_none(self):
        f = self.feats({"spx": series(np.arange(150) + 100.0)})
        self.assertIsNone(f["spx_vs_200dma"])
        self.assertIsNone(f["spx_drawdown_52w"])

    def test_stale_series_gives_none(self):
        f = self.feats({"spx": series(np.arange(300) + 100.0, end=pd.Timestamp("2026-10-01"))})
        self.assertIsNone(f["spx_vs_200dma"])

    def test_missing_series_gives_none(self):
        f = self.feats({"spx": None})
        self.assertTrue(all(v is None for v in f.values()))

    def test_vix_term_structure_ratio(self):
        f = self.feats({"vix": series(np.full(30, 20.0)), "vix3m": series(np.full(30, 25.0))})
        self.assertAlmostEqual(f["vix_term_structure"], 0.8)

    def test_vix_term_structure_uses_common_date_and_checks_freshness(self):
        # VIX3M 比 VIX 少 10 天，共同的最新日期已經過舊
        f = self.feats({"vix": series(np.full(30, 20.0)),
                        "vix3m": series(np.full(30, 25.0), end=END - pd.Timedelta(days=10))})
        self.assertIsNone(f["vix_term_structure"])

    def test_fed_funds_change_in_bp(self):
        days = pd.date_range("2026-01-01", "2026-11-26")
        vals = np.where(days < pd.Timestamp("2026-09-15"), 5.0, 4.75)
        f = self.feats({"dff": pd.Series(vals, index=days)})
        self.assertAlmostEqual(f["fed_funds_3m_change"], -25.0)

    def test_claims_13w_uses_4wk_average(self):
        vals = np.concatenate([np.full(16, 200000.0), np.full(4, 240000.0)])
        f = self.feats({"icsa": series(vals)})
        self.assertAlmostEqual(f["initial_claims_13w"], 20.0)

    def test_balance_sheet_13w(self):
        vals = 100 + np.arange(20.0)
        f = self.feats({"walcl": series(vals)})
        self.assertAlmostEqual(f["fed_balance_sheet_13w"], (vals[-1] / vals[-14] - 1) * 100)

    def test_monthly_sahm_allowed_up_to_75_days(self):
        ok = self.feats({"sahm": series([0.2], end=pd.Timestamp("2026-10-01"))})  # 57 天
        self.assertAlmostEqual(ok["sahm_rule"], 0.2)
        old = self.feats({"sahm": series([0.2], end=pd.Timestamp("2026-08-01"))})  # 118 天
        self.assertIsNone(old["sahm_rule"])


class ScenarioTests(unittest.TestCase):
    def build(self, S, today=TODAY):
        return pipeline.build(S, CFG, today, "2026-11-27T22:30:00Z")

    def test_calm_market(self):
        out = self.build(make("calm"))
        self.assertEqual(out["regime"]["status"], "ok")
        self.assertIn(out["regime"]["state"], ("牛市", "偏多"))
        self.assertAlmostEqual(out["regime"]["completeness"], 1.0)
        self.assertEqual(out["stress"]["status"], "ok")
        # 手算（輸入：HY 3.2137、NFCI -0.5、STLFSI4 -0.8、VIX 14.3719）：
        #   HY = 0.2137*25 = 5.3425；NFCI = (0.1/0.3)*25 = 8.3333；STLFSI4 = 0.2*40 = 8
        #   區塊 = 0.4*5.3425 + 0.3*8.3333 + 0.3*8 = 7.037
        #   VIX = (2.3719/8)*30 = 8.8946
        #   總分 = 0.7*7.037 + 0.3*8.8946 = 7.594，公開輸出四捨五入到 7.6
        self.assertAlmostEqual(out["stress"]["score"], 7.6, places=1)
        self.assertEqual(out["stress"]["light"]["label"], "低壓力")
        self.assertEqual(out["flags"]["gold_equity_drop"]["state"], "none")
        self.assertEqual(out["flags"]["turning_point"]["state"], "not_evaluated")

    def test_crisis_peak(self):
        S = make("crisis")
        n = 341  # 截到 t=340（壓力高峰）
        S = truncate(S, n)
        today = S["spx"].index[-1].date() + timedelta(days=1)
        out = self.build(S, today)
        self.assertEqual(out["stress"]["status"], "ok")
        self.assertEqual(out["stress"]["light"]["label"], "危機級")
        self.assertIn(out["regime"]["state"], ("偏空", "熊市"))
        # 高壓力剛出現，還沒有任何緩解：轉折確認不應成立
        self.assertNotEqual(out["flags"]["turning_point"]["state"], "met")

    def test_recovery_triggers_turning_point(self):
        out = self.build(make("crisis"))
        tp = out["flags"]["turning_point"]
        self.assertEqual(tp["state"], "met")
        self.assertEqual(tp["met_count"], 4)

    def test_gold_and_equity_drop_red_with_high_vix(self):
        S = make("calm")
        spx = S["spx"].copy()
        spx.iloc[-1] = spx.iloc[-11] * 0.94   # 10 個交易日 -6%
        gold = S["gold"].copy()
        gold.iloc[-1] = gold.iloc[-11] * 0.96  # -4%
        vix = S["vix"].copy()
        vix.iloc[-1] = 40.0
        S.update({"spx": spx, "gold": gold, "vix": vix})
        self.assertEqual(self.build(S)["flags"]["gold_equity_drop"]["state"], "red")

    def test_gold_and_equity_drop_yellow_without_confirmation(self):
        # 類似 2022：股金同跌，但信用與 VIX 沒有惡化
        S = make("calm")
        spx = S["spx"].copy()
        spx.iloc[-1] = spx.iloc[-11] * 0.94
        gold = S["gold"].copy()
        gold.iloc[-1] = gold.iloc[-11] * 0.96
        S.update({"spx": spx, "gold": gold})
        flag = self.build(S)["flags"]["gold_equity_drop"]
        self.assertEqual(flag["state"], "yellow")
        self.assertFalse(flag["confirmation_unknown"])

    def test_all_data_missing_does_not_crash(self):
        S = {k: None for k in pipeline.FREQ}
        out = self.build(S)
        self.assertEqual(out["regime"]["status"], "insufficient")
        self.assertEqual(out["stress"]["status"], "insufficient")
        self.assertIsNone(out["regime"]["score"])
        self.assertIsNone(out["stress"]["score"])

    def test_only_fred_down_still_scores_what_it_can(self):
        # Yahoo 三個序列正常，FRED 全掛：A 只剩股市構面（35%）-> 資料不足
        S = make("calm")
        for k in ("hy_oas", "vix", "vix3m", "nfci", "stlfsi4", "walcl", "dff", "dollar", "t10y3m", "sahm", "icsa"):
            S[k] = None
        out = self.build(S)
        self.assertEqual(out["regime"]["status"], "insufficient")
        self.assertEqual(out["stress"]["status"], "insufficient")


class PublicOutputPrivacyTests(unittest.TestCase):
    def test_raw_values_do_not_appear_in_output(self):
        out = pipeline.build(make("calm"), CFG, TODAY, "2026-11-27T22:30:00Z")
        text = json.dumps(out, ensure_ascii=False)
        # 模擬資料裡刻意使用的獨特原始值，不能出現在公開輸出
        for raw in ("3.2137", "14.3719"):
            self.assertNotIn(raw, text)
        for forbidden in ("anchors", "features", "raw", "value"):
            self.assertNotIn(f'"{forbidden}"', text)

    def test_indicator_entries_only_have_light(self):
        out = pipeline.build(make("calm"), CFG, TODAY, "2026-11-27T22:30:00Z")
        for g in out["regime"]["dimensions"] + out["stress"]["blocks"]:
            for ind in g["indicators"]:
                self.assertEqual(set(ind), {"id", "name", "light"})
        for block in out["stress"]["blocks"]:
            self.assertIsNone(block["score"])


if __name__ == "__main__":
    unittest.main()
