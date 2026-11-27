"""backtest.py 的測試（模擬資料，不需要網路）。"""

import os
import sys
import unittest

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import backtest as bt  # noqa: E402
from test_pipeline import CFG, make  # noqa: E402


def dates(S):
    return [d.strftime("%Y-%m-%d") for d in S["spx"].index]


class LagTests(unittest.TestCase):
    def test_lag_shifts_index_forward(self):
        S = make("calm")
        L = bt.apply_publication_lag(S)
        self.assertEqual(L["nfci"].index[-1] - S["nfci"].index[-1], pd.Timedelta(days=5))
        self.assertEqual(L["sahm"].index[-1] - S["sahm"].index[-1], pd.Timedelta(days=35))
        self.assertEqual(L["spx"].index[-1], S["spx"].index[-1])  # 股價不延遲

    def test_does_not_mutate_input(self):
        S = make("calm")
        before = S["nfci"].index[-1]
        bt.apply_publication_lag(S)
        self.assertEqual(S["nfci"].index[-1], before)

    def test_none_stays_none(self):
        S = make("calm")
        S["hy_oas"] = None
        self.assertIsNone(bt.apply_publication_lag(S)["hy_oas"])


class RangesTests(unittest.TestCase):
    def test_compresses_consecutive_days(self):
        df = pd.DataFrame({"date": ["d1", "d2", "d3", "d4", "d5", "d6"]})
        mask = pd.Series([True, True, False, True, False, True])
        self.assertEqual(bt._ranges(df, mask), ["d1～d2", "d4", "d6"])

    def test_empty(self):
        df = pd.DataFrame({"date": ["d1", "d2"]})
        self.assertEqual(bt._ranges(df, pd.Series([False, False])), [])


class CrisisReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.S = bt.apply_publication_lag(make("crisis"))
        d = dates(cls.S)
        cls.start, cls.end = d[200], d[399]
        cls.peak, cls.trough = d[200], d[340]
        cls.rows = bt.run_episode(cls.S, CFG, cls.start, cls.end)
        cls.ep = {"id": "x", "name": "模擬危機", "peak": cls.peak, "trough": cls.trough, "events": {}}
        cls.summary = bt.summarize(cls.rows, cls.ep, cls.S["spx"])

    def test_one_row_per_trading_day(self):
        self.assertEqual(len(self.rows), 200)
        self.assertEqual(self.rows[0]["date"], self.start)
        self.assertEqual(self.rows[-1]["date"], self.end)

    def test_stress_peaks_near_the_trough(self):
        by_date = {r["date"]: r for r in self.rows}
        self.assertGreaterEqual(by_date[self.trough]["b_score"], 80)
        self.assertEqual(by_date[self.trough]["b_label"], "危機級")

    def test_regime_turns_bearish_during_the_decline(self):
        self.assertIsNotNone(self.summary["a_first_bearish"])

    def test_b_warns_before_or_at_the_trough(self):
        self.assertIsNotNone(self.summary["b_first_60"])
        self.assertLessEqual(self.summary["b60_vs_trough_days"], 0)

    def test_turning_point_comes_after_the_trough(self):
        self.assertIsNotNone(self.summary["tp_first_met"])
        self.assertGreater(self.summary["tp_vs_trough_days"], 0)
        self.assertEqual(self.rows[-1]["turning_point"], "met")

    def test_max_drawdown_matches_hand_calculation(self):
        # 高點在 t=200：5000 - 200*(1500/340) = 4117.65；低點 3500 -> 回檔 15.0%
        self.assertAlmostEqual(self.summary["max_drawdown_pct"], 15.0, delta=0.2)

    def test_report_renders(self):
        text = bt.render_report([self.summary], "2026-11-27 00:00 UTC")
        for heading in ("# 歷史回測報告", "## 方法與限制", "## 總覽", "## 模擬危機", "## 怎麼讀這份報告"):
            self.assertIn(heading, text)
        self.assertNotIn("nan", text.lower())
        self.assertNotIn("None", text)


class CalmControlTests(unittest.TestCase):
    def test_calm_period_has_no_red_flags_and_low_stress(self):
        S = bt.apply_publication_lag(make("calm"))
        d = dates(S)
        rows = bt.run_episode(S, CFG, d[250], d[399])
        summary = bt.summarize(rows, {"id": "c", "name": "平靜", "peak": None, "trough": None, "events": {}}, S["spx"])
        self.assertEqual(summary["gold_red_days"], 0)
        self.assertEqual(summary["b_days_60"], 0)
        self.assertEqual(summary["tp_met_days"], 0)
        self.assertLess(summary["b_max"][0], 20)


if __name__ == "__main__":
    unittest.main()
