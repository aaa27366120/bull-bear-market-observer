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


class DiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.S = bt.apply_publication_lag(make("crisis"))
        d = dates(cls.S)
        cls.rows, cls.internals = bt.run_episode_detailed(cls.S, CFG, d[200], d[399])
        ep = {"id": "x", "name": "模擬危機", "peak": d[200], "trough": d[340], "events": {}}
        cls.summary = bt.summarize(cls.rows, ep, cls.S["spx"], cls.internals)

    def test_internals_align_with_rows(self):
        self.assertEqual(len(self.rows), len(self.internals))

    def test_run_episode_wrapper_matches_detailed(self):
        d = dates(self.S)
        self.assertEqual(bt.run_episode(self.S, CFG, d[380], d[399]), self.rows[-20:])

    def test_last_day_has_all_four_conditions_true(self):
        self.assertEqual(self.internals[-1]["conditions"], [True, True, True, True])

    def test_summary_has_condition_diagnostics(self):
        self.assertEqual(len(self.summary["cond_rate_evaluated"]), 4)
        self.assertEqual(len(self.summary["cond_rate_met"]), 4)
        self.assertEqual(self.summary["tp_first_met_conditions"].count(True) >= 3, True)
        self.assertTrue(self.summary["tp_met_ranges"])
        # 轉折確認成立的日子，至少有 3 項條件成立，所以每項的成立比例都不會是 0
        self.assertTrue(all(r is not None for r in self.summary["cond_rate_met"]))

    def test_b_reference_share_is_reported(self):
        self.assertIn("b_ref_pct", self.summary)

    def test_red_reason_vix_only(self):
        S = make("calm")
        spx, gold, vix = S["spx"].copy(), S["gold"].copy(), S["vix"].copy()
        spx.iloc[-1] = spx.iloc[-11] * 0.94
        gold.iloc[-1] = gold.iloc[-11] * 0.96
        vix.iloc[-2] = 35.0  # 公布落後 1 天，倒數第二筆在最後一個交易日才看得到
        S.update({"spx": spx, "gold": gold, "vix": vix})
        L = bt.apply_publication_lag(S)
        d = dates(L)
        rows, its = bt.run_episode_detailed(L, CFG, d[-1], d[-1])
        self.assertEqual(rows[0]["gold_equity"], "red")
        self.assertEqual(rows[0]["red_reason"], "VIX>30")
        self.assertTrue(its[0]["triggered"])

    def test_csv_rows_contain_no_raw_values(self):
        for row in self.rows:
            self.assertNotIn("vix", row)
            self.assertEqual(
                set(row),
                {"date", "a_score", "a_state", "a_status", "a_completeness", "b_score", "b_label",
                 "b_status", "b_completeness", "gold_equity", "red_reason", "turning_point"},
            )


class SensitivityTests(unittest.TestCase):
    def it(self, triggered, b, vix):
        return {"triggered": triggered, "b_score": b, "vix": vix, "had_high": None, "conditions": [None] * 4}

    def test_classification(self):
        c = bt.classify_gold_flag
        self.assertEqual(c(self.it(False, 90, 50), 60, 30), "none")
        self.assertEqual(c(self.it(True, 70, 20), 60, 30), "red")       # B 觸發
        self.assertEqual(c(self.it(True, 40, 35), 60, 30), "red")       # VIX 觸發
        self.assertEqual(c(self.it(True, 40, 35), 60, 40), "yellow")    # 提高 VIX 門檻後變黃
        self.assertEqual(c(self.it(True, 40, 35), 60, None), "yellow")  # 不使用 VIX
        self.assertEqual(c(self.it(True, None, None), 60, 30), "yellow")
        self.assertEqual(c(self.it(True, 40, 30), 60, 30), "yellow")    # VIX 剛好 30 不算

    def test_table_counts(self):
        its = [self.it(True, 40, 35), self.it(True, 70, 20), self.it(True, 40, 25), self.it(False, 90, 50)]
        t = bt.sensitivity({"e": its})["table"]
        self.assertEqual(t["基準：B≥60 或 VIX>30"]["e"], (2, 1))
        self.assertEqual(t["B≥60 或 VIX>35"]["e"], (1, 2))
        self.assertEqual(t["B≥60 或 VIX>40"]["e"], (1, 2))
        self.assertEqual(t["只看 B≥60（不用 VIX）"]["e"], (1, 2))

    def test_report_includes_new_sections(self):
        S = bt.apply_publication_lag(make("crisis"))
        d = dates(S)
        rows, its = bt.run_episode_detailed(S, CFG, d[300], d[399])
        ep = {"id": "x", "name": "模擬危機", "peak": d[300], "trough": d[340], "events": {}}
        res = [bt.summarize(rows, ep, S["spx"], its)]
        text = bt.render_report(res, "測試", bt.sensitivity({"x": its}))
        for h in ("## 敏感度：股金同跌的紅色條件", "轉折確認的四項條件", "轉折確認成立區間", "B 正常／僅供參考"):
            self.assertIn(h, text)
        self.assertNotIn("nan", text.lower())


class BuildDetailedTests(unittest.TestCase):
    def test_public_output_identical_to_build(self):
        import pipeline
        S = make("calm")
        from datetime import date
        a = pipeline.build(S, CFG, date(2026, 11, 27), "t")
        b, details = pipeline.build_detailed(S, CFG, date(2026, 11, 27), "t")
        self.assertEqual(a, b)
        self.assertEqual(len(details["conditions"]), 4)
        self.assertIn("features", details)


if __name__ == "__main__":
    unittest.main()
