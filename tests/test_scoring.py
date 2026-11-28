"""scoring.py 的單元測試。預期值皆為手算，並附算式說明。

執行：python -m unittest discover -s tests -v
"""

import json
import os
import sys
import unittest

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import scoring  # noqa: E402

with open(os.path.join(ROOT, "config", "indicators.yml"), encoding="utf-8") as f:
    CFG = yaml.safe_load(f)


def regime_features(**overrides):
    base = {
        "spx_vs_200dma": 5, "spx_200dma_slope": 1.5, "twii_vs_60dma": 0,
        "fed_balance_sheet_13w": 1, "fed_funds_3m_change": 0, "broad_dollar_3m": 0,
        "yield_curve_10y3m": 0.5, "sahm_rule": 0.1, "initial_claims_13w": 0,
        "vix_term_structure": 0.9, "spx_drawdown_52w": 5,
    }
    base.update(overrides)
    return base


def stress_features(**overrides):
    base = {"hy_oas": 4, "nfci": 0, "stlfsi4": 0, "vix": 20}
    base.update(overrides)
    return base


class InterpolateTests(unittest.TestCase):
    def test_middle_and_exact_anchor(self):
        a = [[-10, 0], [0, 50], [10, 100]]
        self.assertAlmostEqual(scoring.interpolate(5, a), 75)
        self.assertAlmostEqual(scoring.interpolate(0, a), 50)

    def test_clamps_to_endpoints(self):
        a = [[-10, 0], [10, 100]]
        self.assertEqual(scoring.interpolate(-99, a), 0)
        self.assertEqual(scoring.interpolate(99, a), 100)

    def test_decreasing_scores(self):
        a = [[0, 100], [10, 0]]
        self.assertAlmostEqual(scoring.interpolate(5, a), 50)

    def test_unsorted_anchors_are_sorted(self):
        a = [[10, 100], [-10, 0], [0, 50]]
        self.assertAlmostEqual(scoring.interpolate(5, a), 75)

    def test_none_stays_none_not_zero(self):
        self.assertIsNone(scoring.interpolate(None, [[0, 0], [1, 100]]))

    def test_duplicate_anchor_x_raises(self):
        with self.assertRaises(ValueError):
            scoring.interpolate(1, [[0, 0], [0, 50], [2, 100]])


class WeightedMeanTests(unittest.TestCase):
    def test_missing_weight_is_redistributed(self):
        mean, comp = scoring.weighted_mean([(80, 0.5), (None, 0.5)])
        self.assertAlmostEqual(mean, 80)  # 缺值不當 0，否則會是 40
        self.assertAlmostEqual(comp, 0.5)

    def test_all_missing(self):
        mean, comp = scoring.weighted_mean([(None, 1.0)])
        self.assertIsNone(mean)
        self.assertEqual(comp, 0.0)


class LightTests(unittest.TestCase):
    def test_boundaries_go_to_higher_bucket(self):
        idx = scoring.light_index
        self.assertEqual(idx(0), 0)
        self.assertEqual(idx(19.99), 0)
        self.assertEqual(idx(20), 1)
        self.assertEqual(idx(59.99), 2)
        self.assertEqual(idx(60), 3)
        self.assertEqual(idx(80), 4)
        self.assertEqual(idx(100), 4)

    def test_labels(self):
        self.assertEqual(scoring.make_light(70, CFG["lights"]["regime"])["label"], "偏強")
        self.assertEqual(scoring.make_light(70, CFG["lights"]["stress"])["label"], "高壓力")
        self.assertIsNone(scoring.make_light(None, CFG["lights"]["regime"]))


class RegimeStateTests(unittest.TestCase):
    def test_boundaries(self):
        s = scoring.regime_state
        self.assertEqual(s(40.01), "牛市")
        self.assertEqual(s(40), "偏多")
        self.assertEqual(s(10.01), "偏多")
        self.assertEqual(s(10), "盤整")
        self.assertEqual(s(-10), "盤整")
        self.assertEqual(s(-10.01), "偏空")
        self.assertEqual(s(-40), "偏空")
        self.assertEqual(s(-40.01), "熊市")

    def test_cap_only_lowers_bullish_states(self):
        self.assertEqual(scoring.apply_drawdown_cap("牛市", 22, 20), ("盤整", True))
        self.assertEqual(scoring.apply_drawdown_cap("偏空", 22, 20), ("偏空", False))
        self.assertEqual(scoring.apply_drawdown_cap("牛市", 20, 20), ("牛市", False))  # 剛好 20 不觸發
        self.assertEqual(scoring.apply_drawdown_cap("牛市", None, 20), ("牛市", False))


class ScoreRegimeTests(unittest.TestCase):
    def test_hand_calculated_total(self):
        # 股市趨勢 = 0.45*75 + 0.20*75 + 0.35*50 = 66.25
        # 流動性   = 0.40*75 + 0.30*60 + 0.30*65 = 67.5
        # 景氣     = 0.40*60 + 0.40*80 + 0.20*65 = 69
        # 避險情緒 = 0.50*60 + 0.50*80 = 70
        # 加權     = 0.35*66.25 + 0.25*67.5 + 0.25*69 + 0.15*70 = 67.8125
        # 總分     = 2*(67.8125-50) = 35.625
        r = scoring.score_regime(CFG, regime_features())
        scores = [round(g["score"], 4) for g in r["dimensions"]]
        self.assertEqual(scores, [66.25, 67.5, 69.0, 70.0])
        self.assertAlmostEqual(r["score"], 35.625)
        self.assertEqual(r["state"], "偏多")
        self.assertEqual(r["status"], "ok")
        self.assertAlmostEqual(r["completeness"], 1.0)
        self.assertEqual(r["light"]["label"], "偏強")  # (35.625+100)/2 = 67.8

    def test_missing_indicator_reweights_and_lowers_completeness(self):
        # 少 Sahm Rule：景氣 = (0.40*60 + 0.20*65)/0.60 = 61.6667，景氣完整度 0.6
        # 總體完整度 = 0.35 + 0.25 + 0.25*0.6 + 0.15 = 0.90
        r = scoring.score_regime(CFG, regime_features(sahm_rule=None))
        economy = r["dimensions"][2]
        self.assertAlmostEqual(economy["score"], 61.6667, places=3)
        self.assertAlmostEqual(economy["completeness"], 0.6)
        self.assertAlmostEqual(r["completeness"], 0.90)
        self.assertEqual(r["status"], "ok")

    def test_dimension_below_min_completeness_is_excluded(self):
        # 景氣全缺：完整度 0，被排除；總體完整度 = 0.75 -> 僅供參考
        r = scoring.score_regime(
            CFG, regime_features(yield_curve_10y3m=None, sahm_rule=None, initial_claims_13w=None)
        )
        self.assertFalse(r["dimensions"][2]["included"])
        self.assertAlmostEqual(r["completeness"], 0.75)
        self.assertEqual(r["status"], "reference_only")
        # 其餘三個構面重算：(0.35*66.25 + 0.25*67.5 + 0.15*70) / 0.75
        expected_base = (0.35 * 66.25 + 0.25 * 67.5 + 0.15 * 70) / 0.75
        self.assertAlmostEqual(r["score"], 2 * (expected_base - 50), places=4)

    def test_insufficient_gives_no_classification(self):
        only_equity = {k: None for k in regime_features()}
        only_equity.update({"spx_vs_200dma": 5, "spx_200dma_slope": 1.5, "twii_vs_60dma": 0})
        r = scoring.score_regime(CFG, only_equity)  # 完整度 0.35
        self.assertEqual(r["status"], "insufficient")
        self.assertIsNone(r["score"])
        self.assertIsNone(r["state"])

    def test_all_missing(self):
        r = scoring.score_regime(CFG, {k: None for k in regime_features()})
        self.assertEqual(r["status"], "insufficient")
        self.assertIsNone(r["score"])

    def test_drawdown_over_20_caps_bull_to_consolidation(self):
        bullish = regime_features(
            spx_vs_200dma=10, spx_200dma_slope=3, twii_vs_60dma=8,
            fed_balance_sheet_13w=2, fed_funds_3m_change=-100, broad_dollar_3m=-6,
            yield_curve_10y3m=2, sahm_rule=0, initial_claims_13w=-10,
            vix_term_structure=0.75, spx_drawdown_52w=22,
        )
        # 回檔 22% -> 分數 15 - (2/5)*15 = 9；避險情緒 = 0.5*100 + 0.5*9 = 54.5
        # 加權 = 0.35*100 + 0.25*100 + 0.25*100 + 0.15*54.5 = 93.175；總分 86.35 -> 本來是牛市
        r = scoring.score_regime(CFG, bullish)
        self.assertAlmostEqual(r["score"], 86.35, places=4)
        self.assertEqual(r["state"], "盤整")
        self.assertTrue(r["capped_by_drawdown"])


class ScoreStressTests(unittest.TestCase):
    def test_hand_calculated_total(self):
        # 壓力區塊 = 0.40*25 + 0.30*55 + 0.30*40 = 38.5；VIX 20 -> 30
        # 總分 = 0.70*38.5 + 0.30*30 = 35.95 -> 壓力偏低（20～40）
        r = scoring.score_stress(CFG, stress_features())
        self.assertAlmostEqual(r["blocks"][0]["score"], 38.5)
        self.assertAlmostEqual(r["score"], 35.95)
        self.assertEqual(r["light"]["label"], "壓力偏低")
        self.assertEqual(r["status"], "ok")

    def test_missing_hy_oas_reweights_block(self):
        # 壓力區塊 = (0.30*55 + 0.30*40)/0.60 = 47.5，區塊完整度 0.6
        # 總體完整度 = 0.70*0.6 + 0.30 = 0.72 -> 僅供參考
        r = scoring.score_stress(CFG, stress_features(hy_oas=None))
        self.assertAlmostEqual(r["blocks"][0]["score"], 47.5)
        self.assertAlmostEqual(r["completeness"], 0.72)
        self.assertEqual(r["status"], "reference_only")
        self.assertAlmostEqual(r["score"], 0.7 * 47.5 + 0.3 * 30)

    def test_missing_vix_excludes_vix_block(self):
        r = scoring.score_stress(CFG, stress_features(vix=None))
        self.assertFalse(r["blocks"][1]["included"])
        self.assertAlmostEqual(r["completeness"], 0.70)
        self.assertEqual(r["status"], "reference_only")
        self.assertAlmostEqual(r["score"], 38.5)

    def test_backtest_without_hy_oas_uses_remaining_two(self):
        # 回測情境：HY OAS 取不到，NFCI 與 STLFSI4 各半
        r = scoring.score_stress(CFG, stress_features(hy_oas=None, nfci=0.5, stlfsi4=2))
        # NFCI 0.5 -> 80；STLFSI4 2 -> 75；區塊 = 77.5
        self.assertAlmostEqual(r["blocks"][0]["score"], 77.5)


class GoldEquityFlagTests(unittest.TestCase):
    def f(self, spx, gold, stress=None, vix=None, hy=None):
        return scoring.gold_equity_flag(CFG, spx, gold, stress, vix, hy)

    def test_not_triggered(self):
        self.assertEqual(self.f(-3, -4)["state"], "none")
        self.assertEqual(self.f(-5, -2)["state"], "none")

    def test_triggered_exact_thresholds(self):
        self.assertEqual(self.f(-4, -3, stress=30, vix=20, hy=0)["state"], "yellow")

    def test_red_by_each_confirmation(self):
        self.assertEqual(self.f(-5, -4, stress=65, vix=20, hy=0.1)["state"], "red")
        self.assertEqual(self.f(-5, -4, stress=30, vix=36, hy=0.1)["state"], "red")
        self.assertEqual(self.f(-5, -4, stress=30, vix=20, hy=0.5)["state"], "red")

    def test_vix_exactly_35_is_not_red(self):
        self.assertEqual(self.f(-5, -4, stress=30, vix=35, hy=0)["state"], "yellow")

    def test_yellow_2022_style(self):
        r = self.f(-6, -5, stress=35, vix=25, hy=0.2)
        self.assertEqual(r["state"], "yellow")
        self.assertFalse(r["confirmation_unknown"])

    def test_yellow_when_confirmation_all_unknown(self):
        r = self.f(-6, -5)
        self.assertEqual(r["state"], "yellow")
        self.assertTrue(r["confirmation_unknown"])

    def test_unknown_when_returns_missing(self):
        self.assertEqual(self.f(None, -5)["state"], "unknown")
        self.assertEqual(self.f(-5, None)["state"], "unknown")


class TurningPointFlagTests(unittest.TestCase):
    def tp(self, pre, conds):
        return scoring.turning_point_flag(CFG, pre, conds)

    def test_precondition_false(self):
        self.assertEqual(self.tp(False, [True] * 4)["state"], "not_evaluated")

    def test_precondition_unknown(self):
        self.assertEqual(self.tp(None, [True] * 4)["state"], "insufficient")

    def test_met_and_not_met(self):
        self.assertEqual(self.tp(True, [True, True, True, False])["state"], "met")
        self.assertEqual(self.tp(True, [True, True, False, False])["state"], "not_met")

    def test_unknown_conditions_are_not_counted_as_false(self):
        self.assertEqual(self.tp(True, [True, True, None, None])["state"], "insufficient")
        self.assertEqual(self.tp(True, [True, True, True, None])["state"], "met")
        self.assertEqual(self.tp(True, [True, False, False, None])["state"], "not_met")


class PublicOutputTests(unittest.TestCase):
    def build(self):
        regime = scoring.score_regime(CFG, regime_features())
        stress = scoring.score_stress(CFG, stress_features())
        flags = {"gold_equity_drop": {"state": "none"}, "turning_point": {"state": "not_evaluated"}}
        return scoring.to_public(regime, stress, flags, "2026-11-27T22:30:00Z", {"equity": "2026-11-26"})

    def test_json_serializable(self):
        json.dumps(self.build(), ensure_ascii=False)

    def test_indicators_only_have_light_no_score_no_raw(self):
        pub = self.build()
        for g in pub["regime"]["dimensions"] + pub["stress"]["blocks"]:
            for ind in g["indicators"]:
                self.assertEqual(set(ind), {"id", "name", "light"})

    def test_single_indicator_block_score_is_hidden(self):
        # VIX 區塊只有一個指標，區塊分數等於 VIX 分數，公開會洩漏原始值
        pub = self.build()
        vix_block = pub["stress"]["blocks"][1]
        self.assertEqual(vix_block["id"], "vix_level")
        self.assertIsNone(vix_block["score"])
        self.assertIsNotNone(vix_block["light"])

    def test_regime_dimension_scores_are_published(self):
        pub = self.build()
        for dim in pub["regime"]["dimensions"]:
            self.assertIsNotNone(dim["score"])

    def test_stress_block_scores_are_hidden_to_prevent_subtraction(self):
        # 總分 = 0.7 x 壓力區塊 + 0.3 x VIX 分數；若同時公開總分與壓力區塊分數，
        # 相減即可算出 VIX 分數，再用公開錨點反推 VIX 原始值。
        pub = self.build()
        for block in pub["stress"]["blocks"]:
            self.assertIsNone(block["score"])
            self.assertIsNotNone(block["light"])
        self.assertIsNotNone(pub["stress"]["score"])

    def test_no_forbidden_keys(self):
        text = json.dumps(self.build(), ensure_ascii=False)
        for forbidden in ("raw", "anchors", "features", "value"):
            self.assertNotIn(f'"{forbidden}"', text)


class ConfigSanityTests(unittest.TestCase):
    def test_dimension_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(d["weight"] for d in CFG["regime"]["dimensions"]), 1.0)
        self.assertAlmostEqual(sum(b["weight"] for b in CFG["stress"]["blocks"]), 1.0)

    def test_indicator_weights_sum_to_one_per_group(self):
        for group in CFG["regime"]["dimensions"] + CFG["stress"]["blocks"]:
            self.assertAlmostEqual(sum(i["weight"] for i in group["indicators"]), 1.0, msg=group["id"])

    def test_gold_equity_vix_threshold_is_35(self):
        # 2026-11 回測：VIX>30 在 2022 年只製造誤報，2008、2020 不受影響，調為 35
        self.assertEqual(CFG["flags"]["gold_equity_drop"]["red_if_any"]["vix_min"], 35)

    def test_anchors_valid(self):
        for group in CFG["regime"]["dimensions"] + CFG["stress"]["blocks"]:
            for ind in group["indicators"]:
                xs = [a[0] for a in ind["anchors"]]
                self.assertEqual(len(xs), len(set(xs)), msg=ind["id"])
                for _, y in ind["anchors"]:
                    self.assertTrue(0 <= y <= 100, msg=ind["id"])

    def test_regime_anchors_direction(self):
        # A：越偏多分數越高，回檔、利差等「越大越差」的指標，錨點分數應隨原始值遞減
        by_id = {i["id"]: i for d in CFG["regime"]["dimensions"] for i in d["indicators"]}
        for iid in ("spx_drawdown_52w", "sahm_rule", "vix_term_structure", "fed_funds_3m_change",
                    "broad_dollar_3m", "initial_claims_13w"):
            pts = sorted(by_id[iid]["anchors"])
            ys = [y for _, y in pts]
            self.assertEqual(ys, sorted(ys, reverse=True), msg=iid)

    def test_stress_anchors_direction(self):
        # B：原始值越大壓力越高，分數應隨原始值遞增
        for b in CFG["stress"]["blocks"]:
            for ind in b["indicators"]:
                ys = [y for _, y in sorted(ind["anchors"])]
                self.assertEqual(ys, sorted(ys), msg=ind["id"])


if __name__ == "__main__":
    unittest.main()
