"""fetch_and_score.py 的測試。不連網：以 mock 取代網路與外部套件。"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import fetch_and_score as fs  # noqa: E402

SECRET = "SECRET-KEY-abc123"


class ParseFredTests(unittest.TestCase):
    def test_dot_values_are_skipped(self):
        payload = {"observations": [
            {"date": "2026-11-23", "value": "3.10"},
            {"date": "2026-11-24", "value": "."},
            {"date": "2026-11-25", "value": "3.30"},
        ]}
        s = fs.parse_fred_observations(payload)
        self.assertEqual(len(s), 2)
        self.assertAlmostEqual(s.iloc[-1], 3.30)

    def test_sorted_by_date(self):
        payload = {"observations": [
            {"date": "2026-11-25", "value": "2"}, {"date": "2026-11-23", "value": "1"},
        ]}
        s = fs.parse_fred_observations(payload)
        self.assertEqual(list(s.values), [1.0, 2.0])

    def test_empty_or_all_missing(self):
        self.assertIsNone(fs.parse_fred_observations({}))
        self.assertIsNone(fs.parse_fred_observations({"observations": []}))
        self.assertIsNone(fs.parse_fred_observations({"observations": [{"date": "2026-11-25", "value": "."}]}))


class KeyLeakTests(unittest.TestCase):
    def test_error_messages_never_contain_the_api_key(self):
        boom = RuntimeError(f"request failed: https://x/?api_key={SECRET}")
        out = io.StringIO()
        with mock.patch.object(fs.urllib.request, "urlopen", side_effect=boom), \
                mock.patch.object(fs.time, "sleep"), contextlib.redirect_stdout(out):
            result = fs.fetch_fred("VIXCLS", SECRET, "2026-01-01", retries=1)
        self.assertIsNone(result)
        self.assertNotIn(SECRET, out.getvalue())
        self.assertIn("VIXCLS", out.getvalue())
        self.assertIn("RuntimeError", out.getvalue())


def calm_series(end):
    n = 400
    t = np.arange(n, dtype=float)
    ix = pd.bdate_range(end=end, periods=n)

    def s(v):
        return pd.Series(np.asarray(v, dtype=float), index=ix)

    spx = 4000 * (1.0004 ** t)
    return {
        "spx": s(spx), "twii": s(spx * 0.2), "gold": s(np.full(n, 2000.0)),
        "hy_oas": s(np.full(n, 3.2137)), "vix": s(np.full(n, 14.3719)), "vix3m": s(np.full(n, 17.0)),
        "nfci": s(np.full(n, -0.5)), "stlfsi4": s(np.full(n, -0.8)),
        "walcl": s(7_000_000 + t * 1000), "dff": s(np.full(n, 3.5)), "dollar": s(np.full(n, 120.0)),
        "t10y3m": s(np.full(n, 0.5)), "sahm": s(np.full(n, 0.1)), "icsa": s(np.full(n, 210000.0)),
    }


class MainTests(unittest.TestCase):
    def run_main(self, loaded, existing=None, key=SECRET):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "docs", "data.json")
            if existing is not None:
                os.makedirs(os.path.dirname(out_path))
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(existing)
            env = {"FRED_API_KEY": key} if key is not None else {}
            buf = io.StringIO()
            with mock.patch.object(fs, "OUTPUT_PATH", out_path), \
                    mock.patch.object(fs, "load_all", return_value=loaded), \
                    mock.patch.dict(os.environ, env, clear=False), \
                    contextlib.redirect_stdout(buf):
                if key is None:
                    os.environ.pop("FRED_API_KEY", None)
                code = fs.main()
            content = None
            if os.path.exists(out_path):
                with open(out_path, encoding="utf-8") as f:
                    content = f.read()
            return code, content, buf.getvalue()

    def test_missing_key_fails_without_writing(self):
        code, content, _ = self.run_main({}, key=None)
        self.assertEqual(code, 1)
        self.assertIsNone(content)

    def test_success_writes_valid_public_json(self):
        end = pd.Timestamp(datetime.now(timezone.utc).date()) - pd.Timedelta(days=1)
        code, content, log_text = self.run_main(calm_series(end))
        self.assertEqual(code, 0)
        data = json.loads(content)
        self.assertEqual(data["regime"]["status"], "ok")
        self.assertEqual(data["stress"]["status"], "ok")
        # 日誌與輸出都不能出現金鑰或原始數值
        self.assertNotIn(SECRET, log_text)
        self.assertNotIn(SECRET, content)
        for raw in ("3.2137", "14.3719"):
            self.assertNotIn(raw, content)
            self.assertNotIn(raw, log_text)

    def test_total_failure_keeps_previous_file(self):
        previous = '{"keep": "me"}\n'
        code, content, _ = self.run_main({k: None for k in ("spx", "twii", "gold", "hy_oas", "vix")}, existing=previous)
        self.assertEqual(code, 1)
        self.assertEqual(content, previous)


class DropUnfinishedBarTests(unittest.TestCase):
    def s(self, last_day):
        ix = pd.DatetimeIndex(["2026-11-24", "2026-11-25", last_day])
        return pd.Series([1.0, 2.0, 3.0], index=ix)

    def utc(self, text):
        return datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)

    def test_taiwan_intraday_manual_run_is_dropped(self):
        # 台灣時間 13:10（UTC 05:10）手動執行：台股還沒收盤，今天那一筆是盤中資料
        out = fs.drop_unfinished_last_bar(self.s("2026-11-27"), self.utc("2026-11-27 05:10"), fs.MARKET_FINAL_UTC["twii"])
        self.assertEqual(len(out), 2)
        self.assertEqual(out.index[-1], pd.Timestamp("2026-11-25"))

    def test_taiwan_after_close_is_kept(self):
        # 排程時間 UTC 22:30（台灣隔日 06:30）：台股當天早已收盤
        out = fs.drop_unfinished_last_bar(self.s("2026-11-27"), self.utc("2026-11-27 22:30"), fs.MARKET_FINAL_UTC["twii"])
        self.assertEqual(len(out), 3)

    def test_us_session_in_progress_is_dropped_and_after_close_kept(self):
        spx = fs.MARKET_FINAL_UTC["spx"]
        self.assertEqual(len(fs.drop_unfinished_last_bar(self.s("2026-11-27"), self.utc("2026-11-27 15:00"), spx)), 2)
        self.assertEqual(len(fs.drop_unfinished_last_bar(self.s("2026-11-27"), self.utc("2026-11-27 22:30"), spx)), 3)

    def test_gold_futures_scheduled_run_keeps_the_day(self):
        gold = fs.MARKET_FINAL_UTC["gold"]
        self.assertEqual(len(fs.drop_unfinished_last_bar(self.s("2026-11-27"), self.utc("2026-11-27 05:10"), gold)), 2)
        self.assertEqual(len(fs.drop_unfinished_last_bar(self.s("2026-11-27"), self.utc("2026-11-27 22:30"), gold)), 3)

    def test_previous_day_bar_is_kept(self):
        out = fs.drop_unfinished_last_bar(self.s("2026-11-26"), self.utc("2026-11-27 05:10"), fs.MARKET_FINAL_UTC["twii"])
        self.assertEqual(len(out), 3)

    def test_future_dated_bar_is_dropped(self):
        out = fs.drop_unfinished_last_bar(self.s("2026-11-28"), self.utc("2026-11-27 22:30"), fs.MARKET_FINAL_UTC["spx"])
        self.assertEqual(len(out), 2)

    def test_none_and_empty(self):
        self.assertIsNone(fs.drop_unfinished_last_bar(None, self.utc("2026-11-27 05:10"), (6, 0)))


if __name__ == "__main__":
    unittest.main()
