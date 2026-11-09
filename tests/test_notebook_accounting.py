"""Execute notebook request and stage interfaces with an offline client."""
import base64
from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import uuid


ROOT = Path(__file__).resolve().parents[1]


class RateLimitError(Exception):
    response = SimpleNamespace(headers={"retry-after": "0"})


class TimeoutError(Exception):
    pass


class ConnectionError(Exception):
    pass


class StatusError(Exception):
    status_code = 500


class NotebookAccountingTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)
        for name in ("attempts", "images"):
            (self.output / name).mkdir()
        self.events = []
        self.clients = []
        self.postfixes = []
        self.reads = []
        self.ns = dict(globals(), OUTPUT_DIR=self.output, MAX_API_ATTEMPTS=20,
                       MAX_ATTEMPTS_PER_IMAGE=3, RETRY_UNCERTAIN_REQUESTS=False,
                       RETRY_RATE_LIMITED=False, REQUEST_TIMEOUT_SECONDS=60,
                       MAX_BACKOFF_SECONDS=0, SAMPLING_SEED=42, VALIDATION_PER_CLASS=500,
                       STAGE_PER_CLASS={}, CALIBRATION_BINS=10, PROBABILITY_SUM_TOLERANCE=1e-6,
                       PRICING_SOURCE="offline fixture", PRICING_CHECKED_DATE="offline fixture",
                       TOKEN_RATES_USD_PER_MILLION={"uncached_input": 2., "cached_input": .1,
                                                    "cache_write": 2.5, "output": 10.},
                       np=SimpleNamespace(mean=statistics.mean),
                       openai=SimpleNamespace(RateLimitError=RateLimitError, APITimeoutError=TimeoutError,
                                              APIConnectionError=ConnectionError, APIStatusError=StatusError))
        notebook = json.loads((ROOT / "notebooks/cifar10_responses_colab.ipynb").read_text())
        for index in (6, 12):
            exec(compile("".join(notebook["cells"][index]["source"]),
                         f"cifar10_responses_colab:cell{index}", "exec"), self.ns)
        original_read = self.ns["read_json"]

        def read_json(path):
            if Path(path).parent == self.output / "attempts":
                self.reads.append(path)
            return original_read(path)

        self.ns["read_json"] = read_json
        original_classify = self.ns["classify"]
        self.original_classify = original_classify
        self.classify = lambda png, **kwargs: original_classify(
            png, client_factory=self.client_factory, sleeper=lambda _: None, **kwargs)
        self.ns["classify"] = self.classify
        postfixes = self.postfixes

        class Progress:
            def __init__(self, rows, **kwargs):
                self.rows = rows

            def __iter__(self):
                return iter(self.rows)

            def set_postfix(self, value):
                postfixes.append(value)

        self.ns["tqdm"] = Progress

    def response(self, usage=True):
        raw = {"status": "completed", "output": [{"type": "message", "role": "assistant",
               "status": "completed", "content": [{"type": "output_text", "text": '{"label":"cat"}'}]}]}
        if usage:
            raw["usage"] = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
                            "input_tokens_details": {"cached_tokens": 20, "cache_write_tokens": 10,
                                                     "cache_write_tokens_details": {"ttl_30m_tokens": 4,
                                                                                    "ttl_12h_tokens": 6}},
                            "output_tokens_details": {"reasoning_tokens": 5}}
        return raw

    def client_factory(self):
        def create(**request):
            records = self.ns["attempt_records"]()
            self.assertEqual(records[-1]["state"], "intent")
            self.assertEqual(request, self.ns["request_body"](self.current_png))
            event = self.events.pop(0)
            if isinstance(event, BaseException):
                raise event
            return SimpleNamespace(model_dump=lambda **kwargs: event)

        client = SimpleNamespace(responses=SimpleNamespace(create=create), close=Mock())
        self.clients.append(client)
        return client

    def run_stage(self, payloads):
        rows = []
        for png in payloads:
            image_hash = hashlib.sha256(png).hexdigest()
            (self.output / "images" / (image_hash + ".png")).write_bytes(png)
            rows.append({"image_sha256": image_hash, "cache_key": self.ns["cache_key"](png)})
        classify = self.classify

        def classified(png, **kwargs):
            self.current_png = png
            return classify(png, **kwargs)

        self.ns["classify"] = classified
        manifest = {"stage": "offline", "split": "train_validation",
                    "protocol": self.ns["protocol"](), "rows": rows}
        with redirect_stdout(io.StringIO()):
            self.ns["run_stage"](manifest, True)
        return self.ns["read_json"](self.output / "stage_sessions.json")[-1]

    def assert_accounting_matches_ledger(self, accounting):
        expected = self.ns["usage_summary"](self.ns["attempt_records"]())
        for key, value in accounting.summary().items():
            if isinstance(value, float):
                self.assertAlmostEqual(value, expected[key])
            else:
                self.assertEqual(value, expected[key])

    def test_stage_progress_and_cached_rerun_read_ledger_linearly(self):
        self.events = [self.response() for _ in range(8)]
        payloads = [bytes([index]) for index in range(8)]
        session = self.run_stage(payloads)
        self.assertEqual(session["counts"], {"answered": 8})
        cost = .000382
        self.assertEqual(self.postfixes[-1], {"known estimated USD": f"${8*cost:.6f}",
                         "projected 1000 USD": f"${1000*cost:.6f}", "projection observed images": 8,
                         "priced attempts": "8/8"})
        for client in self.clients:
            client.close.assert_called_once()
        self.reads.clear()
        previous_postfix = self.postfixes[-1]
        self.run_stage(payloads)
        self.assertEqual(len(self.clients), 8)
        self.assertEqual(len(self.reads), 16)
        self.assertTrue(all(value == previous_postfix for value in self.postfixes[-8:]))
        self.assertEqual(self.ns["cost_progress"](), previous_postfix)
        summary = self.ns["usage_summary"](self.ns["attempt_records"]())
        self.assertEqual(summary["token_totals_available"], {"input_tokens": 800, "output_tokens": 160,
                         "total_tokens": 960, "cached_input_tokens": 160, "cache_write_tokens": 80,
                         "reasoning_tokens": 40})

    def test_retries_unknown_usage_and_budget_keep_durable_totals(self):
        self.ns["MAX_API_ATTEMPTS"] = 3
        self.events = [RateLimitError(), self.response(), self.response(usage=False)]
        session = self.run_stage([b"retry", b"unknown", b"over-budget"])
        self.assertEqual(session["counts"], {"answered": 2, "pending": 1})
        records = self.ns["attempt_records"]()
        self.assertEqual([r["state"] for r in records], ["rate_limit", "returned", "returned"])
        self.assertEqual(self.postfixes[-1], {"known estimated USD": "$0.000382",
                         "projected 1000 USD": "unavailable", "projection observed images": 0,
                         "priced attempts": "1/3"})
        accounting = self.ns["CostAccounting"](records)
        self.assert_accounting_matches_ledger(accounting)
        for record in records:
            accounting.update(record)
        self.assert_accounting_matches_ledger(accounting)
        self.run_stage([b"retry", b"unknown"])
        self.assertEqual(len(self.clients), 2)

    def test_crash_reservation_and_explicit_uncertain_resume(self):
        self.events = [KeyboardInterrupt()]
        with self.assertRaises(KeyboardInterrupt):
            self.run_stage([b"interrupted"])
        self.clients[0].close.assert_called_once()
        self.assertEqual(self.ns["attempt_records"]()[0]["state"], "intent")
        self.run_stage([b"interrupted"])
        self.assertEqual(len(self.clients), 1)
        self.ns["RETRY_UNCERTAIN_REQUESTS"] = True
        self.events = [TimeoutError(), self.response()]
        self.run_stage([b"interrupted"])
        records = self.ns["attempt_records"]()
        self.assertEqual([r["state"] for r in records], ["intent", "uncertain", "returned"])
        self.assertEqual(self.postfixes[-1]["priced attempts"], "1/3")
        self.assertEqual(self.postfixes[-1]["projected 1000 USD"], "unavailable")
        self.assert_accounting_matches_ledger(self.ns["CostAccounting"](records))

    def test_persisted_updates_match_ledger_through_failure_and_success(self):
        accounting = self.ns["CostAccounting"]([])
        observations = []

        def persisted(record):
            accounting.update(record)
            self.assert_accounting_matches_ledger(accounting)
            observations.append(record["state"])

        self.current_png = b"server-error"
        self.events = [StatusError()]
        result = self.classify(self.current_png, on_record=persisted)
        self.assertEqual(result["status"], "pending")
        self.current_png = b"success"
        self.events = [self.response()]
        result = self.classify(self.current_png, on_record=persisted)
        self.assertEqual(result["status"], "answered")
        self.assertEqual(observations, ["intent", "uncertain", "intent", "returned"])
        self.assertEqual(accounting.summary()["projection_observed_images"], 1)
        self.assertEqual(accounting.summary()["priced_attempts"], 1)

    def test_rate_limit_resume_and_initialization_failure(self):
        self.ns["MAX_ATTEMPTS_PER_IMAGE"] = 1
        self.events = [RateLimitError()]
        self.run_stage([b"rate-limited"])
        self.run_stage([b"rate-limited"])
        self.assertEqual(len(self.clients), 1)
        self.ns["RETRY_RATE_LIMITED"] = True
        self.events = [self.response()]
        self.run_stage([b"rate-limited"])
        self.assertEqual(self.postfixes[-1]["priced attempts"], "1/2")
        self.assertEqual(self.postfixes[-1]["projection observed images"], 0)
        before = self.ns["attempt_count"]()
        with self.assertRaises(RuntimeError):
            self.original_classify(b"initialization", client_factory=Mock(side_effect=RuntimeError("offline initialization failure")))
        self.assertEqual(self.ns["attempt_count"](), before)


class NotebookIgnoreTest(unittest.TestCase):
    def test_git_ignores_outputs_and_backups_at_both_execution_locations(self):
        paths = [prefix + path for prefix in ("", "notebooks/") for path in (
            "cifar10_responses_sol_results/dataset/data.bin",
            "cifar10_responses_sol_results/images/cache.png",
            "cifar10_responses_sol_results/attempts/request.json",
            "cifar10_responses_sol_results/exports/metrics.json",
            "cifar10_responses_sol_results_backup.zip")]
        visible = ["notebooks/cifar10_responses_colab.ipynb", "results/summary.json"]
        result = subprocess.run(["git", "-c", "core.excludesFile=/dev/null", "check-ignore",
                                 "--no-index", "--stdin"], cwd=ROOT,
                                input="\n".join(paths + visible) + "\n", text=True,
                                capture_output=True, check=True)
        self.assertEqual(set(result.stdout.splitlines()), set(paths))


if __name__ == "__main__":
    unittest.main()
