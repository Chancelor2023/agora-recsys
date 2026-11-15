"""Verify config routing and existing CLI behaviour without installed packages."""
import contextlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from agora.evaluation.__main__ import main
from agora.evaluation.config import load_cutoffs

ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def test_shared_cutoffs(self):
        path = ROOT / "configs/experiment.toml"
        self.assertEqual(load_cutoffs(path), (10, 20))
        self.assertEqual(load_cutoffs(path, "supplementary"), (5, 5))

    def run_cli(self, extra):
        argv = ["agora.evaluation", "--predictions",
                str(ROOT / "examples/mock_predictions.json"), "--labels",
                str(ROOT / "examples/mock_labels.json"), *extra]
        output = io.StringIO()
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(output):
            main()
        return json.loads(output.getvalue())

    def test_cli_default_unchanged(self):
        result = self.run_cli([])
        self.assertAlmostEqual(result["metrics"]["Recall@20"], 5 / 6)

    def test_cli_supplementary_configuration(self):
        result = self.run_cli(["--config", str(ROOT / "configs/experiment.toml"),
                               "--scope", "supplementary"])
        self.assertEqual(set(result["metrics"]), {"NDCG@5", "Recall@5"})
        self.assertEqual(result["evaluation_settings"]["scope"], "supplementary")

    def test_cli_explicit_cutoff_override(self):
        result = self.run_cli(["--config", str(ROOT / "configs/experiment.toml"),
                               "--ndcg-k", "1"])
        self.assertEqual(set(result["metrics"]), {"NDCG@1", "Recall@20"})

    def test_config_rejects_bad_policy_or_cutoff(self):
        for contents in (
            b'[evaluation]\nzero_positive_policy="zero"\nndcg_k=10\nrecall_k=20',
            b'[evaluation]\nzero_positive_policy="exclude_and_report_count"\nndcg_k=0\nrecall_k=20',
        ):
            with self.subTest(contents=contents), patch("builtins.open",
                    return_value=io.BytesIO(contents)), self.assertRaises(ValueError):
                load_cutoffs("unused.toml")


if __name__ == "__main__":
    unittest.main()
