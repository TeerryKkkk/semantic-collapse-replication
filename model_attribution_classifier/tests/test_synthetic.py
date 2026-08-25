from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

from model_attribution_classifier.canonical_parser import parse_transcript
from model_attribution_classifier.pipeline import (
    FAMILY_ORDER,
    build_fold,
    build_independent_reference,
    load_independent_manifest,
    parse_independent_runs,
)


def transcript_text(horizon: int, marker: str) -> str:
    lines: list[str] = []
    for round_id in range(1, horizon + 1):
        lines.extend([
            f"===== Round {round_id} order: ['AAA', 'BBB', 'CCC'] =====",
            f"[Round {round_id}] (MAIN) AAA said: '{marker} round {round_id} canonical response.'",
            "continuation content",
            "    -> action_name=Example",
        ])
    return "\n".join(lines) + "\n"


class ParserTests(unittest.TestCase):
    def test_structural_lines_are_excluded_and_rounds_do_not_cross(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.txt"
            path.write_text(
                "===== Round 1 order: ['AAA', 'BBB', 'CCC'] =====\n"
                "[Round 1] (MAIN) AAA said: 'first'\n"
                "legitimate continuation\n"
                "Group established by AAA\n"
                "===== Round 2 order: ['AAA', 'BBB', 'CCC'] =====\n"
                "[Round 2] (REACTION) BBB said: 'second'\n"
                "    -> action_name=Example\n",
                encoding="utf-8",
            )
            result = parse_transcript(path, "GPT-4o-mini", 1)
            result.raise_for_structural_errors()
            self.assertEqual(sorted(result.documents), [1, 2])
            self.assertIn("legitimate continuation", result.documents[1].text)
            self.assertNotIn("Group established", result.documents[1].text)
            self.assertNotIn("Round 2", result.documents[1].text)


class DesignTests(unittest.TestCase):
    def test_independent_reference_and_loro_logic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "independent.csv"
            rows = []
            for role, horizon in (("reference", 200), ("test", 1000)):
                for family_index, family in enumerate(FAMILY_ORDER):
                    for run_id in (1, 2, 3):
                        path = root / f"{role}_{family_index}_{run_id}.txt"
                        path.write_text(transcript_text(horizon, f"{role} {family} run {run_id}"), encoding="utf-8")
                        rows.append({"role": role, "model_family": family, "run_id": run_id, "transcript_path": path.name})
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["role", "model_family", "run_id", "transcript_path"])
                writer.writeheader()
                writer.writerows(rows)
            specs = load_independent_manifest(manifest)
            references, tests = parse_independent_runs(specs)
            data = build_independent_reference(references, tests)
            self.assertEqual(len(references), 15)
            self.assertEqual(len(tests), 15)
            self.assertEqual(len(data.training), 3000)
            self.assertEqual(len(data.test), 15000)
            self.assertEqual(len(data.reference_run_counts), 15)
            self.assertTrue(math.isclose(float(data.sample_weights.mean()), 1.0, abs_tol=1e-12))

            fold = build_fold(tests, 1)
            self.assertEqual(fold.held_out_run_id, 3)
            self.assertEqual(len(fold.training), 2000)
            self.assertEqual(len(fold.test), 5000)
            self.assertFalse(any(record.run_id == 3 for record in fold.training))
            self.assertTrue(math.isclose(float(fold.sample_weights.mean()), 1.0, abs_tol=1e-12))


if __name__ == "__main__":
    unittest.main()
