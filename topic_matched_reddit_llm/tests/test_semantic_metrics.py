import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analyze_semantic_diversity import vendi_and_similarity, paired_bootstrap


class SemanticMetricTests(unittest.TestCase):
    def test_identical_vectors_have_minimum_normalized_vendi(self):
        x = np.ones((20, 8), dtype=float)
        result = vendi_and_similarity(x)
        self.assertAlmostEqual(result["normalized_vendi"], 1 / 20, places=10)
        self.assertAlmostEqual(result["mean_pairwise_cosine"], 1.0, places=10)

    def test_orthogonal_vectors_have_unit_normalized_vendi(self):
        x = np.eye(20, dtype=float)
        result = vendi_and_similarity(x)
        self.assertAlmostEqual(result["normalized_vendi"], 1.0, places=10)
        self.assertAlmostEqual(result["mean_pairwise_cosine"], 0.0, places=10)

    def test_bootstrap_uses_question_level_model_equal_values(self):
        rows = []
        for thread_id, base in [("a", 1.0), ("b", 3.0)]:
            for model_key in ("m1", "m2"):
                rows.append({
                    "thread_id": thread_id,
                    "llm_minus_human_vendi": base,
                    "llm_minus_human_pairwise_cosine": base + 1,
                    "llm_minus_human_early_minus_late": base + 2,
                })
        result = {row["metric"]: row for row in paired_bootstrap(rows, draws=100, seed=1)}
        self.assertAlmostEqual(result["normalized_vendi"]["estimate_llm_minus_human"], 2.0)
        self.assertEqual(result["normalized_vendi"]["n_questions"], 2)


if __name__ == "__main__":
    unittest.main()
