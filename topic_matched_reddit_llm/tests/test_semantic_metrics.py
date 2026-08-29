import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analyze_semantic_diversity import (
    anchored_cosine_distances,
    cumulative_unique_unigram_counts,
    lexical_unigrams,
    paired_bootstrap,
    split_trajectory_token_ids,
    summarize_matched_trajectory,
    vendi_and_similarity,
)


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

    def test_trajectory_split_is_exactly_100_consecutive_windows(self):
        token_ids = list(range(20_000))
        windows = split_trajectory_token_ids(token_ids)
        self.assertEqual(len(windows), 100)
        self.assertTrue(all(len(window) == 200 for window in windows))
        self.assertEqual([token for window in windows for token in window], token_ids)
        self.assertEqual(windows[0], list(range(0, 200)))
        self.assertEqual(windows[-1], list(range(19_800, 20_000)))

    def test_anchor_distance_and_cosine_distance_are_correct(self):
        vectors = np.array([[2.0, 0.0], [0.0, 3.0], [-1.0, 0.0]])
        distances = anchored_cosine_distances(vectors)
        np.testing.assert_allclose(distances, [0.0, 1.0, 2.0], atol=1e-12)
        self.assertEqual(float(distances[0]), 0.0)

    def test_within_interval_distance_is_one_minus_similarity(self):
        result = vendi_and_similarity(np.eye(3, dtype=float))
        self.assertAlmostEqual(
            result["mean_pairwise_cosine_distance"],
            1.0 - result["mean_pairwise_cosine_similarity"],
        )

    def test_cumulative_lexical_diversity_uses_preprocessed_unigrams(self):
        self.assertEqual(lexical_unigrams("The café CAFÉ and dogs_123 x"), ["cafe", "cafe", "dogs"])
        counts = cumulative_unique_unigram_counts([
            "Alpha beta the", "beta gamma", "alpha delta",
        ])
        self.assertEqual(counts, [2, 3, 4])
        self.assertEqual(counts, sorted(counts))

    def test_trajectory_summary_equal_weights_models_within_question(self):
        rows = []
        for thread_id, human, llms in (("a", 0.8, (0.2, 0.4)), ("b", 0.6, (0.5, 0.7))):
            for model_index, llm in enumerate(llms, 1):
                rows.append({
                    "thread_id": thread_id,
                    "model_key": f"m{model_index}",
                    "human_mean_anchored_within_run_cosine_distance_windows_2_100": human,
                    "llm_mean_anchored_within_run_cosine_distance_windows_2_100": llm,
                })
        summary, questions = summarize_matched_trajectory(rows, draws=100, seed=1)
        by_question = {row["thread_id"]: row for row in questions}
        self.assertAlmostEqual(by_question["a"]["model_equal_llm_mean"], 0.3)
        self.assertAlmostEqual(by_question["a"]["human_minus_llm"], 0.5)
        self.assertAlmostEqual(summary[0]["human_mean"], 0.7)
        self.assertAlmostEqual(summary[0]["model_equal_llm_mean"], 0.45)
        self.assertAlmostEqual(summary[0]["human_minus_llm"], 0.25)


if __name__ == "__main__":
    unittest.main()
