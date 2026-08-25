import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import multiagent_simulation as sim
from run_topic_matched_simulations import inject_seed_at_round_one


class ProtocolTests(unittest.TestCase):
    def test_topic_free_infinite_space_sentence_is_absent(self):
        source = (ROOT / "multiagent_simulation.py").read_text(encoding="utf-8")
        self.assertNotIn("You exist in an infinite space with no constraints or rules.", source)
        self.assertIn("Take any action you want, along or with others.", source)
        self.assertIn("Describe what you do next.", source)

    def test_seed_is_only_explicitly_inserted_at_round_one(self):
        seed = "Example Reddit question?"
        base = [sim.SystemMessage(content="system"), sim.UserMessage(content="Describe what you do next.")]
        round_one = inject_seed_at_round_one(list(base), current_round=1, seed_text=seed)
        later = inject_seed_at_round_one(list(base), current_round=2, seed_text=seed)
        self.assertEqual([m.content for m in round_one], ["system", seed, "Describe what you do next."])
        self.assertEqual([m.content for m in later], ["system", "Describe what you do next."])


if __name__ == "__main__":
    unittest.main()
