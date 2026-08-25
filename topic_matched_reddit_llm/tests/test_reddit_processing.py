import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sample_reddit_discussions import clean_and_sort_comments, fixed_seed_sample


class RedditProcessingTests(unittest.TestCase):
    def test_cleaner_filters_and_orders_comments(self):
        thread = {
            "post": {"id": "abc"},
            "comments": [
                {"id": "3", "body": "later", "created_utc": 30, "author": "user", "link_id": "t3_abc"},
                {"id": "1", "body": "[deleted]", "created_utc": 10, "author": "user", "link_id": "t3_abc"},
                {"id": "2", "body": "bot", "created_utc": 15, "author": "AutoModerator", "link_id": "t3_abc"},
                {"id": "4", "body": "earlier", "created_utc": 20, "author": "user", "link_id": "t3_abc"},
            ],
        }
        self.assertEqual(clean_and_sort_comments(thread), ["earlier", "later"])

    def test_fixed_seed_sampling_is_reproducible(self):
        rows = [{"thread_id": str(i)} for i in range(10)]
        first = fixed_seed_sample(rows, n=4, seed=123)
        second = fixed_seed_sample(list(reversed(rows)), n=4, seed=123)
        self.assertEqual([row["thread_id"] for row in first], [row["thread_id"] for row in second])


if __name__ == "__main__":
    unittest.main()
