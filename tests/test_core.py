import unittest

from recsys_ranking_lab.core import Event, Item, RecommendationEngine, embed, mrr, ndcg_at_k, recall_at_k


class RecommenderTests(unittest.TestCase):
    def setUp(self):
        self.items = [
            Item("a", "vision", "camera model", 0.4),
            Item("b", "vision", "image model", 0.5),
            Item("c", "systems", "cache runtime", 0.9),
        ]
        self.engine = RecommendationEngine(self.items)

    def test_embedding_is_deterministic(self):
        self.assertEqual(embed("same text"), embed("same text"))

    def test_seen_items_are_filtered(self):
        ranked = self.engine.recommend([Event("a", "view", 1)], 3)
        self.assertNotIn("a", ranked)

    def test_metrics(self):
        ranked = ["a", "b", "c"]
        self.assertEqual(recall_at_k(ranked, "b", 2), 1.0)
        self.assertAlmostEqual(mrr(ranked, "b"), 0.5)
        self.assertGreater(ndcg_at_k(ranked, "b", 2), 0)

    def test_hard_negatives_exclude_positive(self):
        events = [Event("a", "save", 1)]
        negatives = self.engine.hard_negatives(events, {"b"}, 2)
        self.assertNotIn("b", negatives)


if __name__ == "__main__":
    unittest.main()
