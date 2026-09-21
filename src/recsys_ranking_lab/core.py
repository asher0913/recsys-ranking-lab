from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
from statistics import mean


def embed(text: str, dimensions: int = 24) -> list[float]:
    vector = [0.0] * dimensions
    for token in text.lower().split():
        digest = hashlib.blake2b(token.encode(), digest_size=16).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1 if digest[4] % 2 == 0 else -1
        vector[index] += sign * (1 + digest[5] / 255)
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass(frozen=True)
class Item:
    item_id: str
    category: str
    text: str
    popularity: float

    @property
    def vector(self) -> list[float]:
        return embed(f"{self.category} {self.text}")


@dataclass(frozen=True)
class Event:
    item_id: str
    action: str
    age_hours: float


class RecommendationEngine:
    ACTION_WEIGHT = {"view": 1.0, "save": 2.0, "cart": 3.0, "buy": 4.0}

    def __init__(self, catalog: list[Item]) -> None:
        self.catalog = {item.item_id: item for item in catalog}

    def user_vector(self, events: list[Event], long_term: bool = True) -> list[float]:
        selected = events if long_term else events[-5:]
        result = [0.0] * 24
        total = 0.0
        for event in selected:
            item = self.catalog[event.item_id]
            recency = math.exp(-event.age_hours / (24 * (30 if long_term else 3)))
            weight = self.ACTION_WEIGHT[event.action] * recency
            total += weight
            for index, value in enumerate(item.vector):
                result[index] += weight * value
        norm = math.sqrt(sum(value * value for value in result)) or total or 1.0
        return [value / norm for value in result]

    def retrieve(self, events: list[Event], k: int = 20) -> list[tuple[str, float]]:
        short = self.user_vector(events, long_term=False)
        long = self.user_vector(events, long_term=True)
        seen = {event.item_id for event in events}
        scores = []
        for item in self.catalog.values():
            if item.item_id in seen:
                continue
            score = 0.65 * cosine(short, item.vector) + 0.35 * cosine(long, item.vector)
            scores.append((item.item_id, score))
        return sorted(scores, key=lambda pair: pair[1], reverse=True)[:k]

    def rank(self, events: list[Event], candidates: list[tuple[str, float]], k: int = 10) -> list[str]:
        categories = [self.catalog[event.item_id].category for event in events]
        favorite = max(set(categories), key=categories.count)
        scored = []
        for item_id, retrieval in candidates:
            item = self.catalog[item_id]
            category_match = 1.0 if item.category == favorite else 0.0
            coarse = 0.75 * retrieval + 0.25 * item.popularity
            fine = 0.70 * coarse + 0.30 * category_match
            scored.append((item_id, fine))
        return [item_id for item_id, _ in sorted(scored, key=lambda pair: pair[1], reverse=True)[:k]]

    def recommend(self, events: list[Event], k: int = 10) -> list[str]:
        return self.rank(events, self.retrieve(events, max(k * 3, 20)), k)

    def hard_negatives(self, events: list[Event], positives: set[str], k: int = 5) -> list[str]:
        return [item for item, _ in self.retrieve(events, 50) if item not in positives][:k]


def recall_at_k(ranked: list[str], truth: str, k: int) -> float:
    return float(truth in ranked[:k])


def ndcg_at_k(ranked: list[str], truth: str, k: int) -> float:
    if truth not in ranked[:k]:
        return 0.0
    return 1 / math.log2(ranked.index(truth) + 2)


def mrr(ranked: list[str], truth: str) -> float:
    return 1 / (ranked.index(truth) + 1) if truth in ranked else 0.0


def fixture(seed: int = 11) -> tuple[RecommendationEngine, list[tuple[list[Event], str]]]:
    rng = random.Random(seed)
    categories = ["vision", "agents", "systems", "security"]
    catalog = [Item(f"i{i}", categories[i % 4], f"{categories[i % 4]} topic-{i % 7}", rng.random()) for i in range(80)]
    engine = RecommendationEngine(catalog)
    sessions = []
    for user in range(24):
        category = categories[user % 4]
        matches = [item for item in catalog if item.category == category]
        history = [Event(item.item_id, "save" if index % 3 == 0 else "view", (8 - index) * 12) for index, item in enumerate(matches[:8])]
        sessions.append((history[:-1], history[-1].item_id))
    return engine, sessions


def demo() -> dict[str, object]:
    engine, sessions = fixture()
    ranked = [(engine.recommend(events, 10), truth) for events, truth in sessions]
    return {
        "users": len(sessions),
        "recall_at_10": mean(recall_at_k(items, truth, 10) for items, truth in ranked),
        "ndcg_at_10": mean(ndcg_at_k(items, truth, 10) for items, truth in ranked),
        "mrr": mean(mrr(items, truth) for items, truth in ranked),
    }


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2, sort_keys=True))
