"""
Comprehensive Evaluation Metrics Suite for RecSys
Computes NDCG@K, HitRate@K, MAP@K, MRR, Coverage, Diversity, Novelty & Latency
"""

import math
import numpy as np
from typing import List, Dict, Set, Any, Optional
from collections import defaultdict


def dcg_at_k(r: List[float], k: int) -> float:
    """Discounted Cumulative Gain at rank K."""
    r = np.asarray(r, dtype=np.float64)[:k]
    if r.size:
        return float(np.sum(r / np.log2(np.arange(2, r.size + 2))))
    return 0.0


def ndcg_at_k(r: List[float], k: int) -> float:
    """Normalized Discounted Cumulative Gain at rank K."""
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k) / dcg_max


def hit_rate_at_k(recommended: List[str], ground_truth: Set[str], k: int) -> float:
    """Hit Rate @ K: 1.0 if at least one ground-truth item in top K, else 0.0."""
    top_k = recommended[:k]
    return 1.0 if any(item in ground_truth for item in top_k) else 0.0


def precision_at_k(recommended: List[str], ground_truth: Set[str], k: int) -> float:
    """Precision @ K."""
    if k == 0:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in ground_truth)
    return hits / float(k)


def recall_at_k(recommended: List[str], ground_truth: Set[str], k: int) -> float:
    """Recall @ K."""
    if not ground_truth:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in ground_truth)
    return hits / float(len(ground_truth))


def average_precision(recommended: List[str], ground_truth: Set[str], k: int) -> float:
    """Average Precision @ K."""
    if not ground_truth:
        return 0.0
    hits = 0
    sum_precisions = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in ground_truth:
            hits += 1
            sum_precisions += hits / (i + 1.0)
    return sum_precisions / min(len(ground_truth), k) if ground_truth else 0.0


def reciprocal_rank(recommended: List[str], ground_truth: Set[str], k: int) -> float:
    """Reciprocal Rank @ K: 1 / rank of first relevant item."""
    for i, item in enumerate(recommended[:k]):
        if item in ground_truth:
            return 1.0 / (i + 1.0)
    return 0.0


def intra_list_diversity(
    recommended: List[str], item_embeddings: Dict[str, np.ndarray], k: int
) -> float:
    """
    Intra-List Diversity (ILS): Average pairwise distance between recommended items.
    Diversity = 1 - Mean Cosine Similarity
    """
    items = [item for item in recommended[:k] if item in item_embeddings]
    n = len(items)
    if n <= 1:
        return 0.0

    total_sim = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            vec_a = item_embeddings[items[i]]
            vec_b = item_embeddings[items[j]]
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)
            if norm_a > 0 and norm_b > 0:
                sim = np.dot(vec_a, vec_b) / (norm_a * norm_b)
                total_sim += sim
                count += 1

    if count == 0:
        return 0.5  # fallback
    mean_sim = total_sim / count
    return float(max(0.0, min(1.0, 1.0 - mean_sim)))


def catalog_coverage(all_recommended: List[List[str]], all_catalog_items: Set[str], k: int) -> float:
    """Catalog Coverage: Fraction of all catalog items recommended at least once."""
    if not all_catalog_items:
        return 0.0
    recommended_set = set()
    for rec_list in all_recommended:
        recommended_set.update(rec_list[:k])
    return len(recommended_set) / float(len(all_catalog_items))


def novelty_score(recommended: List[str], item_popularity: Dict[str, int], total_interactions: int, k: int) -> float:
    """
    Novelty: Average self-information of recommended items.
    Higher means system recommends less common / long-tail items.
    Novelty = - sum(log2(p(i))) / K
    """
    top_k = recommended[:k]
    if not top_k or total_interactions == 0:
        return 0.0

    scores = []
    for item in top_k:
        pop = item_popularity.get(item, 1)
        prob = pop / float(total_interactions)
        scores.append(-math.log2(prob))
    return float(np.mean(scores))


class RecSysEvaluator:
    """
    Standard evaluation suite runner across multiple users and ground-truth sets.
    """

    def __init__(
        self,
        ground_truth: Dict[str, Set[str]],
        catalog_items: Set[str],
        item_popularity: Dict[str, int],
        total_interactions: int,
        item_embeddings: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.ground_truth = ground_truth
        self.catalog_items = catalog_items
        self.item_popularity = item_popularity
        self.total_interactions = total_interactions
        self.item_embeddings = item_embeddings or {}

    def evaluate(self, recommendations: Dict[str, List[str]], k: int = 10) -> Dict[str, float]:
        """
        Calculates all key metrics for a given model's recommendations.
        """
        ndcgs = []
        hit_rates = []
        maps = []
        mrrs = []
        precisions = []
        recalls = []
        diversities = []
        novelties = []
        all_recs = []

        for uid, target_items in self.ground_truth.items():
            if not target_items:
                continue
            recs = recommendations.get(uid, [])
            all_recs.append(recs)

            # Relevance indicators (1.0 if in target, else 0.0)
            rel_vector = [1.0 if item in target_items else 0.0 for item in recs[:k]]
            ndcgs.append(ndcg_at_k(rel_vector, k))
            hit_rates.append(hit_rate_at_k(recs, target_items, k))
            maps.append(average_precision(recs, target_items, k))
            mrrs.append(reciprocal_rank(recs, target_items, k))
            precisions.append(precision_at_k(recs, target_items, k))
            recalls.append(recall_at_k(recs, target_items, k))

            if self.item_embeddings:
                diversities.append(intra_list_diversity(recs, self.item_embeddings, k))

            novelties.append(novelty_score(recs, self.item_popularity, self.total_interactions, k))

        coverage = catalog_coverage(all_recs, self.catalog_items, k)

        return {
            f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
            f"hit_rate@{k}": float(np.mean(hit_rates)) if hit_rates else 0.0,
            f"map@{k}": float(np.mean(maps)) if maps else 0.0,
            f"mrr@{k}": float(np.mean(mrrs)) if mrrs else 0.0,
            f"precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
            f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
            "catalog_coverage": float(coverage),
            "diversity_ils": float(np.mean(diversities)) if diversities else 0.5,
            "novelty_score": float(np.mean(novelties)) if novelties else 0.0,
        }
