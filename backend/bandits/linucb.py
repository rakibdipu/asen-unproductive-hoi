"""
Gen 6: LinUCB Contextual Bandit (Netflix Artwork Personalization, Amat Gil et al. 2018)
Dynamically chooses the optimal thumbnail/artwork variant for each movie/item
based on user context features (e.g. genre affinities, device, time).
Formula:
p_{t, a} = theta_a^T x_{t, a} + alpha * sqrt(x_{t, a}^T A_a^{-1} x_{t, a})
where A_a = D_a^T D_a + I_d, and b_a = D_a^T c_a
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pydantic import BaseModel


class BanditDecision(BaseModel):
    item_id: str
    selected_arm: str
    expected_reward: float
    confidence_bound: float
    total_score: float
    context_vector: List[float]
    arm_statistics: Dict[str, Dict[str, float]]


class LinUCBBandit:
    """
    Disjoint Linear Upper Confidence Bound (LinUCB) Bandit.
    Used for Netflix-style dynamic Artwork/Thumbnail selection and item exploration.
    """

    def __init__(self, arm_names: List[str], feature_dim: int = 5, alpha: float = 0.25):
        self.arm_names = arm_names
        self.feature_dim = feature_dim
        self.alpha = alpha  # Exploration control parameter

        # For each arm a:
        # A_a: [d x d] covariance matrix (init as Identity)
        # b_a: [d x 1] reward response vector (init as zero)
        self.A: Dict[str, np.ndarray] = {arm: np.eye(feature_dim, dtype=np.float32) for arm in arm_names}
        self.b: Dict[str, np.ndarray] = {arm: np.zeros((feature_dim, 1), dtype=np.float32) for arm in arm_names}
        self.pull_counts: Dict[str, int] = {arm: 0 for arm in arm_names}
        self.reward_sums: Dict[str, float] = {arm: 0.0 for arm in arm_names}

    def select_arm(self, context_vec: np.ndarray, item_id: str = "") -> BanditDecision:
        """
        Selects the arm with the highest UCB score given context vector x.
        context_vec: [feature_dim] (e.g., user genre preference proportions)
        """
        x = context_vec.reshape((self.feature_dim, 1)).astype(np.float32)

        best_arm = self.arm_names[0]
        highest_score = -float("inf")
        best_expected = 0.0
        best_cb = 0.0
        arm_stats = {}

        for arm in self.arm_names:
            A_inv = np.linalg.inv(self.A[arm])
            theta = np.dot(A_inv, self.b[arm])  # Ridge regression weights [d x 1]

            # Exploitation: Expected reward theta^T * x
            expected_r = float(np.dot(theta.T, x)[0, 0])

            # Exploration: UCB confidence interval alpha * sqrt(x^T A_inv x)
            var = float(np.dot(np.dot(x.T, A_inv), x)[0, 0])
            confidence_bound = self.alpha * np.sqrt(max(1e-8, var))

            score = expected_r + confidence_bound

            arm_stats[arm] = {
                "expected_reward": round(expected_r, 4),
                "confidence_bound": round(confidence_bound, 4),
                "total_ucb": round(score, 4),
                "pull_count": self.pull_counts[arm],
                "avg_reward": round(self.reward_sums[arm] / max(1, self.pull_counts[arm]), 3),
            }

            if score > highest_score:
                highest_score = score
                best_arm = arm
                best_expected = expected_r
                best_cb = confidence_bound

        return BanditDecision(
            item_id=item_id,
            selected_arm=best_arm,
            expected_reward=best_expected,
            confidence_bound=best_cb,
            total_score=highest_score,
            context_vector=context_vec.tolist(),
            arm_statistics=arm_stats,
        )

    def update(self, selected_arm: str, context_vec: np.ndarray, reward: float):
        """
        Online Bayesian update upon observing user click/conversion:
        A_a = A_a + x * x^T
        b_a = b_a + r * x
        """
        if selected_arm not in self.A:
            return

        x = context_vec.reshape((self.feature_dim, 1)).astype(np.float32)
        self.A[selected_arm] += np.dot(x, x.T)
        self.b[selected_arm] += reward * x
        self.pull_counts[selected_arm] += 1
        self.reward_sums[selected_arm] += reward
