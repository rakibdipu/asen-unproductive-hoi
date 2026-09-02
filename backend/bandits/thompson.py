"""
Gen 6: Thompson Sampling Multi-Armed Bandit
Bayesian probability matching:
Samples reward probabilities from Beta posterior distributions:
theta_k ~ Beta(alpha_k, beta_k)
Picks arm with highest sampled probability.
Updates:
If reward = 1: alpha_k = alpha_k + 1
If reward = 0: beta_k = beta_k + 1
"""

import numpy as np
from typing import Dict, List, Any
from pydantic import BaseModel


class ThompsonDecision(BaseModel):
    selected_arm: str
    sampled_probability: float
    alpha_parameters: Dict[str, float]
    beta_parameters: Dict[str, float]
    expected_means: Dict[str, float]


class ThompsonSamplingBandit:
    def __init__(self, arm_names: List[str], prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.arm_names = arm_names
        self.alphas: Dict[str, float] = {arm: prior_alpha for arm in arm_names}
        self.betas: Dict[str, float] = {arm: prior_beta for arm in arm_names}
        self.pull_counts: Dict[str, int] = {arm: 0 for arm in arm_names}
        self.reward_sums: Dict[str, float] = {arm: 0.0 for arm in arm_names}

    def select_arm(self) -> ThompsonDecision:
        samples = {}
        means = {}
        for arm in self.arm_names:
            sample = np.random.beta(self.alphas[arm], self.betas[arm])
            samples[arm] = float(sample)
            means[arm] = float(self.alphas[arm] / (self.alphas[arm] + self.betas[arm]))

        best_arm = max(samples.items(), key=lambda x: x[1])[0]

        return ThompsonDecision(
            selected_arm=best_arm,
            sampled_probability=samples[best_arm],
            alpha_parameters={k: round(v, 2) for k, v in self.alphas.items()},
            beta_parameters={k: round(v, 2) for k, v in self.betas.items()},
            expected_means={k: round(v, 4) for k, v in means.items()},
        )

    def update(self, selected_arm: str, reward: float):
        if selected_arm not in self.alphas:
            return
        if reward > 0.5:
            self.alphas[selected_arm] += 1.0
        else:
            self.betas[selected_arm] += 1.0
        self.pull_counts[selected_arm] += 1
        self.reward_sums[selected_arm] += reward
