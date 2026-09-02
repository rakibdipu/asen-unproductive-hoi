"""
Unified Multi-Domain Dataset Loader & Splitter
Supports Movies (Netflix), Music (Spotify), E-Commerce (Amazon), and Short Feed (TikTok)
Uses JSON-based disk caching for universal zero-dependency compatibility.
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import pandas as pd
from backend.config import settings
from backend.data.synthetic import SyntheticDataGenerator


class RecDataset:
    """Encapsulates items, users, and interaction data with train/test splitting."""

    def __init__(
        self,
        domain: str,
        items_df: pd.DataFrame,
        users_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
    ):
        self.domain = domain
        self.items_df = items_df
        self.users_df = users_df
        self.interactions_df = interactions_df

    def train_test_split(
        self, test_ratio: float = 0.2, strategy: str = "leave_k_out", k: int = 3
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits interactions into train and test sets.
        Strategy 'leave_k_out': Leaves the last K interactions per user for testing (temporal split).
        Strategy 'random': Random sampling across all interactions.
        """
        df = self.interactions_df.sort_values(["user_id", "timestamp"]).copy()

        if strategy == "leave_k_out":
            test_mask = df.groupby("user_id").cumcount(ascending=False) < k
            test_df = df[test_mask].copy()
            train_df = df[~test_mask].copy()
        else:
            shuffled = df.sample(frac=1.0, random_state=42)
            n_test = int(len(shuffled) * test_ratio)
            test_df = shuffled.iloc[:n_test].copy()
            train_df = shuffled.iloc[n_test:].copy()

        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def get_user_history(self, user_id: str) -> pd.DataFrame:
        """Returns ordered history for a specific user."""
        return self.interactions_df[self.interactions_df["user_id"] == user_id].sort_values("timestamp")

    def get_item_lookup(self) -> Dict[str, Dict[str, Any]]:
        """Dictionary lookup mapping item_id to metadata dictionary."""
        return self.items_df.set_index("item_id").to_dict(orient="index")


class DatasetManager:
    """Manages loading, caching, and serving datasets across all domains."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or settings.DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.generator = SyntheticDataGenerator(seed=42)
        self._cache: Dict[str, RecDataset] = {}

    def load_domain(self, domain: str = "movies", force_reload: bool = False) -> RecDataset:
        """Loads or generates dataset for specified domain."""
        if not force_reload and domain in self._cache:
            return self._cache[domain]

        domain_path = self.data_dir / domain
        domain_path.mkdir(parents=True, exist_ok=True)

        items_file = domain_path / "items.json"
        users_file = domain_path / "users.json"
        interactions_file = domain_path / "interactions.json"

        if items_file.exists() and users_file.exists() and interactions_file.exists() and not force_reload:
            items_df = pd.read_json(items_file)
            users_df = pd.read_json(users_file)
            interactions_df = pd.read_json(interactions_file)
        else:
            # Generate domain-specific catalog
            if domain == "movies":
                items_df = self.generator.generate_movies_catalog(num_items=120)
            elif domain == "music":
                items_df = self.generator.generate_music_catalog(num_items=120)
            elif domain == "ecommerce":
                items_df = self.generator.generate_ecommerce_catalog(num_items=120)
            elif domain == "short_feed":
                items_df = self.generator.generate_short_feed_catalog(num_items=120)
            else:
                items_df = self.generator.generate_movies_catalog(num_items=120)

            users_df = self.generator.generate_users(num_users=60)
            interactions_df = self.generator.generate_interactions(users_df, items_df, domain=domain)

            # Persist to disk as JSON
            items_df.to_json(items_file, orient="records", indent=2)
            users_df.to_json(users_file, orient="records", indent=2)
            interactions_df.to_json(interactions_file, orient="records", indent=2)

        dataset = RecDataset(domain, items_df, users_df, interactions_df)
        self._cache[domain] = dataset
        return dataset


dataset_manager = DatasetManager()
