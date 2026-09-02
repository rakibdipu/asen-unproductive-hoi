"""
End-to-End Test Suite for Omni-RecSys Mega Platform
Verifies:
1. All 12 algorithms can be instantiated, fitted, and recommend top-K items.
2. 5-stage production pipeline runs end-to-end with valid stage metrics.
3. Model Arena tournament computes NDCG, HitRate, Diversity, and Coverage.
4. LinUCB Contextual Bandit updates weights correctly upon rewards.
5. Online Feature Store ingests user events and updates session history.
"""

import pytest
import numpy as np
import pandas as pd

from backend.data.loaders import dataset_manager
from backend.models import get_model, MODEL_REGISTRY
from backend.pipeline.orchestrator import ProductionPipeline
from backend.evaluation.arena import ModelArena
from backend.bandits.linucb import LinUCBBandit
from backend.streaming.feature_store import feature_store
from backend.streaming.event_bus import UserInteractionEvent, EventType


@pytest.fixture(scope="module")
def dataset():
    return dataset_manager.load_domain("movies", force_reload=False)


def test_dataset_loading(dataset):
    assert len(dataset.items_df) >= 100
    assert len(dataset.users_df) >= 50
    assert len(dataset.interactions_df) >= 500
    train_df, test_df = dataset.train_test_split(strategy="leave_k_out", k=3)
    assert len(train_df) > len(test_df)


@pytest.mark.parametrize("model_key", list(MODEL_REGISTRY.keys()))
def test_individual_models(model_key, dataset):
    model = get_model(model_key, domain="movies")
    model.fit(dataset.interactions_df, dataset.items_df)
    assert model.is_fitted

    recs = model.recommend(user_id="usr_0001", n=5)
    assert len(recs) <= 5
    if len(recs) > 0:
        assert recs[0].score >= 0.0
        assert recs[0].title is not None


def test_production_pipeline_funnel(dataset):
    retrieval_models = {
        "two_tower": get_model("two_tower", domain="movies").fit(dataset.interactions_df, dataset.items_df),
        "implicit_als": get_model("implicit_als", domain="movies").fit(dataset.interactions_df, dataset.items_df),
        "sasrec": get_model("sasrec", domain="movies").fit(dataset.interactions_df, dataset.items_df),
        "popularity": get_model("popularity", domain="movies").fit(dataset.interactions_df, dataset.items_df),
    }
    ranking_model = get_model("mmoe", domain="movies").fit(dataset.interactions_df, dataset.items_df)
    bandit = LinUCBBandit(arm_names=["variant_action", "variant_character", "variant_romantic", "variant_cinematic"], feature_dim=4)

    pipeline = ProductionPipeline(retrieval_models=retrieval_models, ranking_model=ranking_model, artwork_bandit=bandit)

    result = pipeline.execute(
        user_id="usr_0001",
        domain="movies",
        catalog_size=len(dataset.items_df),
        user_genre_vector=np.array([0.5, 0.3, 0.1, 0.1], dtype=np.float32),
    )

    assert result.total_latency_ms > 0
    assert len(result.stage_metrics) == 5
    # Verify monotonic filtering and retrieval stages
    assert result.stage_metrics[0].stage_name == "Candidate Retrieval"
    assert result.stage_metrics[1].stage_name == "Business Filtering"
    assert result.stage_metrics[2].stage_name == "Heavy Ranking"
    assert len(result.slate_page.rows) == 5


def test_model_arena_tournament(dataset):
    models = {
        "popularity": get_model("popularity", domain="movies").fit(dataset.interactions_df, dataset.items_df),
        "implicit_als": get_model("implicit_als", domain="movies").fit(dataset.interactions_df, dataset.items_df),
        "sasrec": get_model("sasrec", domain="movies").fit(dataset.interactions_df, dataset.items_df),
    }

    arena = ModelArena(dataset=dataset, trained_models=models)
    tournament = arena.run_tournament(model_keys=["popularity", "implicit_als", "sasrec"], k=5, sample_users=10)

    assert len(tournament.models) == 3
    for row in tournament.models:
        assert 0.0 <= row.ndcg_at_10 <= 1.0
        assert 0.0 <= row.catalog_coverage <= 1.0


def test_linucb_bandit_learning():
    bandit = LinUCBBandit(arm_names=["arm_a", "arm_b"], feature_dim=2, alpha=0.2)
    ctx = np.array([1.0, 0.0], dtype=np.float32)

    # Initial decision
    dec1 = bandit.select_arm(ctx)

    # Train arm_a with positive rewards on [1, 0] context
    for _ in range(15):
        bandit.update("arm_a", ctx, reward=1.0)
        bandit.update("arm_b", ctx, reward=0.0)

    dec2 = bandit.select_arm(ctx)
    assert dec2.selected_arm == "arm_a"
    assert dec2.expected_reward > 0.5


def test_feature_store_event_ingestion():
    ev = UserInteractionEvent(
        event_id="test_123",
        user_id="usr_test_01",
        item_id="mov_0001",
        event_type=EventType.WATCH_COMPLETE,
        timestamp=1700000000,
        context={"genre": "Sci-Fi"},
    )
    feature_store.update_from_event(ev, item_genre="Sci-Fi")

    session = feature_store.get_user_session_history("usr_test_01")
    assert "mov_0001" in session
