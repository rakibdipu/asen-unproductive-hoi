export interface ItemScore {
  item_id: string;
  score: number;
  rank: number;
  title: string;
  category?: string;
  thumbnail_url?: string;
  artwork_variant_id?: string;
  source_model?: string;
  metadata: Record<string, any>;
}

export interface StageMetric {
  stage_name: string;
  stage_number: number;
  input_count: number;
  output_count: number;
  latency_ms: number;
  description: string;
}

export interface SlateRow {
  row_id: string;
  title: string;
  description: string;
  items: ItemScore[];
}

export interface BanditDecision {
  item_id: string;
  selected_arm: string;
  expected_reward: number;
  confidence_bound: number;
  total_score: number;
  context_vector: number[];
  arm_statistics: Record<string, {
    expected_reward: number;
    confidence_bound: number;
    total_ucb: number;
    pull_count: number;
    avg_reward: number;
  }>;
}

export interface SlatePage {
  user_id: string;
  domain: string;
  rows: SlateRow[];
  artwork_decisions: Record<string, BanditDecision>;
}

export interface PipelineResult {
  user_id: string;
  domain: string;
  total_latency_ms: number;
  stage_metrics: StageMetric[];
  slate_page: SlatePage;
  final_items: ItemScore[];
}

export interface RecommendationResult {
  user_id: string;
  domain: string;
  algorithm: string;
  total_candidates_evaluated: number;
  latency_ms: number;
  items: ItemScore[];
}

export interface ModelBenchmarkRow {
  model_key: string;
  model_name: string;
  generation: string;
  company: string;
  year: number;
  ndcg_at_10: number;
  hit_rate_at_10: number;
  map_at_10: number;
  mrr_at_10: number;
  catalog_coverage: number;
  diversity_ils: number;
  novelty_score: number;
  avg_latency_ms: number;
}

export interface ArenaBenchmarkResponse {
  domain: string;
  num_test_users: number;
  num_catalog_items: number;
  models: ModelBenchmarkRow[];
  summary: string;
}

export interface ExplainResult {
  user_id: string;
  item_id: string;
  item_title: string;
  score: number;
  algorithm: string;
  top_contributing_interactions: Array<{
    item_id: string;
    title: string;
    attention_weight?: number;
    similarity?: number;
    genre?: string;
  }>;
  attention_weights: number[];
  natural_language_explanation: string;
  feature_importance: Record<string, number>;
}

export interface UserProfile {
  user_id: string;
  username: string;
  persona: string;
  age: number;
  fav_genres: string | string[];
  fav_music: string | string[];
  fav_ecom: string | string[];
  activity_level: string;
}

export interface UserInteractionEvent {
  event_id: string;
  user_id: string;
  item_id: string;
  event_type: string;
  timestamp: number;
  watch_ratio?: number;
  rating?: number;
  artwork_variant_id?: string;
  context?: Record<string, any>;
}
