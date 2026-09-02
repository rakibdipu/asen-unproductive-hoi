import type {
  RecommendationResult,
  PipelineResult,
  ArenaBenchmarkResponse,
  ExplainResult,
  UserProfile,
  UserInteractionEvent,
} from "./types";

const API_BASE = "http://localhost:8000";

async function safeFetch<T>(input: RequestInfo, init?: RequestInit): Promise<T> {
  const res = await fetch(input, init);
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`API ${res.status}: ${err}`);
  }
  return res.json() as Promise<T>;
}

export async function fetchHealth() {
  return safeFetch<{ status: string }>(`${API_BASE}/api/health`);
}

export async function fetchUsers(domain: string = "movies"): Promise<UserProfile[]> {
  return safeFetch<UserProfile[]>(`${API_BASE}/api/users?domain=${domain}&limit=50`);
}

export async function fetchUserHistory(userId: string, domain: string = "movies") {
  return safeFetch(`${API_BASE}/api/users/${userId}/history?domain=${domain}`);
}

export async function fetchItems(domain: string = "movies", category?: string, query?: string) {
  let url = `${API_BASE}/api/items?domain=${domain}`;
  if (category) url += `&category=${encodeURIComponent(category)}`;
  if (query) url += `&query=${encodeURIComponent(query)}`;
  return safeFetch<any[]>(url);
}

export async function recommendSingle(
  userId: string,
  algorithm: string,
  domain: string = "movies",
  n: number = 10,
  context: Record<string, any> = {}
): Promise<RecommendationResult> {
  return safeFetch<RecommendationResult>(`${API_BASE}/api/recommend/single`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_id: userId,
      algorithm,
      domain,
      n,
      exclude_consumed: true,
      context,
    }),
  });
}

export async function recommendPipeline(
  userId: string,
  domain: string = "movies",
  context: Record<string, any> = {}
): Promise<PipelineResult> {
  return safeFetch<PipelineResult>(`${API_BASE}/api/recommend/pipeline`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, domain, context }),
  });
}

export async function runTournament(
  domain: string = "movies",
  models?: string[],
  k: number = 10,
  sampleUsers: number = 25
): Promise<ArenaBenchmarkResponse> {
  return safeFetch<ArenaBenchmarkResponse>(`${API_BASE}/api/benchmark/arena`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ domain, models, k, sample_users: sampleUsers }),
  });
}

export async function explainRecommendation(
  userId: string,
  itemId: string,
  algorithm: string = "din",
  domain: string = "movies"
): Promise<ExplainResult> {
  return safeFetch<ExplainResult>(`${API_BASE}/api/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, item_id: itemId, algorithm, domain }),
  });
}

export async function getBanditStatus() {
  return safeFetch(`${API_BASE}/api/artwork/status`);
}

export async function simulateBanditRound(
  itemId: string,
  tasteProfile: {
    name: string;
    action_affinity: number;
    scifi_affinity: number;
    romance_affinity: number;
    cinematic_affinity: number;
  },
  userClicked: boolean = true
) {
  return safeFetch(`${API_BASE}/api/artwork/simulate-round`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      item_id: itemId,
      taste_profile: tasteProfile,
      user_clicked: userClicked,
    }),
  });
}

export async function ingestEvent(event: UserInteractionEvent) {
  return safeFetch(`${API_BASE}/api/events`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(event),
  });
}
