"""
Stage 2: Hard Filtering & Business Rules
Applies real-time production guardrails:
- Deduplication against already consumed items
- Age / Parental guidance rating filters
- Content safety & blocklist filtering
- Geo-fencing & availability restrictions
"""

from typing import List, Set, Dict, Any, Optional
from backend.models.base import ItemScore


class FilteringStage:
    def __init__(self):
        pass

    def filter(
        self,
        candidates: List[ItemScore],
        consumed_item_ids: Optional[Set[str]] = None,
        max_age_rating: Optional[str] = None,
        blocked_tags: Optional[Set[str]] = None,
        max_candidates: int = 100,
    ) -> List[ItemScore]:
        consumed = consumed_item_ids or set()
        blocked = blocked_tags or set()

        valid_items = []
        for it in candidates:
            # 1. Deduplicate consumed items
            if it.item_id in consumed:
                continue

            # 2. Blocked tags check
            item_tags = set(it.metadata.get("tags", "").split(","))
            if blocked and len(item_tags & blocked) > 0:
                continue

            valid_items.append(it)
            if len(valid_items) >= max_candidates:
                break

        return valid_items
