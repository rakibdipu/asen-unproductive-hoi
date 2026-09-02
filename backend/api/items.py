"""
Items Catalog API Endpoints
Provides catalog browsing, category filtering, item metadata, and artwork variants.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from backend.data.loaders import dataset_manager

router = APIRouter(prefix="/api/items", tags=["Items"])


@router.get("")
def list_items(
    domain: str = "movies",
    category: Optional[str] = None,
    query: Optional[str] = None,
    limit: int = 60,
) -> List[Dict[str, Any]]:
    dataset = dataset_manager.load_domain(domain)
    df = dataset.items_df

    if category:
        genre_col = "primary_genre" if "primary_genre" in df.columns else ("genre" if "genre" in df.columns else "category")
        df = df[df[genre_col].str.lower() == category.lower()]

    if query:
        df = df[df["title"].str.lower().str.contains(query.lower())]

    return df.head(limit).to_dict(orient="records")


@router.get("/{item_id}")
def get_item(item_id: str, domain: str = "movies") -> Dict[str, Any]:
    dataset = dataset_manager.load_domain(domain)
    item_row = dataset.items_df[dataset.items_df["item_id"] == item_id]
    if item_row.empty:
        raise HTTPException(status_code=404, detail=f"Item '{item_id}' not found.")
    return item_row.iloc[0].to_dict()
