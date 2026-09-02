"""
Users API Endpoints
Provides listing of user profiles, personas, and historical interactions.
"""

from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from backend.data.loaders import dataset_manager

router = APIRouter(prefix="/api/users", tags=["Users"])


@router.get("")
def list_users(domain: str = "movies", limit: int = 50) -> List[Dict[str, Any]]:
    dataset = dataset_manager.load_domain(domain)
    return dataset.users_df.head(limit).to_dict(orient="records")


@router.get("/{user_id}")
def get_user(user_id: str, domain: str = "movies") -> Dict[str, Any]:
    dataset = dataset_manager.load_domain(domain)
    user_row = dataset.users_df[dataset.users_df["user_id"] == user_id]
    if user_row.empty:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found.")
    return user_row.iloc[0].to_dict()


@router.get("/{user_id}/history")
def get_user_history(user_id: str, domain: str = "movies", limit: int = 20) -> List[Dict[str, Any]]:
    dataset = dataset_manager.load_domain(domain)
    history = dataset.get_user_history(user_id).tail(limit)
    item_lookup = dataset.get_item_lookup()

    records = []
    for _, row in history.iterrows():
        rec = row.to_dict()
        iid = str(rec["item_id"])
        if iid in item_lookup:
            meta = item_lookup[iid]
            rec["title"] = meta.get("title", f"Item {iid}")
            rec["category"] = meta.get("primary_genre", meta.get("genre", meta.get("category", "")))
            rec["thumbnail_url"] = meta.get("thumbnail_url")
        records.append(rec)

    return records
