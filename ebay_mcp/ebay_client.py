"""eBay Browse API client. Standalone Python, no MCP dependency.

Forked from Suat Oneren's ebay-browse-mcp. The client itself is essentially
upstream's — the fork adds an HTTP transport and container packaging on top
(see ``server.py``). Kept framework-free so it can also be driven from the CLI
(`python -m ebay_mcp.ebay_client "<query>"`) for debugging inside the image.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import requests
from dotenv import load_dotenv

# Load a local .env if present (developer convenience). In the container the
# real values arrive as environment variables via compose `env_file:`, and
# python-dotenv never overrides an already-set env var, so this is a no-op there.
load_dotenv()

CLIENT_ID = os.environ.get("EBAY_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("EBAY_CLIENT_SECRET", "")
MARKETPLACE = os.environ.get("EBAY_MARKETPLACE", "EBAY_DE")
ENV = os.environ.get("EBAY_ENV", "sandbox").lower()

if ENV == "production":
    BASE = "https://api.ebay.com"
else:
    BASE = "https://api.sandbox.ebay.com"

TOKEN_URL = f"{BASE}/identity/v1/oauth2/token"
SEARCH_URL = f"{BASE}/buy/browse/v1/item_summary/search"
ITEM_URL = f"{BASE}/buy/browse/v1/item"

# In-memory token cache (process-local).
_token_cache: dict[str, object] = {"value": None, "expires_at": 0.0}


def credentials_configured() -> bool:
    """Whether a keyset is present. Used by the container readiness probe."""
    return bool(CLIENT_ID and CLIENT_SECRET)


def _get_access_token() -> str:
    """Fetch an OAuth application access token via client-credentials flow.

    Cached in memory and reused until 60s before expiry.
    """
    if not credentials_configured():
        raise RuntimeError(
            "EBAY_CLIENT_ID and EBAY_CLIENT_SECRET must be set (env or .env)"
        )

    now = time.time()
    if _token_cache["value"] and now < _token_cache["expires_at"]:  # type: ignore[operator]
        return _token_cache["value"]  # type: ignore[return-value]

    response = requests.post(
        TOKEN_URL,
        auth=(CLIENT_ID, CLIENT_SECRET),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "client_credentials",
            "scope": "https://api.ebay.com/oauth/api_scope",
        },
        timeout=10,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Token request failed ({response.status_code}): {response.text[:300]}"
        )
    payload = response.json()
    _token_cache["value"] = payload["access_token"]
    # 60s safety margin before actual expiry.
    _token_cache["expires_at"] = now + payload["expires_in"] - 60
    return _token_cache["value"]  # type: ignore[return-value]


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "X-EBAY-C-MARKETPLACE-ID": MARKETPLACE,
        "Content-Type": "application/json",
    }


def search_items(
    query: str,
    limit: int = 10,
    filter_expr: Optional[str] = None,
    sort: Optional[str] = None,
) -> dict:
    """Search eBay listings via the Browse API.

    Returns the raw Browse API response dict (itemSummaries, total, etc).
    """
    params: dict[str, object] = {"q": query, "limit": max(1, min(limit, 200))}
    if filter_expr:
        params["filter"] = filter_expr
    if sort:
        params["sort"] = sort

    token = _get_access_token()
    response = requests.get(
        SEARCH_URL, headers=_headers(token), params=params, timeout=15
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Search failed ({response.status_code}): {response.text[:500]}"
        )
    return response.json()


def get_item(item_id: str, fieldgroups: Optional[str] = None) -> dict:
    """Retrieve full details of a specific eBay item via the Browse API.

    ``item_id`` is the REST identifier from search (format ``v1|<numeric>|0``);
    legacy numeric ids alone are not accepted by this endpoint.
    """
    # item_id contains "|" which must be URL-encoded into the path.
    url = f"{ITEM_URL}/{requests.utils.quote(item_id, safe='')}"
    params = {}
    if fieldgroups:
        params["fieldgroups"] = fieldgroups

    token = _get_access_token()
    response = requests.get(url, headers=_headers(token), params=params, timeout=15)
    if response.status_code != 200:
        raise RuntimeError(
            f"getItem failed ({response.status_code}): {response.text[:500]}"
        )
    return response.json()


if __name__ == "__main__":
    import sys

    # Usage:
    #   python -m ebay_mcp.ebay_client "<query>"        -> search
    #   python -m ebay_mcp.ebay_client item "<item_id>" -> get item
    if len(sys.argv) > 2 and sys.argv[1] == "item":
        item = get_item(sys.argv[2])
        print(f"[env: {ENV}, marketplace: {MARKETPLACE}]")
        print(f"Title: {item.get('title', '')[:120]}")
        price = item.get("price", {})
        print(f"Price: {price.get('value', '?')} {price.get('currency', '?')}")
        print(f"Web URL: {item.get('itemWebUrl', '')}")
    else:
        q = sys.argv[1] if len(sys.argv) > 1 else "iphone"
        result = search_items(q, limit=3)
        print(f"[env: {ENV}, marketplace: {MARKETPLACE}]")
        print(f"Total: {result.get('total', 0)}\n")
        for it in result.get("itemSummaries", []):
            price = it.get("price", {})
            print(f"- {it.get('title', '')[:80]}")
            print(f"  itemId: {it.get('itemId', '?')}")
            print(f"  {price.get('value', '?')} {price.get('currency', '?')}")
            print(f"  {it.get('itemWebUrl', '')}\n")
