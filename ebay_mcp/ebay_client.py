"""eBay Browse API client. Standalone Python, no MCP dependency.

Forked from Suat Oneren's ebay-browse-mcp. The client itself is essentially
upstream's — the fork adds an HTTP transport and container packaging on top
(see ``server.py``). Kept framework-free so it can also be driven from the CLI
(`python -m ebay_mcp.ebay_client "<query>"`) for debugging inside the image.
"""
from __future__ import annotations

import email.utils
import os
import random
import threading
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
# FastMCP runs the sync tools in a thread pool; the lock dedupes concurrent
# token refreshes so an agent burst with an expired cache fires one OAuth
# token request, not one per call (token endpoints are rate-limited too).
_token_lock = threading.Lock()

# Process-level session: one connection pool reused across token/search/item
# calls, with a consistent browser-like default header set. Per-request
# headers (_headers) are merged on top of these by requests.
# Thread-safety caveat: the sync tools run in a thread pool, but this session
# (and the module's cookie jar) is written for single-threaded use — requests
# only guarantees thread-safe connection pooling, not cookie state.
_session = requests.Session()
_session.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64; rv:132.0) Gecko/20100101 Firefox/132.0"
        ),
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
    }
)

_MAX_ATTEMPTS = 4
# Ceiling applied to both the exponential backoff and a server-sent Retry-After.
# Configurable for servers that legitimately ask for longer waits; the default
# still protects against pathological values (Retry-After measured in hours).
try:
    _RETRY_CAP = float(os.environ.get("EBAY_RETRY_CAP", "8"))
except ValueError:
    _RETRY_CAP = 8.0


def _retry_delay(attempt: int, response: Optional[requests.Response]) -> float:
    """Sleep delay after a failed attempt: Retry-After if given, else
    jittered exponential backoff."""
    retry_after: Optional[float] = None
    if response is not None:
        raw = response.headers.get("Retry-After")
        if raw:
            try:
                retry_after = max(0.0, float(raw))
            except ValueError:
                try:
                    target = email.utils.parsedate_to_datetime(raw)
                    retry_after = max(0.0, target.timestamp() - time.time())
                except (TypeError, ValueError):
                    retry_after = None
    if retry_after is None:
        retry_after = min(_RETRY_CAP, 0.5 * 2**attempt)
    return min(_RETRY_CAP, retry_after) + random.uniform(0, 0.25)


def _request(
    method: str, url: str, *, context: str, **kwargs: object
) -> requests.Response:
    """Run one HTTP request with bounded retries on 429/5xx.

    Retries transient status codes with jittered exponential backoff,
    honoring a numeric or HTTP-date ``Retry-After`` header. Non-transient
    failures raise a RuntimeError with response context (single attempt,
    no retry loop)."""
    kwargs.setdefault("timeout", 15)
    for attempt in range(_MAX_ATTEMPTS):
        response = _session.request(method, url, **kwargs)  # type: ignore[arg-type]
        if response.status_code in (429, 500, 502, 503, 504) and attempt < _MAX_ATTEMPTS - 1:
            time.sleep(_retry_delay(attempt, response))
            continue
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"{context} failed ({response.status_code}): {response.text[:500]}"
            ) from exc
        return response
    # Unreachable: the loop either returns or raises on its final attempt.
    raise RuntimeError(f"{context} failed after {_MAX_ATTEMPTS} attempts")  # pragma: no cover


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

    with _token_lock:
        # Re-check under the lock: another thread may have refreshed while we waited.
        now = time.time()
        if _token_cache["value"] and now < _token_cache["expires_at"]:  # type: ignore[operator]
            return _token_cache["value"]  # type: ignore[return-value]
        response = _request(
            "post",
            TOKEN_URL,
            context="Token request",
            auth=(CLIENT_ID, CLIENT_SECRET),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "client_credentials",
                "scope": "https://api.ebay.com/oauth/api_scope",
            },
            timeout=10,
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
    response = _request(
        "get", SEARCH_URL, context="Search", headers=_headers(token), params=params
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
    response = _request(
        "get", url, context="getItem", headers=_headers(token), params=params
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
