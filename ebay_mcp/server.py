"""MCP server exposing eBay listing search via the Browse API.

Forked from Suat Oneren's ebay-browse-mcp, which spoke stdio only. This fork
keeps the two tools and the response shaping but swaps the transport for
streamable-HTTP so it can run as a long-lived container behind LibreChat (and
adds a ``/healthz`` readiness probe), mirroring the sibling kleinanzeigen-mcp.

The Browse API is the buyer-side, read-only search API. The default marketplace
is ``EBAY_DE`` (ebay.de); override with ``EBAY_MARKETPLACE``.
"""

from __future__ import annotations

import logging
import os
import re
from importlib.metadata import PackageNotFoundError, version
from typing import Annotated, Any, Optional

from fastmcp import FastMCP
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse

from ebay_mcp.ebay_client import (
    ENV,
    MARKETPLACE,
    credentials_configured,
    get_item,
    search_items,
)

log = logging.getLogger("ebay-mcp")

try:
    # Single source of truth is pyproject.toml's [project.version]; this reads
    # it back from the installed package's metadata instead of hardcoding a
    # second copy here that silently drifts (as the 0.1.1 literal already had).
    _VERSION = version("ebay-mcp")
except PackageNotFoundError:
    # Editable/uninstalled checkout (e.g. running server.py directly without
    # `pip install .`) has no package metadata to read.
    _VERSION = "0.0.0-dev"


def _coerce_int(
    value: str | int | None, field: str, *, ge: int | None = None
) -> int | None:
    """Coerce the numeric strings LLMs routinely send for int parameters.

    FastMCP validates tool input against the JSON schema before the function
    runs, so a parameter typed ``int`` rejects the string ``"10"`` outright
    (the same bug fixed in kleinanzeigen-mcp 0.1.1, geizhals-mcp 0.1.3 and
    aliexpress-mcp 0.1.1). Accepting ``str | int`` in the schema and
    normalising here keeps the model-facing contract lenient while the client
    still sees a real int.
    """
    if value is None or isinstance(value, int):
        result = value
    elif isinstance(value, str) and value.strip():
        try:
            result = int(value.strip())
        except ValueError as exc:
            raise ValueError(f"{field} must be an integer, got {value!r}") from exc
    else:
        raise ValueError(f"{field} must be an integer, got {value!r}")
    if ge is not None and result is not None and result < ge:
        raise ValueError(f"{field} must be >= {ge}, got {result}")
    return result


def _coerce_float(
    value: str | int | float | None, field: str, *, ge: float | None = None
) -> float | None:
    """Coerce the numeric strings LLMs routinely send for float parameters.

    Same rationale as ``_coerce_int`` above (FastMCP's schema-level ``float``
    type rejects the string ``"99.99"`` outright, before this function ever
    runs) applied to prices, which are rarely whole numbers so ``_coerce_int``
    itself won't do. Mirrors the ``_coerce_float`` already shipped in
    aliexpress-mcp 0.1.x and amazon-mcp so the coercion contract is identical
    across every sibling server that takes a price.
    """
    if value is None or isinstance(value, (int, float)):
        result = float(value) if value is not None else None
    elif isinstance(value, str) and value.strip():
        try:
            result = float(value.strip())
        except ValueError as exc:
            raise ValueError(f"{field} must be a number, got {value!r}") from exc
    else:
        raise ValueError(f"{field} must be a number, got {value!r}")
    if ge is not None and result is not None and result < ge:
        raise ValueError(f"{field} must be >= {ge}, got {result}")
    return result


mcp = FastMCP(
    name="ebay",
    version=_VERSION,
    instructions=(
        "Search eBay listings via the official Browse API. The marketplace is "
        f"{MARKETPLACE} (e.g. EBAY_DE = ebay.de), so titles, prices and sellers "
        "are localised to that site and prices are in its currency. Start with "
        "`search_ebay` to get listing summaries and their `item_id` values, then "
        "call `get_item_details` on the `item_id` of anything worth a closer "
        "look to get the full description, item specifics, shipping and return "
        "terms. This is read-only: it cannot bid, buy or watch.\n\n"
        "`search_ebay`'s `query` ANDs every keyword against the listing TITLE "
        "only, so it rewards short, literal product keywords over a "
        "descriptive sentence — see the `query` parameter's own description "
        "for the exact rule before building one."
    ),
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

# Words a model reaches for to describe condition, use-case or seller instead
# of naming the product — none of these are typically in a listing TITLE, so
# eBay's title-only AND zeroes the result set the moment one appears. Dropped
# by the one-shot retry in _narrow_query below.
_STOP_WORDS = frozenset(
    {
        "used",
        "gebraucht",
        "cheap",
        "best",
        "local",
        "seller",
        "deep",
        "learning",
        "ai",
        "training",
        "complete",
        "system",
    }
)

# How many keywords a narrowed retry keeps. eBay ANDs every keyword against
# the title, so fewer keywords means a wider net; 3 is generous enough to stay
# specific (a bare 'RTX' pulls in every RTX card ever listed) while still
# recovering from one bad word.
_NARROW_KEYWORD_COUNT = 3


def _narrow_query(query: str) -> str:
    """Drop stop-list words, then truncate to the first few tokens.

    One deterministic pass, no recursion: this is called at most once per
    search (see the zero-result retry in ``search_ebay``), so there is no way
    for a narrowed query to trigger another narrowing.
    """
    tokens = query.split()
    filtered = [t for t in tokens if t.lower() not in _STOP_WORDS]
    if not filtered:
        # Dropping every stop-list word would leave nothing to search — better
        # to keep the original tokens (just truncated) than search for "".
        filtered = tokens
    return " ".join(filtered[:_NARROW_KEYWORD_COUNT])


def _zero_result_hint(query: str) -> str:
    """Explain the AND-over-title semantics and name likely culprit tokens.

    Attached to any response with ``total == 0`` so the model sees *why*,
    instead of concluding the item doesn't exist and rephrasing into a longer,
    more descriptive query — the exact wrong direction (that's what the AND
    semantics punish hardest).
    """
    culprits = [t for t in query.split() if t.lower() in _STOP_WORDS]
    if culprits:
        culprit_note = f" Likely culprit(s) in this query: {', '.join(culprits)}."
    else:
        culprit_note = (
            " None of these words are on the known stop-list — try dropping "
            "one keyword at a time to find the one that isn't in any title."
        )
    return (
        "eBay's Browse API ANDs every keyword in `query` against the listing "
        "TITLE only — a single word that isn't in any title zeroes the whole "
        "result set. Retry with 2-4 short product keywords and move "
        "condition/location/use-case words into `filter_expr` instead." + culprit_note
    )


def _resolve_limit(limit: str | int | None, max_results: str | int | None) -> int:
    """Accept either name for the result-cap knob.

    Five sibling MCP servers share one model: geizhals-mcp and baumarkt-mcp
    call this ``max_results``, kleinanzeigen-mcp and aliexpress-mcp call it
    ``limit``. FastMCP emits ``additionalProperties: false``, so a model that
    carries the wrong name over from a sibling server gets a hard schema
    rejection (probed live: ``max_results`` against this server was rejected
    outright) — the same class of bug ``_resolve_page_count`` fixed in
    kleinanzeigen-mcp. ``limit`` stays canonical; ``max_results`` is declared
    in the schema (not silently swallowed) as a deprecated alias.
    """
    if limit is not None and max_results is not None:
        resolved = _coerce_int(limit, "limit", ge=1)
        alias = _coerce_int(max_results, "max_results", ge=1)
        if resolved != alias:
            raise ValueError(
                "limit and max_results are two names for the same parameter "
                f"but were given different values ({resolved} and {alias}); "
                "pass limit only"
            )
    elif max_results is not None:
        resolved = _coerce_int(max_results, "max_results", ge=1)
    else:
        resolved = _coerce_int(limit, "limit", ge=1)
    return min(resolved or 10, 200)


# ISO 4217 currency for each eBay marketplace ID this server might plausibly
# be pointed at via EBAY_MARKETPLACE. The Browse API's price-range filter
# requires an accompanying ``priceCurrency`` and never infers one from the
# marketplace header itself, so min_price/max_price need a currency from
# *somewhere*. Money is not a field worth guessing at: a marketplace missing
# from this table raises (see _price_currency below) instead of silently
# attaching the wrong currency, which could zero out the search or — worse —
# silently filter a different currency than the caller typed. Extend this
# table before pointing EBAY_MARKETPLACE at a marketplace not listed here.
_MARKETPLACE_CURRENCY: dict[str, str] = {
    "EBAY_US": "USD",
    "EBAY_GB": "GBP",
    "EBAY_DE": "EUR",
    "EBAY_AT": "EUR",
    "EBAY_CH": "CHF",
    "EBAY_FR": "EUR",
    "EBAY_IT": "EUR",
    "EBAY_ES": "EUR",
    "EBAY_NL": "EUR",
    "EBAY_BE": "EUR",
    "EBAY_IE": "EUR",
    "EBAY_AU": "AUD",
    "EBAY_CA": "CAD",
}


def _price_currency() -> str:
    """Resolve the ISO currency to pair with a min_price/max_price filter.

    Looked up from the configured ``MARKETPLACE`` (see ebay_client.py, set via
    ``EBAY_MARKETPLACE`` and defaulting to ``EBAY_DE``) rather than hardcoded,
    so a deployment pointed at a different marketplace filters in that
    marketplace's currency instead of silently mislabeling it as EUR.
    """
    try:
        return _MARKETPLACE_CURRENCY[MARKETPLACE]
    except KeyError:
        raise ValueError(
            f"no known currency for marketplace {MARKETPLACE!r}; add it to "
            "_MARKETPLACE_CURRENCY in server.py before using min_price/"
            "max_price on this marketplace, or pass a full "
            "'price:[min..max],priceCurrency:<code>' filter via filter_expr "
            "instead"
        ) from None


# Matches a `price` filter *key* right at the start of filter_expr or right
# after a comma — not "priceCurrency", whose extra characters between "price"
# and ":" fail the \s*: requirement immediately after "price". This is the
# substring trap called out in the task: a filter_expr that only sets
# priceCurrency (no range) must NOT be treated as a pre-existing price filter
# by *this* regex — _PRICE_CURRENCY_FILTER_RE below is what catches that case.
_PRICE_FILTER_RE = re.compile(r"(?i)(?:^|,)\s*price\s*:")

# Matches a `priceCurrency` filter *key*, same anchoring as _PRICE_FILTER_RE.
# The literal "Currency" right after "price" means this can never match a
# bare `price:[...]` range (which has "[" right after "price", not "C"), so
# the two regexes are mutually exclusive by construction — neither can trip
# on the other's key.
_PRICE_CURRENCY_FILTER_RE = re.compile(r"(?i)(?:^|,)\s*priceCurrency\s*:")


def _resolve_price_filter(
    min_price: float | None, max_price: float | None, filter_expr: str | None
) -> str | None:
    """Fold min_price/max_price into filter_expr as a Browse API price range.

    Shaped like _resolve_limit above: two ways to say the same thing, so a
    combination that could silently pick a winner instead raises. Here that
    means a caller-supplied `price:[...]` or `priceCurrency:...` filter *and*
    min_price/max_price at the same time — resolving that silently would
    either drop a constraint the caller wrote, or (for priceCurrency
    specifically) emit two conflicting `priceCurrency` keys in one filter
    string, which is malformed and leaves it ambiguous which currency the
    Browse API would honour.

    The Browse API's price filter is an open-ended range — `price:[100..]`
    (min only), `price:[..500]` (max only), `price:[100..500]` (both) — and a
    price filter always requires an accompanying `priceCurrency`, which has no
    default and is looked up via _price_currency() above. Because
    `priceCurrency` is only ever meaningful alongside a price range, a caller
    who set it is expressing price through filter_expr, so it gets the same
    "raise, don't reconcile" treatment as an explicit `price:` filter — this
    check only runs when we're about to add our own derived filter (i.e.
    min_price or max_price was given); with neither given, filter_expr
    (including a lone priceCurrency) passes through untouched below.
    """
    if min_price is None and max_price is None:
        return filter_expr
    if filter_expr and _PRICE_FILTER_RE.search(filter_expr):
        raise ValueError(
            "filter_expr already contains a `price` filter and min_price/"
            f"max_price were also given (min_price={min_price!r}, "
            f"max_price={max_price!r}); express the price range in one place "
            "only — either filter_expr's price:[...] or min_price/max_price, "
            "not both"
        )
    if filter_expr and _PRICE_CURRENCY_FILTER_RE.search(filter_expr):
        raise ValueError(
            "filter_expr already contains a `priceCurrency` filter and "
            f"min_price/max_price were also given (min_price={min_price!r}, "
            f"max_price={max_price!r}); a price range always carries its own "
            "priceCurrency, so express it in one place only — either "
            "filter_expr's price:[...],priceCurrency:... or min_price/"
            "max_price, not both"
        )
    if min_price is not None and max_price is not None and min_price > max_price:
        raise ValueError(
            f"min_price ({min_price}) is greater than max_price ({max_price})"
        )
    low = "" if min_price is None else _format_price(min_price)
    high = "" if max_price is None else _format_price(max_price)
    price_filter = f"price:[{low}..{high}],priceCurrency:{_price_currency()}"
    return f"{filter_expr},{price_filter}" if filter_expr else price_filter


def _format_price(value: float) -> str:
    """Render a coerced price without a spurious trailing ``.0``.

    100, "100" and 100.0 all coerce to the float 100.0 (see _coerce_float);
    without this, only the last of those would render as "price:[100..]" —
    the other two would produce "price:[100.0..]", which is functionally
    identical to the Browse API but makes the log line below noisier for no
    reason.
    """
    return str(int(value)) if value == int(value) else str(value)


def _shape_summaries(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Trim a Browse API search response down to what a model needs.

    Shared between the primary search and the zero-result fallback so the two
    can never drift into shaping results differently.
    """
    items = []
    for it in raw.get("itemSummaries", []):
        price = it.get("price") or {}
        seller = it.get("seller") or {}
        items.append(
            {
                "item_id": it.get("itemId"),
                "title": it.get("title"),
                "price": price.get("value"),
                "currency": price.get("currency"),
                "condition": it.get("condition"),
                "seller_username": seller.get("username"),
                "seller_feedback_pct": seller.get("feedbackPercentage"),
                "seller_feedback_score": seller.get("feedbackScore"),
                "item_location": (it.get("itemLocation") or {}).get("country"),
                "url": it.get("itemWebUrl"),
                "image": (it.get("image") or {}).get("imageUrl"),
            }
        )
    return items


# --------------------------------------------------------------------------- #
# tools
# --------------------------------------------------------------------------- #


@mcp.tool
def search_ebay(
    query: Annotated[
        str,
        Field(
            description=(
                "2-4 short product keywords ANDed against the listing TITLE "
                "(e.g. 'RTX 3090 24GB') — never a sentence. Words like "
                "'used'/'deep learning'/'local seller' are rarely in titles "
                "and return zero results; put condition/location in "
                "`filter_expr` instead. On a German marketplace prefer "
                "German terms ('gebraucht', 'Grafikkarte')."
            )
        ),
    ],
    limit: Annotated[
        str | int | None, Field(description="Maximum results to return (default 10)")
    ] = None,
    max_results: Annotated[
        str | int | None,
        Field(description="Deprecated alias for `limit`; prefer `limit`"),
    ] = None,
    filter_expr: Annotated[
        Optional[str],
        Field(
            description=(
                "eBay Browse API filter syntax, comma-separated. Examples: "
                "'price:[100..500],priceCurrency:EUR' (price range — prefer "
                "`min_price`/`max_price` below instead of writing this by "
                "hand), 'conditions:{NEW}' (new only), "
                "'buyingOptions:{FIXED_PRICE}' (Buy It Now only), "
                "'itemLocationCountry:DE' (German sellers only). Do not "
                "combine a `price` or `priceCurrency` filter here with "
                "`min_price`/`max_price` — pick one or the other."
            )
        ),
    ] = None,
    min_price: Annotated[
        str | int | float | None,
        Field(
            description=(
                "Minimum price, in the configured marketplace's currency "
                "(e.g. EUR on EBAY_DE). Convenience over hand-writing "
                "filter_expr's 'price:[min..],priceCurrency:...' range — do "
                "not also set a `price` or `priceCurrency` filter in "
                "filter_expr, that raises an error."
            )
        ),
    ] = None,
    max_price: Annotated[
        str | int | float | None,
        Field(
            description=(
                "Maximum price, in the configured marketplace's currency. "
                "See `min_price` — same rule: don't combine with a `price` "
                "or `priceCurrency` filter in filter_expr."
            )
        ),
    ] = None,
    sort: Annotated[
        Optional[str],
        Field(
            description=(
                "'price' (ascending), '-price' (descending), 'newlyListed' or "
                "'endingSoonest'. Omit for eBay Best Match."
            )
        ),
    ] = None,
) -> dict[str, Any]:
    """Search eBay listings by keyword, with optional filters and sorting.

    `query` is a strict AND over 2-4 short keywords matched against the
    listing TITLE only (see the `query` field's description) — a zero-result
    response carries a `hint` explaining why instead of a bare empty list, and
    one automatic narrowed retry may already have run (see `fallback_query`).
    Returns listing summaries — id, title, price, condition, seller reputation,
    location, thumbnail and URL. Pass an item's `item_id` to `get_item_details`
    for the full record. Prices and currency follow the configured marketplace.
    `min_price`/`max_price` are a convenience over `filter_expr`'s `price`
    range syntax — see those parameters' own descriptions for the rule against
    combining the two.
    """
    resolved_limit = _resolve_limit(limit, max_results)
    min_price = _coerce_float(min_price, "min_price", ge=0)
    max_price = _coerce_float(max_price, "max_price", ge=0)
    resolved_filter_expr = _resolve_price_filter(min_price, max_price, filter_expr)
    raw = search_items(
        query=query, limit=resolved_limit, filter_expr=resolved_filter_expr, sort=sort
    )
    total = raw.get("total", 0)

    # One automatic keyword-drop retry on a zeroed search: the AND-over-title
    # semantics mean a longer, more "descriptive" query is *more* likely to
    # zero out, so the wrong instinct (add more words) makes it worse. This
    # runs at most once — _narrow_query is a single deterministic pass with no
    # recursion, so a fallback that also returns zero just falls through to
    # the hint below instead of retrying again.
    fallback_query: str | None = None
    tokens = query.split()
    if total == 0 and len(tokens) >= 3:
        narrowed = _narrow_query(query)
        if narrowed.split() != tokens:
            # filter_expr here is resolved_filter_expr, not the caller's raw
            # filter_expr — the retry must carry the derived price filter
            # exactly as the first call had it. Silently dropping it on
            # fallback would return results the caller explicitly excluded
            # with min_price/max_price.
            fallback_raw = search_items(
                query=narrowed,
                limit=resolved_limit,
                filter_expr=resolved_filter_expr,
                sort=sort,
            )
            fallback_query = narrowed
            fallback_total = fallback_raw.get("total", 0)
            if fallback_total > 0:
                raw = fallback_raw
                total = fallback_total

    items = _shape_summaries(raw)

    result: dict[str, Any] = {
        "environment": ENV,
        "marketplace": MARKETPLACE,
        "query": query,
        "total": total,
        "returned": len(items),
        "items": items,
    }
    if fallback_query is not None:
        result["fallback_query"] = fallback_query
    if total == 0:
        result["hint"] = _zero_result_hint(fallback_query or query)

    log.info(
        "search_ebay query=%r filter_expr=%r limit=%s total=%s fallback_query=%r",
        query,
        resolved_filter_expr,
        resolved_limit,
        total,
        fallback_query,
    )
    return result


@mcp.tool
def get_item_details(
    item_id: Annotated[
        str,
        Field(
            description=(
                "REST item id from search_ebay's `item_id` field "
                "(format 'v1|<numeric>|0'). Legacy numeric ids are not accepted."
            )
        ),
    ],
    fieldgroups: Annotated[
        Optional[str],
        Field(
            description=(
                "Optional response shaping: 'COMPACT' (small subset for change "
                "detection) or 'PRODUCT' (adds catalogue info). Omit for full "
                "details."
            )
        ),
    ] = None,
) -> dict[str, Any]:
    """Retrieve the shaped detail record for a single eBay item.

    Use after `search_ebay` surfaces something worth a closer look: full
    description, item specifics (aspects), all images, shipping options, return
    policy, seller and location.
    """
    # No `raw` key: the dict below already surfaces every field an agent
    # realistically acts on (description, aspects, images, shipping, returns,
    # seller, location). The Browse API's getItem response carries a good deal
    # more (taxes, affiliate/guest-checkout flags, priority-listing metadata,
    # fitment/compatibility, ...) that's rarely useful here, and echoing all of
    # it — description text included — a second time as `raw` would roughly
    # double the token cost of *every* detail call to hedge a handful of niche
    # fields. Add fields to the shaped dict below if a real gap shows up.
    raw = get_item(item_id=item_id, fieldgroups=fieldgroups)

    price = raw.get("price") or {}
    seller = raw.get("seller") or {}
    location = raw.get("itemLocation") or {}
    primary_image = (raw.get("image") or {}).get("imageUrl")
    additional_images = [
        img.get("imageUrl")
        for img in (raw.get("additionalImages") or [])
        if img.get("imageUrl")
    ]

    # Item specifics (aspects) come back as a list of {name, value} dicts.
    aspects: dict[str, Any] = {}
    for aspect in raw.get("localizedAspects") or []:
        name = aspect.get("name")
        value = aspect.get("value")
        if name and value is not None:
            aspects[name] = value

    shipping_options = []
    for opt in raw.get("shippingOptions") or []:
        cost = opt.get("shippingCost") or {}
        shipping_options.append(
            {
                "type": opt.get("type"),
                "cost": cost.get("value"),
                "currency": cost.get("currency"),
                "min_estimated_delivery": opt.get("minEstimatedDeliveryDate"),
                "max_estimated_delivery": opt.get("maxEstimatedDeliveryDate"),
            }
        )

    return_terms = raw.get("returnTerms") or {}

    return {
        "environment": ENV,
        "marketplace": MARKETPLACE,
        "item_id": raw.get("itemId"),
        "legacy_item_id": raw.get("legacyItemId"),
        "title": raw.get("title"),
        "subtitle": raw.get("subtitle"),
        "short_description": raw.get("shortDescription"),
        "description": raw.get("description"),
        "price": price.get("value"),
        "currency": price.get("currency"),
        "condition": raw.get("condition"),
        "condition_id": raw.get("conditionId"),
        "brand": raw.get("brand"),
        "mpn": raw.get("mpn"),
        "category_path": raw.get("categoryPath"),
        "buying_options": raw.get("buyingOptions"),
        "estimated_availabilities": raw.get("estimatedAvailabilities"),
        "aspects": aspects,
        "images": ([primary_image] if primary_image else []) + additional_images,
        "seller": {
            "username": seller.get("username"),
            "feedback_pct": seller.get("feedbackPercentage"),
            "feedback_score": seller.get("feedbackScore"),
        },
        "item_location": {
            "country": location.get("country"),
            "city": location.get("city"),
            "postal_code": location.get("postalCode"),
        },
        "shipping_options": shipping_options,
        "return_terms": {
            "returns_accepted": return_terms.get("returnsAccepted"),
            "return_period": (return_terms.get("returnPeriod") or {}).get("value"),
            "refund_method": return_terms.get("refundMethod"),
            "return_shipping_cost_payer": return_terms.get("returnShippingCostPayer"),
        },
        "url": raw.get("itemWebUrl"),
    }


# --------------------------------------------------------------------------- #
# transport
# --------------------------------------------------------------------------- #


@mcp.custom_route("/healthz", methods=["GET"])
async def healthz(_: Request) -> JSONResponse:
    """Container readiness probe: the API is unusable without a keyset."""
    if not credentials_configured():
        return JSONResponse(
            {"status": "misconfigured", "detail": "EBAY_CLIENT_ID/SECRET unset"},
            status_code=503,
        )
    return JSONResponse({"status": "ok", "environment": ENV, "marketplace": MARKETPLACE})


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if not credentials_configured():
        # Not fatal — the server still starts so /healthz can report why — but
        # surface it loudly, since every tool call will otherwise fail.
        log.warning("EBAY_CLIENT_ID/EBAY_CLIENT_SECRET are not set; tools will fail")

    transport = os.getenv("MCP_TRANSPORT", "http")
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(
            transport="http",
            host=os.getenv("MCP_HOST", "0.0.0.0"),  # noqa: S104 - containerised
            port=int(os.getenv("MCP_PORT", "8000")),
            path=os.getenv("MCP_PATH", "/mcp"),
        )


if __name__ == "__main__":
    main()
