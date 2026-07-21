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

mcp = FastMCP(
    name="ebay",
    version="0.1.0",
    instructions=(
        "Search eBay listings via the official Browse API. The marketplace is "
        f"{MARKETPLACE} (e.g. EBAY_DE = ebay.de), so titles, prices and sellers "
        "are localised to that site and prices are in its currency. Start with "
        "`search_ebay` to get listing summaries and their `item_id` values, then "
        "call `get_item_details` on the `item_id` of anything worth a closer "
        "look to get the full description, item specifics, shipping and return "
        "terms. This is read-only: it cannot bid, buy or watch."
    ),
)


# --------------------------------------------------------------------------- #
# tools
# --------------------------------------------------------------------------- #


@mcp.tool
def search_ebay(
    query: Annotated[
        str,
        Field(description="Search keywords, e.g. 'ThinkPad T14 AMD' or 'Rolex Datejust'"),
    ],
    limit: Annotated[
        int, Field(description="Maximum results to return", ge=1, le=200)
    ] = 10,
    filter_expr: Annotated[
        Optional[str],
        Field(
            description=(
                "eBay Browse API filter syntax, comma-separated. Examples: "
                "'price:[100..500],priceCurrency:EUR' (price range), "
                "'conditions:{NEW}' (new only), "
                "'buyingOptions:{FIXED_PRICE}' (Buy It Now only), "
                "'itemLocationCountry:DE' (German sellers only)."
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

    Returns listing summaries — id, title, price, condition, seller reputation,
    location, thumbnail and URL. Pass an item's `item_id` to `get_item_details`
    for the full record. Prices and currency follow the configured marketplace.
    """
    raw = search_items(query=query, limit=limit, filter_expr=filter_expr, sort=sort)

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

    return {
        "environment": ENV,
        "marketplace": MARKETPLACE,
        "query": query,
        "total": raw.get("total", 0),
        "returned": len(items),
        "items": items,
    }


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
    """Retrieve the full record of a single eBay item.

    Use after `search_ebay` surfaces something worth a closer look: full
    description, item specifics (aspects), all images, shipping options, return
    policy, seller and location. The `raw` key holds the unfiltered Browse API
    response in case a field was dropped.
    """
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
