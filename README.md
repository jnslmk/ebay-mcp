# ebay-mcp

An [MCP](https://modelcontextprotocol.io) server that lets an LLM search and
inspect [eBay](https://www.ebay.de) listings through the official
[Browse API](https://developer.ebay.com/api-docs/buy/browse/overview.html).
Read-only, buyer-side: it searches and reads listings, it does not bid or buy.

This is a fork of Suat Oneren's [ebay-browse-mcp][upstream], which speaks stdio.
The fork keeps the two tools and their response shaping and swaps the transport
for **streamable-HTTP** (`:8000/mcp`) plus a `/healthz` probe, so it runs as a
self-hosted container behind an MCP client such as LibreChat — the same shape as
its sibling [kleinanzeigen-mcp](https://github.com/jnslmk/kleinanzeigen-mcp).

Unlike Kleinanzeigen (no public API, so scraping is unavoidable), eBay offers a
sanctioned API — so this server hits it directly: no headless browser, no
bot-detection fragility.

[upstream]: https://github.com/capitansuat/ebay-browse-mcp

## Tools

| Tool | What it does |
|------|--------------|
| `search_ebay` | Search by keyword with optional `min_price`/`max_price` (or a raw price filter via `filter_expr` — not both) plus condition, buying option and seller-country filters, and sort. Returns listing summaries + `item_id`s. `query` is ANDed against the listing title only, so it wants 2-4 short product keywords, not a sentence — a zero-result response explains why and may already include one narrowed retry (which preserves any price filter). |
| `get_item_details` | Full record for one `item_id`: description, item specifics, images, shipping options, return terms, seller. |

## Configuration

Set an eBay Developer keyset (free — <https://developer.ebay.com/my/keys>):

| Env var | Default | Meaning |
|---------|---------|---------|
| `EBAY_CLIENT_ID` | — | App ID (Client ID) from your keyset |
| `EBAY_CLIENT_SECRET` | — | Cert ID (Client Secret) from your keyset |
| `EBAY_MARKETPLACE` | `EBAY_DE` | Marketplace: `EBAY_DE`, `EBAY_US`, `EBAY_GB`, … |
| `EBAY_ENV` | `sandbox` | `sandbox` or `production` |
| `EBAY_RETRY_CAP` | `8` | Upper bound (seconds) on retry sleeps, including a server-sent `Retry-After` |
| `MCP_TRANSPORT` | `http` | `http` (streamable-HTTP) or `stdio` |
| `MCP_HOST` / `MCP_PORT` / `MCP_PATH` | `0.0.0.0` / `8000` / `/mcp` | HTTP bind |

Sandbox has almost no real inventory — use it to prove the wiring, then set
`EBAY_ENV=production` (with a production, OAuth-enabled keyset) for real results.

## Run

### Docker (recommended)

```bash
docker run --rm -p 8000:8000 \
  -e EBAY_CLIENT_ID=... -e EBAY_CLIENT_SECRET=... \
  -e EBAY_MARKETPLACE=EBAY_DE -e EBAY_ENV=production \
  ghcr.io/jnslmk/ebay-mcp:latest
# MCP endpoint: http://localhost:8000/mcp   health: http://localhost:8000/healthz
```

### Local (stdio, for desktop MCP clients)

```bash
python3 -m venv .venv && .venv/bin/pip install .
cp .env.example .env   # fill in your keyset
MCP_TRANSPORT=stdio .venv/bin/ebay-mcp
```

Quick client-free check of the keyset:

```bash
.venv/bin/python -m ebay_mcp.ebay_client "raspberry pi 5"
```

## License

MIT. See [LICENSE](LICENSE) — retains the original upstream copyright.
