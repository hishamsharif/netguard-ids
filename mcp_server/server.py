"""
NetGuard MCP Server — exposes IDS inference as Claude/Codex tools via FastMCP.
Run with: python -m mcp_server.server
"""
import httpx
from fastmcp import FastMCP
from app.config import settings

mcp = FastMCP("NetGuard IDS")
API_BASE = f"http://localhost:{settings.app_port}"


# ── Testable HTTP helpers (used by tools and tests alike) ────────────────────

async def call_predict_flow(features: dict) -> dict:
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{API_BASE}/predict/flow", json={"features": features})
        r.raise_for_status()
        return r.json()


async def call_explain_flow(features: dict) -> dict:
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{API_BASE}/explain/flow", json={"features": features})
        r.raise_for_status()
        return r.json()


async def call_list_alerts(limit: int = 20) -> list:
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{API_BASE}/alerts/", params={"limit": limit})
        r.raise_for_status()
        return r.json()


async def call_get_metrics() -> dict:
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{API_BASE}/metrics/")
        r.raise_for_status()
        return r.json()


async def call_update_threshold(threshold: float) -> dict:
    async with httpx.AsyncClient() as client:
        r = await client.put(f"{API_BASE}/metrics/threshold", json={"threshold": threshold})
        r.raise_for_status()
        return r.json()


# ── MCP Tool registrations ────────────────────────────────────────────────────

@mcp.tool()
async def predict_flow(features: dict) -> dict:
    """
    Classify a single network flow.
    Args:
        features: dict of CIC-IDS2017 feature name → float value
    Returns:
        rule verdict, consensus (SAFE/CONFIRMED/KNOWN_PATTERN/ZERO_DAY), severity, ai_skipped flag
    """
    return await call_predict_flow(features)


@mcp.tool()
async def explain_flow(features: dict) -> dict:
    """
    Explain a flow prediction with RF feature importance and AE reconstruction errors.
    Returns top contributing features and a natural-language summary.
    """
    return await call_explain_flow(features)


@mcp.tool()
async def list_recent_alerts(limit: int = 20) -> list:
    """Retrieve the most recent IDS alerts from the persistent alert log."""
    return await call_list_alerts(limit)


@mcp.tool()
async def get_metrics() -> dict:
    """Return live IDS performance metrics: precision, recall, F1, FPR, alert counts."""
    return await call_get_metrics()


@mcp.tool()
async def update_ae_threshold(threshold: float) -> dict:
    """
    Adjust the Autoencoder anomaly detection threshold at runtime.
    Args:
        threshold: float between 0.0 and 1.0
    """
    return await call_update_threshold(threshold)


if __name__ == "__main__":
    mcp.run(port=settings.mcp_port)
