import pytest
from unittest.mock import patch, AsyncMock, MagicMock


def _make_http_mock(data) -> tuple:
    """Returns (mock_response, mock_client) wired for `async with httpx.AsyncClient() as client:`."""
    mock_response = MagicMock()
    mock_response.json.return_value = data
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.put = AsyncMock(return_value=mock_response)
    return mock_response, mock_client


@pytest.mark.asyncio
async def test_predict_flow_helper():
    _, mock_client = _make_http_mock({
        "flow_id": "abc123", "consensus": "SAFE", "severity": "LOW",
    })
    with patch("httpx.AsyncClient", return_value=mock_client):
        from mcp_server.server import call_predict_flow
        result = await call_predict_flow({"feature_1": 0.5})
    assert result["consensus"] == "SAFE"


@pytest.mark.asyncio
async def test_get_metrics_helper():
    _, mock_client = _make_http_mock({
        "ae_threshold": 0.023, "total_alerts": 10, "feedback_count": 3,
    })
    with patch("httpx.AsyncClient", return_value=mock_client):
        from mcp_server.server import call_get_metrics
        result = await call_get_metrics()
    assert result["ae_threshold"] == 0.023


@pytest.mark.asyncio
async def test_list_alerts_helper():
    _, mock_client = _make_http_mock([
        {"id": 1, "consensus": "CONFIRMED", "severity": "CRITICAL"},
    ])
    with patch("httpx.AsyncClient", return_value=mock_client):
        from mcp_server.server import call_list_alerts
        result = await call_list_alerts(limit=5)
    assert isinstance(result, list)
    assert result[0]["consensus"] == "CONFIRMED"
