"""Shared pytest fixtures — isolate user endpoint discoveries from the real .rcr store."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_user_endpoints(tmp_path, monkeypatch):
    """Prevent ModelRegistry from loading the developer's .rcr/user_endpoints.json."""
    monkeypatch.setenv("RCR_USE_USER_ENDPOINTS", "false")
    monkeypatch.setenv("RCR_USER_ENDPOINTS_PATH", str(tmp_path / "no-user-endpoints.json"))
    # Keep unit tests on in-memory stores unless a test opts into OpenSearch.
    monkeypatch.setenv("RCR_USE_OPENSEARCH", "false")
