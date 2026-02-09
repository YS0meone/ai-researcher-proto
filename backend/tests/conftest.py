import pytest
from httpx import ASGITransport, AsyncClient

from app.webapp.app import app


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac
