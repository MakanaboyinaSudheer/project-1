import pytest
from pymilvus import MilvusClient

from discovery_agent.config import settings


@pytest.mark.integration
def test_milvus_is_ready_and_accepts_client_requests() -> None:
    client = MilvusClient(uri=settings.milvus_uri, timeout=10)
    try:
        collections = client.list_collections()
    finally:
        client.close()

    assert isinstance(collections, list)
