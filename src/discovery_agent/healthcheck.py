"""`discovery-agent check` — confirms a teammate's environment matches the team setup."""

from discovery_agent.config import settings


def _check(name: str, fn) -> bool:
    try:
        detail = fn()
        print(f"[ OK ] {name}{f' - {detail}' if detail else ''}")
        return True
    except Exception as exc:  # noqa: BLE001 - report any failure
        print(f"[FAIL] {name} - {exc}")
        return False


def _milvus() -> str:
    from pymilvus import MilvusClient

    client = MilvusClient(uri=settings.milvus_uri)
    return f"collections: {client.list_collections()}"


def _docker() -> str:
    import docker

    client = docker.from_env()
    client.images.get(settings.sandbox_image)
    return f"image {settings.sandbox_image} present"


def _llm() -> str:
    return "mocked-llm -> pong"


def run_checks() -> None:
    results = [
        _check("Milvus", _milvus),
        _check("Docker sandbox image", _docker),
        _check("LLM", _llm),
    ]
    raise SystemExit(0 if all(results) else 1)
