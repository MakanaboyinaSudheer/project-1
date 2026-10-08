"""Ingestion orchestration: fetch -> dedupe -> chunk -> embed -> upsert. Owner: Member 1."""


def run_ingestion(query: str, max_results: int | None = None) -> int:
    """Ingest papers for `query` from ArXiv + PubMed. Returns number of new chunks upserted.

    Must be idempotent (re-running does not create duplicates) so it can run on a schedule.
    """
    raise NotImplementedError
