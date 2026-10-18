"""Measure ranked retrieval quality against manually labeled query/paper pairs."""

import time
from collections.abc import Callable, Sequence

from discovery_agent.retrieval.retriever import retrieve


def benchmark_retrieval(
    cases: Sequence[dict[str, object]],
    k: int = 8,
    retrieve_fn: Callable = retrieve,
) -> dict[str, object]:
    if not cases:
        raise ValueError("At least one labeled query is required")
    if k < 1:
        raise ValueError("k must be positive")
    recalls = []
    reciprocal_ranks = []
    latencies_ms = []
    for index, case in enumerate(cases, start=1):
        query = case.get("query")
        relevant = case.get("relevant_paper_ids")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Benchmark case {index} has an empty query")
        if not isinstance(relevant, list) or not relevant:
            raise ValueError(f"Benchmark case {index} must label relevant_paper_ids")
        started = time.perf_counter()
        results = retrieve_fn(query, k=k, peer_reviewed_only=True)
        latencies_ms.append((time.perf_counter() - started) * 1000)
        ranked_ids = [item.paper_id for item in results]
        relevant_ids = set(relevant)
        recalls.append(len(set(ranked_ids) & relevant_ids) / len(relevant_ids))
        reciprocal_ranks.append(
            next(
                (1 / rank for rank, paper_id in enumerate(ranked_ids, start=1) if paper_id in relevant_ids),
                0.0,
            )
        )
    return {
        "query_count": len(cases),
        "k": k,
        "recall_at_k": sum(recalls) / len(recalls),
        "mean_reciprocal_rank": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "mean_latency_ms": sum(latencies_ms) / len(latencies_ms),
    }
