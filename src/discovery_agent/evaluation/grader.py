"""Grade agent output against the holdout set. Owner: Member 1 (Week 4).

Metrics: hypothesis overlap with holdout findings (embedding similarity + LLM judge),
citation validity rate, experiment success rate, statistical validity, cost per run.
"""


def grade_run(run_dir: str) -> dict:
    raise NotImplementedError
