# M4 Evaluation, Synthesis, and Dashboard Report

## Delivered

- **M4.1 telemetry:** each completed LLM callback writes token counts, model,
  graph node, estimated USD cost, and UTC timestamp to the run's `costs.jsonl`.
- **M4.2 dashboard:** displays runs, graph trace, the generated hypothesis's
  cited sources and quotations, attempt reward curve and code diffs, critic
  report, per-model spend, live refresh, and JSON/paper-draft downloads.
- **M4.3 critic:** deterministic metric validation and Holm-Bonferroni
  adjustment precede the LLM methodology review. The review is cost-tracked and
  cannot change the deterministic statistical fields.
- **M4.4 synthesis:** Jinja2 renders `abstract.tex`, `paper.tex`, `refs.bib`,
  and `draft.md`. Synthesis attempts PDF compilation using Tectonic or
  pdfLaTeX when installed and reports `compiled` or `unavailable` in graph
  state. Compiler failures are surfaced rather than reported as success.
- **M4.5 evaluation:** the fixed holdout loader validates JSONL records and
  removes matching IDs through `exclude_holdout_papers`. The grader persists
  embedding similarity, structured 0–5 LLM evidence-alignment scores,
  peer-reviewed citation validity, experiment success, critic validity, and
  total run cost to `grade.json`.

## Validation and benchmark status

Unit tests exercise all the above paths with deterministic fakes, including
holdout exclusion, grader report persistence, dashboard citation/diff/reward
data, and LaTeX compiler success/failure reporting. These tests are software
validation, not scientific benchmark scores.

No live holdout grading result is claimed here: it requires the configured
embedding model and LLM and must be run against a completed run with
`discovery-agent grade --run-dir runs/<id>`.

## Integration dependency

`src/discovery_agent/ingestion/pipeline.py` remains a separate M1.4 stub. The
holdout exclusion helper is implemented and tested, but the ingestion command
cannot yet call it until M1.4 implements its fetch/embed/upsert flow. The end-
to-end no-leakage ingestion check therefore remains pending that module.

PDF compilation also depends on an external LaTeX compiler; install Tectonic or
pdfLaTeX on a machine that needs `paper.pdf`. Without one, the workflow reports
that compilation was unavailable and retains the generated LaTeX source.
