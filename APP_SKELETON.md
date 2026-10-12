# Dashboard App Skeleton and MVP Plan

## Product goal

Build a browser-based dashboard for observing scientific discovery runs. The dashboard
does not run agents itself: it reads run artifacts and presents their progress and
results to a human supervisor.

## Technology and operating model

- **UI:** Streamlit, written in Python and opened in a web browser.
- **Run data:** files under `runs/<run_id>/`, including cost JSONL and later run
  summaries, traces, attempts, citations, critic reports, and paper drafts.
- **Workflow data:** LangGraph checkpoints when the graph/checkpointer integration is
  available.
- **First release:** read-only local dashboard; no separate React frontend or API
  service is required.

Start the dashboard from the repository root:

```bash
streamlit run dashboard/app.py
```

## Suggested app structure

Keep the first implementation in `dashboard/app.py`. Extract data-reading helpers into
`dashboard/data.py` only when the app needs shared parsing or validation.

The app should be organized into these user-facing areas:

1. **Run list:** available run IDs, latest status, topic, and last-updated time.
2. **Run overview:** selected run's topic, status, and summary metrics.
3. **Trace:** ordered workflow steps from the run trace.
4. **Evidence and results:** citations, experiment attempts, critic report, and generated
   paper artifacts as those become available.
5. **Costs:** per-call model, token counts, and estimated USD from `costs.jsonl`.

Read each artifact through a small helper that validates the expected JSON shape. Show a
clear empty state when an optional artifact has not been generated yet. Surface malformed
data as a visible warning; do not silently substitute fabricated values or fail the
entire page when one optional section is unavailable.

## Incremental MVPs

### MVP 1 — Run monitor (Days 3–4)

**Purpose:** make it possible to see that a run exists and inspect its progress.

- List run directories and select one.
- Show its available status and topic/summary fields.
- Display trace entries in workflow order as a readable logic tree or timeline.
- Handle an empty `runs/` directory and a run without a trace.

**Acceptance criteria:** the dashboard starts with the documented command; a user can
select a run and inspect its trace; empty or incomplete run data produces a useful
message rather than a crash.

### MVP 2 — Experiment review (Days 8–10)

**Purpose:** explain what the agent tried and how it evaluated the result.

- Show code attempts, source/output details, and diffs between attempts when available.
- Plot reward progression across attempts.
- Show literature citations and the critic report.
- Show cost totals and per-model/per-call token and estimated USD details.
- Provide a repeatable walkthrough for a live run as part of the Day 10 MVP review.

**Acceptance criteria:** a supervisor can follow a run from trace to experiment attempt,
review its evidence and critic result, and see its costs; missing optional artifacts are
labelled as unavailable.

### MVP 3 — Live monitoring and export (Day 17)

**Purpose:** make the dashboard useful while runs are still progressing and make their
results easy to share.

- Add configurable or user-triggered refresh without requiring a server restart.
- Refresh the selected run's data safely while preserving the selection.
- Export supported run summaries and results in useful downloadable formats.

**Acceptance criteria:** newly written run data becomes visible after refresh, exports
contain the selected run's actual data, and incomplete runs remain viewable.

### Final dashboard — Release (Day 20)

**Purpose:** deliver a polished, reliable read-only view as part of the full agent.

- Verify all dashboard sections against an end-to-end run.
- Improve navigation, labels, empty/error states, and layout consistency.
- Document a clean-start and demo workflow.

**Acceptance criteria:** the dashboard can inspect a completed run from literature and
trace through attempts, critic, costs, and generated paper output, and it remains useful
when artifacts are missing or a run fails.

## Scope boundaries

- The dashboard is a browser-based web UI; Streamlit is the selected first implementation.
- Do not add React, a backend API, authentication, or run-control actions to these MVPs.
- Do not imply that estimated token costs are provider-billed totals.
- Dashboard features depend on agent modules producing their documented run artifacts;
  until then, show the relevant section as pending or unavailable.

## Schedule alignment

The milestones above follow the Member 4 (Sudheer) M4.2 dashboard schedule in
[`MODULES.md`](./MODULES.md) and the Day 10/Day 20 project milestones in
[`PLAN.md`](./PLAN.md). System data sources and output expectations are described in
[`ARCHITECTURE.md`](./ARCHITECTURE.md).
