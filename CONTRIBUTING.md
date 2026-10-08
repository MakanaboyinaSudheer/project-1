# Contributing: Team Workflow

## Branches
```
main        ← protected; only release merges from develop (end of Week 2, Week 4)
develop     ← protected; integration branch, every PR targets this
feature/<module-id>-<short-desc>   e.g. feature/M1.2-pubmed-client
fix/<module-id>-<short-desc>       e.g. fix/M3.1-timeout-kill
docs/<short-desc>
```
- Never push directly to `main` or `develop`. Open a PR and get **1 approving review** from a teammate.
- Keep branches short-lived (≤3 days). Rebase on `develop` daily.
- Squash-merge PRs into `develop`.

## Commits ([Conventional Commits](https://www.conventionalcommits.org/))
```
<type>(<module-id>): <imperative summary, ≤72 chars>

feat(M1.2): add PubMed efetch XML parsing
fix(M3.1): kill container on wall-clock timeout
test(M2.3): reject hypotheses citing unknown papers
docs: add Milvus troubleshooting to README
chore(ci): cache pip in GitHub Actions
```
Types: `feat`, `fix`, `test`, `docs`, `refactor`, `chore`, `perf`.

**Daily commits are mandatory.** Push to your feature branch every day, even for WIP
(use a `wip:` prefix; the squash merge cleans it up).

## Before you push
```bash
ruff check . --fix && ruff format .
pytest -m "not integration"
```
pre-commit runs ruff automatically on `git commit`.

## Shared contracts
`src/discovery_agent/schemas.py` and `src/discovery_agent/graph/state.py` are shared by
everyone. Change them in a dedicated PR, tag all members, and update MODULES.md.

## Secrets
Only in `.env` (gitignored). Never paste keys in code, issues or PRs. If a key leaks, rotate it immediately.
