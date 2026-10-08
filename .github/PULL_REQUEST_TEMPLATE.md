## What
<!-- One or two sentences. Link the module ID from MODULES.md, e.g. M1.2 -->

## Why

## How tested
- [ ] `pytest -m "not integration"` passes
- [ ] `ruff check . && ruff format --check .` passes
- [ ] Integration tested locally (Milvus / Docker / LLM), if relevant

## Checklist
- [ ] No secrets or `.env` committed
- [ ] `schemas.py` / `graph/state.py` unchanged, **or** all 3 members tagged for review
- [ ] MODULES.md / ARCHITECTURE.md updated if interfaces changed
