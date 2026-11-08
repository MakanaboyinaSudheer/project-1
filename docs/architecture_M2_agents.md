# Architecture Report: Stream M2 - Core Agents & Orchestration
**Owner:** Aadil (Member 2)

## 1. Executive Summary
Stream M2 handles the central brain of the Discovery Agent. It utilizes a state machine built on LangGraph to orchestrate a multi-agent workflow (Reviewer -> Hypothesis -> Executor -> Critic). By implementing token-budgeted memory management and cyclic refinement safety policies, the orchestration layer supports long-running, autonomous scientific discovery without context exhaustion or deadlocks.

## 2. Core Components

### 2.1 LangGraph State Machine (`graph/workflow.py`)
- **Structure:** A directed cyclic graph defining the agent workflow.
- **State Persistence:** Utilizes `SqliteSaver` via SQLite checkpointing to persist thread states, allowing the system to resume seamlessly after process kills or container restarts.

### 2.2 Memory Management (`memory/manager.py`)
- **Token Budgeting:** Dynamically estimates token usage (`~4 chars/token` heuristic) to prevent local LLMs from overflowing their context windows.
- **Rolling Compression:** Older `CodeAttempt` logs are truncated and aggressively summarized into a `memory_summary`, preserving only the two most recent attempts alongside the core hypothesis and critic feedback.

### 2.3 Refinement Loop & Safety Routing (`graph/routing.py`)
- **Critic Integration:** The graph cycles between the Executor and the Critic.
- **Deadlock Prevention:** The `route_after_critic` policy enforces a strict maximum iteration budget (`MAX_REFINEMENT_STEPS = 3`). If the Critic continuously rejects the outputs, the router forces progression to the Synthesizer node to avoid infinite loops.

## 3. Future Enhancements
- Transition token estimation to a strict tokenizer (e.g., `tiktoken`) if migrating away from local Ollama models.
- Implement parallel agent execution for multi-threaded hypothesis generation.
