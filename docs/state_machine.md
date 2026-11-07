# Discovery Agent: State Machine Specification (M2.6)

## Multi-Agent Graph Architecture
The Core Orchestration graph utilizes `LangGraph` and `SqliteSaver` for checkpointing. The state transitions are governed by a strict token-budgeted memory manager and a max-iteration routing policy to prevent deadlocks.

### Visual Graph
```mermaid
stateDiagram-v2
    [*] --> LiteratureReviewer
    LiteratureReviewer --> HypothesisGenerator
    HypothesisGenerator --> CodeExecutor
    CodeExecutor --> StatisticalCritic

    state route_after_critic <<choice>>
    StatisticalCritic --> route_after_critic

    route_after_critic --> Execution : APPROVED or Max Iterations Reached
    route_after_critic --> HypothesisGenerator : REJECTED (Loop back)

    Execution --> Synthesizer
    Synthesizer --> [*]
