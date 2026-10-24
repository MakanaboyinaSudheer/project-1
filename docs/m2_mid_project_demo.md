# M2: Core Agents & Orchestration - Mid-Project Demo

## Objective
Demonstrate the end-to-end execution of the v0.1 LangGraph state machine, highlighting the integration of the Literature Reviewer and the structured Hypothesis Generator.

## Prerequisites
1. Check out the `develop` branch.
2. Ensure the SQLite checkpointer volume exists (`mkdir -p volumes`).
3. Export your LLM API keys.

## Demo Steps

### Step 1: Initialize the Run
Run the discovery CLI or execute a test script targeting `run_discovery` with a sample topic:
`topic = "Federated learning over multimodal clinical data"`

### Step 2: Literature Reviewer (M2.2)
- **Action:** Show the terminal output as the graph enters `literature_reviewer`.
- **Verification:** Inspect the `ResearchState` to confirm `literature_summary` is populated.

### Step 3: Hypothesis Generator (M2.3)
- **Action:** Step into the `hypothesis_generator` node.
- **Verification:** Print `state["hypothesis"]`. Emphasize that the output is strictly typed via Pydantic (`title`, `reasoning`, `proposed_methodology`, `citations`).
- **Validation:** Demonstrate the citation validation retry loop by temporarily injecting a fake citation ID into the code and showing the LLM correcting it.

### Step 4: Routing & Stubs
- **Action:** Show the state machine seamlessly routing to the remaining nodes (Code Executor, Critic, Synthesizer).
- **Verification:** Prove the graph reaches `END` without crashing, successfully preserving the thread state in SQLite.
