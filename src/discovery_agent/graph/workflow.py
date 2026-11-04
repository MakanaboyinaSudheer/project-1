import os
import sqlite3
import subprocess
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from discovery_agent.tools.milvus_tools import search_research_papers


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    research_topic: str
    memory_summary: str
    code_attempts: list[dict]


llm = ChatOllama(model="llama3.1", base_url="http://ollama:11434", temperature=0)
tools = [search_research_papers]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)


def review_node(state: AgentState):
    messages = state.get("messages", [])
    topic = state.get("research_topic", "")

    if not messages:
        sys_msg = SystemMessage(
            content=(
                "You are an AI literature reviewer. Use the search_research_papers "
                "tool to find context, then summarize the findings."
            )
        )
        human_msg = HumanMessage(content=f"Find papers and summarize context for this topic: {topic}")
        messages = [sys_msg, human_msg]

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def hypothesis_node(state: AgentState):
    messages = state.get("messages", [])
    topic = state.get("research_topic", "")
    review_summary = messages[-1].content if messages else "No prior context."
    sys_msg = SystemMessage(
        content=(
            "You are a lead AI research scientist specializing in causal inference. "
            "Based on the provided literature review, formulate a novel, testable hypothesis. "
            "Clearly define the recommended treatment, outcome, and potential confounders to test."
        )
    )
    human_msg = HumanMessage(
        content=f"Topic: {topic}\n\nLiterature Review:\n{review_summary}\n\nGenerate the hypothesis."
    )
    response = llm.invoke([sys_msg, human_msg])
    combined_memory = (
        f"**1. Literature Review:**\n{review_summary}\n\n**2. Proposed Hypothesis:**\n{response.content}"
    )
    return {"messages": [response], "memory_summary": combined_memory}


def code_node(state: AgentState):
    messages = state.get("messages", [])

    hypothesis = messages[-1].content if messages else "No hypothesis provided."

    sys_msg = SystemMessage(
        content=(
            "You are an expert Python data scientist. Based on the provided hypothesis, "
            "write a Python script using the DoWhy library for causal inference. "
            "Output ONLY valid Python code enclosed in ```python blocks. No explanations."
        )
    )
    human_msg = HumanMessage(content=f"Write code to test this hypothesis:\n{hypothesis}")

    response = llm.invoke([sys_msg, human_msg])

    raw_text = response.content
    generated_code = raw_text
    if "```python" in raw_text:
        generated_code = raw_text.split("```python")[1].split("```")[0].strip()
    elif "```" in raw_text:
        generated_code = raw_text.split("```")[1].split("```")[0].strip()

    code_attempt = {"status": "generated", "code": generated_code}
    current_attempts = state.get("code_attempts", []) + [code_attempt]

    return {"messages": [response], "code_attempts": current_attempts}


def critic_node(state: AgentState):
    code_attempts = state.get("code_attempts", [])
    latest_code = code_attempts[-1]["code"] if code_attempts else "No code provided."

    sys_msg = SystemMessage(
        content=(
            "You are a code reviewer. Check if the provided Python script effectively "
            "tests the research hypothesis. If it is mathematically and logically sound, "
            "output EXACTLY 'APPROVED'. Otherwise, provide brief revision instructions."
        )
    )
    human_msg = HumanMessage(content=f"Review this code:\n{latest_code}")

    response = llm.invoke([sys_msg, human_msg])
    feedback = response.content.strip()

    code_attempts[-1]["critic_feedback"] = feedback
    code_attempts[-1]["status"] = "approved" if feedback == "APPROVED" else "rejected"

    return {"messages": [response], "code_attempts": code_attempts}


def execution_node(state: AgentState):
    attempts = state.get("code_attempts", [])
    if not attempts or attempts[-1].get("status") != "approved":
        return {}

    approved_code = attempts[-1]["code"]

    with open("temp_eval.py", "w") as f:
        f.write(approved_code)

    try:
        result = subprocess.run(["python", "temp_eval.py"], capture_output=True, text=True, timeout=60)
        output = result.stdout if result.returncode == 0 else result.stderr
    except Exception as e:
        output = str(e)

    if os.path.exists("temp_eval.py"):
        os.remove("temp_eval.py")

    execution_msg = HumanMessage(content=f"Execution Results:\n{output}")
    return {"messages": [execution_msg]}


def report_node(state: AgentState):
    messages = state.get("messages", [])
    execution_result = messages[-1].content if messages else "No execution data."
    context = state.get("memory_summary", "")

    sys_msg = SystemMessage(
        content=(
            "You are a Senior AI Researcher. Write a concise final markdown report "
            "summarizing the initial hypothesis, the experimental approach, and the "
            "final execution results."
        )
    )
    human_msg = HumanMessage(
        content=f"Context:\n{context}\n\nExecution Results:\n{execution_result}\n\nWrite the report."
    )

    response = llm.invoke([sys_msg, human_msg])
    return {"messages": [response]}


def route_review(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "hypothesis"


def route_critic(state: AgentState):
    attempts = state.get("code_attempts", [])
    if attempts and attempts[-1].get("status") == "approved":
        return "execute"
    if not attempts or len(attempts) >= 3:
        return END
    return "code"


workflow = StateGraph(AgentState)
workflow.add_node("review", review_node)
workflow.add_node("tools", tool_node)
workflow.add_node("hypothesis", hypothesis_node)
workflow.add_node("code", code_node)
workflow.add_node("critic", critic_node)
workflow.add_node("execute", execution_node)
workflow.add_node("report", report_node)

workflow.add_edge(START, "review")
workflow.add_conditional_edges("review", route_review, {"tools": "tools", "hypothesis": "hypothesis"})
workflow.add_edge("tools", "review")
workflow.add_edge("hypothesis", "code")
workflow.add_edge("code", "critic")
workflow.add_conditional_edges("critic", route_critic, {"code": "code", "execute": "execute", END: END})
workflow.add_edge("execute", "report")
workflow.add_edge("report", END)

os.makedirs("volumes", exist_ok=True)


def run_discovery(topic: str, thread_id: str) -> dict:
    with sqlite3.connect("volumes/checkpoints.sqlite", check_same_thread=False) as conn:
        memory = SqliteSaver(conn)
        app = workflow.compile(checkpointer=memory)

        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {"research_topic": topic, "messages": []}

        final_state = app.invoke(initial_state, config=config)
        return final_state


def get_run_status(thread_id: str) -> dict:
    with sqlite3.connect("volumes/checkpoints.sqlite", check_same_thread=False) as conn:
        memory = SqliteSaver(conn)
        app = workflow.compile(checkpointer=memory)
        config = {"configurable": {"thread_id": thread_id}}
        state = app.get_state(config)

        if state and hasattr(state, "values"):
            return {"status": "completed" if not state.next else "in_progress", "pending_tasks": state.next}
        return {}
