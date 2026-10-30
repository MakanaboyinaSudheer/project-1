import os
import sqlite3
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
    last_message = state["messages"][-1].content if state.get("messages") else "No summary generated."
    return {"memory_summary": f"Review Summary: {last_message}"}


def code_node(state: AgentState):
    return {"code_attempts": [{"status": "pending", "code": "# Code generation pending"}]}


def critic_node(state: AgentState):
    return {}


def route_review(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "hypothesis"


workflow = StateGraph(AgentState)
workflow.add_node("review", review_node)
workflow.add_node("tools", tool_node)
workflow.add_node("hypothesis", hypothesis_node)
workflow.add_node("code", code_node)
workflow.add_node("critic", critic_node)

workflow.add_edge(START, "review")
workflow.add_conditional_edges("review", route_review, {"tools": "tools", "hypothesis": "hypothesis"})
workflow.add_edge("tools", "review")
workflow.add_edge("hypothesis", "code")
workflow.add_edge("code", "critic")
workflow.add_edge("critic", END)

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
