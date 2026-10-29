from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from tools.milvus_tools import search_research_papers


def test_literature_agent():
    print("Initializing Agent with Llama 3.1 and Milvus Search Tool...")

    llm = ChatOllama(model="llama3.1", base_url="http://ollama:11434", temperature=0)

    tools = [search_research_papers]
    agent_executor = create_react_agent(llm, tools)

    question = "Based on our database, how do shipping times impact e-commerce?"
    print(f"\nUser Question: {question}")
    print("-" * 50)

    for chunk in agent_executor.stream({"messages": [("user", question)]}, stream_mode="values"):
        message = chunk["messages"][-1]
        if message.content:
            print(f"Agent: {message.content}")
        elif hasattr(message, "tool_calls") and message.tool_calls:
            print(f"[Agent is calling tool: {message.tool_calls[0]['name']}]")


if __name__ == "__main__":
    test_literature_agent()
