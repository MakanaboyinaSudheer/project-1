from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings
from pymilvus import MilvusClient


@tool
def search_research_papers(query: str, limit: int = 2) -> str:
    """
    Search the Milvus vector database for research papers relevant to the query.
    Returns a formatted string of the most relevant titles and abstracts.
    """
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://ollama:11434")
    query_vector = embeddings.embed_query(query)

    client = MilvusClient(uri="http://milvus:19530")
    results = client.search(
        collection_name="research_papers",
        data=[query_vector],
        limit=limit,
        output_fields=["title", "abstract"],
    )

    if not results or not results[0]:
        return "No relevant research papers found in the database."

    formatted_results = "--- Relevant Research Papers ---\n"
    for hit in results[0]:
        formatted_results += f"Title: {hit['entity']['title']}\n"
        formatted_results += f"Abstract: {hit['entity']['abstract']}\n"
        formatted_results += f"Similarity: {hit['distance']:.4f}\n\n"

    return formatted_results
