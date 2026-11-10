from langchain_ollama import OllamaEmbeddings
from pymilvus import MilvusClient


def search_papers():
    print("Embedding the search query...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://ollama:11434")

    query = "How do shipping times impact e-commerce customers?"
    query_vector = embeddings.embed_query(query)

    print("Searching Milvus...")
    client = MilvusClient(uri="http://milvus:19530")

    results = client.search(
        collection_name="research_papers",
        data=[query_vector],
        limit=2,  # return top 2 results
        output_fields=["title", "abstract"],
    )

    print("\n--- Search Results ---")
    for hits in results:
        for hit in hits:
            print(f"Score: {hit['distance']:.4f} (Lower means more similar)")
            print(f"Title: {hit['entity']['title']}")
            print(f"Abstract: {hit['entity']['abstract']}\n")


if __name__ == "__main__":
    search_papers()
