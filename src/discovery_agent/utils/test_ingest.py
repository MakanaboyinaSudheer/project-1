from langchain_ollama import OllamaEmbeddings
from pymilvus import MilvusClient


def insert_test_paper():
    print("Generating embeddings using Ollama...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://ollama:11434")

    title = "AI-Driven Causal Inference in E-Commerce"
    abstract = (
        "This paper explores the application of propensity score weighting "
        "and Hajek-IPTW estimators on the Olist dataset to evaluate the "
        "causal impact of shipping times on customer satisfaction."
    )

    vector = embeddings.embed_query(abstract)

    print("Connecting to Milvus to insert data...")
    client = MilvusClient(uri="http://milvus:19530")

    data = [{"title": title, "abstract": abstract, "embedding": vector}]

    res = client.insert(collection_name="research_papers", data=data)
    print(f"Successfully inserted {res['insert_count']} paper(s) into Milvus!")


if __name__ == "__main__":
    insert_test_paper()
