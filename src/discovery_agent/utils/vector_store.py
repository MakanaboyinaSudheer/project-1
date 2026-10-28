from pymilvus import DataType, MilvusClient


def setup_collection():
    client = MilvusClient(uri="http://milvus:19530")
    collection_name = "research_papers"

    if client.has_collection(collection_name):
        print(f"Collection '{collection_name}' already exists.")
        return client

    print(f"Creating collection '{collection_name}'...")

    schema = MilvusClient.create_schema(
        auto_id=True,
        enable_dynamic_field=True,
    )

    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=1000)
    schema.add_field(field_name="abstract", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=768)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")

    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)

    print(f"Collection '{collection_name}' created successfully.")
    return client


if __name__ == "__main__":
    setup_collection()
