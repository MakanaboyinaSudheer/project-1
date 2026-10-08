"""Central settings, loaded from environment / .env. Import `settings` everywhere."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    llm_model: str = "anthropic:claude-sonnet-5-5"
    llm_temperature: float = 0.2
    vlm_model: str = "ollama:llava"
    ollama_base_url: str = "http://localhost:11434"

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Milvus
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "papers"

    # Literature APIs
    ncbi_email: str = ""
    ncbi_api_key: str = ""
    arxiv_categories: str = "cs.LG,cs.AI,q-bio.QM"
    ingest_max_results: int = 50

    # Sandbox
    sandbox_image: str = "discovery-sandbox:latest"
    sandbox_timeout_s: int = 120
    sandbox_mem_limit: str = "2g"
    sandbox_cpus: float = 1.0

    # Agent loop
    max_debug_iterations: int = 5
    runs_dir: str = "runs"
    checkpoint_db: str = "checkpoints/graph.sqlite"

    @property
    def arxiv_category_list(self) -> list[str]:
        return [c.strip() for c in self.arxiv_categories.split(",") if c.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
