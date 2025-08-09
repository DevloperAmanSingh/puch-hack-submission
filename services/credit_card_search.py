import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pinecone import Pinecone

from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as OpenAILLM
from llama_index.vector_stores.pinecone import PineconeVectorStore


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "cards-embeddings")

assert OPENAI_API_KEY, "OPENAI_API_KEY is required"
assert PINECONE_API_KEY, "PINECONE_API_KEY is required"


class CreditCardSearchService:
    """Search service using LlamaIndex over Pinecone vectors."""

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

        # Configure embeddings and LLM
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=OPENAI_API_KEY,
        )
        Settings.llm = OpenAILLM(
            model="gpt-4o-mini",
            api_key=OPENAI_API_KEY,
        )

        # Connect Pinecone and wrap as a vector store
        pc = Pinecone(api_key=PINECONE_API_KEY)
        self._index = pc.Index(PINECONE_INDEX_NAME)
        vector_store = PineconeVectorStore(pinecone_index=self._index)

        # Build an index object from the existing vector store
        self._index_obj = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        self._qe = self._index_obj.as_query_engine(
            similarity_top_k=self.top_k,
            response_mode="compact",
        )

    def search(self, query: str, include_sources: bool = True) -> Dict[str, Any]:
        resp = self._qe.query(query)
        answer = str(resp)

        sources: List[Dict[str, Any]] = []
        if include_sources and getattr(resp, "source_nodes", None):
            for sn in resp.source_nodes:
                meta = sn.node.metadata or {}
                sources.append(
                    {
                        "source": meta.get("source", ""),
                        "bank": meta.get("bank", ""),
                        "card_name": meta.get("card_name", ""),
                        "content_snippet": sn.node.get_text()[:300],
                        "score": getattr(sn, "score", None),
                    }
                )

        return {"answer": answer, "sources": sources}

    def get_health_status(self) -> Dict[str, Any]:
        status: Dict[str, Any] = {"status": "ok"}
        try:
            stats = self._index.describe_index_stats()
            status["vectordb_status"] = "ok"
            status["documents_count"] = stats.get("total_vector_count", "unknown")
        except Exception as e:
            status["status"] = "degraded"
            status["vectordb_status"] = "error"
            status["error"] = str(e)
        return status


_service: Optional[CreditCardSearchService] = None


def get_search_service() -> CreditCardSearchService:
    global _service
    if _service is None:
        _service = CreditCardSearchService(top_k=5)
    return _service


