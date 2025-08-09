import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore

from pinecone import Pinecone, ServerlessSpec

load_dotenv()

DATA_DIR = "llm_data"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
EMBED_MODEL_NAME = "text-embedding-3-small"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "cards-embeddings")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

assert PINECONE_API_KEY, "PINECONE_API_KEY is required"
assert OPENAI_API_KEY, "OPENAI_API_KEY is required"


def load_documents() -> List[Document]:
    docs: List[Document] = []
    base = Path(DATA_DIR)
    for bank_dir in base.iterdir():
        if not bank_dir.is_dir():
            continue
        bank = bank_dir.name.replace("-", " ").title()
        for txt_file in bank_dir.glob("*.txt"):
            content = txt_file.read_text(encoding="utf-8").strip()
            if not content:
                continue
            card_name = txt_file.stem.replace("_", " ")
            docs.append(
                Document(
                    text=content,
                    metadata={
                        "source": str(txt_file),
                        "bank": bank,
                        "card_name": card_name,
                    },
                )
            )
    return docs


def ensure_pinecone_index(pc: Pinecone, name: str, dim: int = 1536, metric: str = "cosine") -> None:
    existing = {idx["name"] for idx in pc.list_indexes()}
    if name not in existing:
        pc.create_index(
            name=name,
            dimension=dim,
            metric=metric,
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )


def main() -> None:
    print("Loading documents from llm_data/ ...")
    documents = load_documents()
    print(f"Loaded {len(documents)} documents")

    print(f"Chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}) ...")
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Created {len(nodes)} chunks")

    print("Initializing Pinecone ...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    ensure_pinecone_index(pc, PINECONE_INDEX_NAME, dim=1536)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)

    print("Setting up embeddings ...")
    embed_model = OpenAIEmbedding(model=EMBED_MODEL_NAME, api_key=OPENAI_API_KEY)

    print("Building vector store and indexing ...")
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build index from nodes (will embed + upsert to Pinecone)
    VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    print("Done. Chunks embedded and stored in Pinecone.")


if __name__ == "__main__":
    main()