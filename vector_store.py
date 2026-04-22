import chromadb
from chromadb.config import Settings
from openai import OpenAI


_client: OpenAI | None = None
_chroma: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None

COLLECTION_NAME = "pdf_chunks"
EMBED_MODEL = "text-embedding-3-small"


def _get_openai(api_key: str) -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=api_key)
    return _client


def _get_collection() -> chromadb.Collection:
    global _chroma, _collection
    if _chroma is None:
        _chroma = chromadb.Client(Settings(anonymized_telemetry=False))
    if _collection is None:
        _collection = _chroma.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def embed_texts(texts: list[str], api_key: str) -> list[list[float]]:
    client = _get_openai(api_key)
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def add_chunks(chunks: list[dict], api_key: str) -> int:
    collection = _get_collection()

    # 이미 로드된 source 파일 목록 확인
    existing = collection.get(include=["metadatas"])
    existing_sources = {m["source"] for m in existing["metadatas"]} if existing["metadatas"] else set()

    new_chunks = [c for c in chunks if c["source"] not in existing_sources]
    if not new_chunks:
        return 0

    texts = [c["text"] for c in new_chunks]
    embeddings = embed_texts(texts, api_key)

    ids = [f"{c['source']}__p{c['page']}__c{c['chunk_index']}" for c in new_chunks]
    metadatas = [{"source": c["source"], "page": c["page"]} for c in new_chunks]

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return len(new_chunks)


def search(query: str, api_key: str, n_results: int = 5) -> list[dict]:
    collection = _get_collection()
    if collection.count() == 0:
        return []

    query_embedding = embed_texts([query], api_key)[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text": doc,
            "source": meta["source"],
            "page": meta["page"],
            "score": 1 - dist,  # cosine similarity
        })
    return hits


def list_sources() -> list[str]:
    collection = _get_collection()
    if collection.count() == 0:
        return []
    result = collection.get(include=["metadatas"])
    return sorted({m["source"] for m in result["metadatas"]})


def remove_source(source: str) -> None:
    collection = _get_collection()
    result = collection.get(include=["metadatas"])
    ids_to_delete = [
        id_ for id_, meta in zip(result["ids"], result["metadatas"])
        if meta["source"] == source
    ]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)


def reset_collection() -> None:
    global _collection
    if _chroma is not None:
        _chroma.delete_collection(COLLECTION_NAME)
        _collection = None
