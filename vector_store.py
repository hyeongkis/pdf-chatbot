from __future__ import annotations

import math
from typing import Dict, List, Optional
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"

# 인메모리 저장소
_store: Dict[str, dict] = {}   # id -> {text, source, page, embedding}
_openai_client: Optional[OpenAI] = None


def _get_openai(api_key: str) -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def _embed(texts: List[str], api_key: str) -> List[List[float]]:
    client = _get_openai(api_key)
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def add_chunks(chunks: List[dict], api_key: str) -> int:
    existing_sources = {v["source"] for v in _store.values()}
    new_chunks = [c for c in chunks if c["source"] not in existing_sources]
    if not new_chunks:
        return 0

    texts = [c["text"] for c in new_chunks]
    embeddings = _embed(texts, api_key)

    for chunk, emb in zip(new_chunks, embeddings):
        chunk_id = f"{chunk['source']}__p{chunk['page']}__c{chunk['chunk_index']}"
        _store[chunk_id] = {
            "text": chunk["text"],
            "source": chunk["source"],
            "page": chunk["page"],
            "embedding": emb,
        }
    return len(new_chunks)


def search(query: str, api_key: str, n_results: int = 5) -> List[dict]:
    if not _store:
        return []

    query_emb = _embed([query], api_key)[0]
    scored = [
        (chunk_id, _cosine(query_emb, v["embedding"]))
        for chunk_id, v in _store.items()
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:n_results]

    return [
        {
            "text": _store[cid]["text"],
            "source": _store[cid]["source"],
            "page": _store[cid]["page"],
            "score": score,
        }
        for cid, score in top
    ]


def list_sources() -> List[str]:
    return sorted({v["source"] for v in _store.values()})


def remove_source(source: str) -> None:
    to_delete = [k for k, v in _store.items() if v["source"] == source]
    for k in to_delete:
        del _store[k]


def reset_collection() -> None:
    _store.clear()
