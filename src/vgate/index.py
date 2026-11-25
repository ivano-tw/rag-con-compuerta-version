"""Representación vectorial e indexación en Qdrant.

La recuperación es deliberadamente agnóstica a la versión: no se filtra por versión
porque la contribución evaluada interviene en la decisión de responder, no en qué se
recupera. El payload conserva la versión para calcular la señal de alineación.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from . import config


@lru_cache(maxsize=1)
def embedder() -> SentenceTransformer:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(config.EMBED_MODEL, device=device)
    model.max_seq_length = 384  # fragmentos de ~750 caracteres caben holgadamente
    return model


def embed_passages(texts: list[str], batch_size: int = 16) -> np.ndarray:
    # Los modelos E5 requieren los prefijos "passage: " / "query: ".
    return embedder().encode(
        [f"passage: {t}" for t in texts], batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True
    )


def embed_queries(texts: list[str]) -> np.ndarray:
    return embedder().encode([f"query: {t}" for t in texts], normalize_embeddings=True)


@lru_cache(maxsize=1)
def client() -> QdrantClient:
    return QdrantClient(url=config.QDRANT_URL, timeout=60, check_compatibility=False)


def build_index(chunks: pd.DataFrame) -> None:
    vecs = embed_passages(chunks["text"].tolist())
    np.save(config.PROCESSED / "chunk_vectors.npy", vecs)
    qc = client()
    if qc.collection_exists(config.QDRANT_COLLECTION):
        qc.delete_collection(config.QDRANT_COLLECTION)
    qc.create_collection(
        config.QDRANT_COLLECTION,
        vectors_config=models.VectorParams(size=vecs.shape[1], distance=models.Distance.COSINE),
    )
    qc.create_payload_index(config.QDRANT_COLLECTION, "version", models.PayloadSchemaType.KEYWORD)
    qc.create_payload_index(config.QDRANT_COLLECTION, "doc", models.PayloadSchemaType.KEYWORD)
    step = 512
    for i in range(0, len(chunks), step):
        part = chunks.iloc[i : i + step]
        qc.upsert(
            config.QDRANT_COLLECTION,
            points=models.Batch(
                ids=part["point_id"].tolist(),
                vectors=vecs[i : i + step].tolist(),
                payloads=[
                    {"chunk_id": r.chunk_id, "version": r.version, "doc": r.doc} for r in part.itertuples()
                ],
            ),
        )


def search(query: str, k: int = config.TOP_K, version: str | None = None, doc: str | None = None) -> list[dict]:
    """Búsqueda por similitud. `version`/`doc` solo se usan en la construcción del
    conjunto de evaluación (para localizar pasajes de oro), nunca en el sistema RAG."""
    qv = embed_queries([query])[0].tolist()
    must = []
    if version:
        must.append(models.FieldCondition(key="version", match=models.MatchValue(value=version)))
    if doc:
        must.append(models.FieldCondition(key="doc", match=models.MatchValue(value=doc)))
    res = client().query_points(
        config.QDRANT_COLLECTION,
        query=qv,
        limit=k,
        query_filter=models.Filter(must=must) if must else None,
        with_payload=True,
    )
    return [{"point_id": p.id, "score": p.score, **p.payload} for p in res.points]
