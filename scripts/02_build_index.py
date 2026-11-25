"""Paso 2: embeddings (multilingual-e5-base) e indexación en Qdrant.

Requiere Qdrant en ejecución (ver README: docker run ... qdrant/qdrant).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgate import corpus, index  # noqa: E402

t = time.time()
chunks = corpus.load_chunks()
index.build_index(chunks)
print(f"Indexados {len(chunks)} fragmentos en {time.time() - t:.0f}s")
for r in index.search("¿Cómo convertir un patrón glob en una expresión regular?", k=5):
    print(round(r["score"], 3), r["chunk_id"])
