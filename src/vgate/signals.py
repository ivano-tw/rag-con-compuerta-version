"""Definición operativa de las señales de la compuerta.

S_suf  suficiencia del contexto: puntuación del autorater (Joren et al., 2024).
S_conf confianza del generador: confianza verbalizada (0-1). La API no expone
       log-probabilidades; se documenta como limitación.
S_ver  alineación de versión: para cada fragmento recuperado se calcula la fracción
       de sus segmentos cuyo texto fuente existe, idéntico, en el mismo documento de
       la versión solicitada. Un fragmento de 3.13 idéntico en 3.12 sigue siendo válido
       para una consulta sobre 3.12; uno con segmentos nuevos o modificados, no.
       Se agregan:
         ver_w     media ponderada por rango (pesos 1/log2(r+2)) sobre los top-k
         ver_top1  alineación del primer fragmento
         ver_cited media sobre los fragmentos citados por el generador
"""
from __future__ import annotations

import math
import re

from . import config

_VER = re.compile(r"\b3\.(1[0-9])\b")


def requested_version(question: str) -> str:
    """Versión solicitada en la consulta; si no hay, la versión por defecto."""
    found = [f"3.{m}" for m in _VER.findall(question)]
    found = [v for v in found if v in config.VERSIONS]
    return found[0] if found else config.DEFAULT_VERSION


def chunk_alignment(chunk_hashes: list[str], doc: str, target: str, presence: dict) -> float:
    if not chunk_hashes:
        return 0.0
    pres = presence[target]
    return sum((doc, h) in pres for h in chunk_hashes) / len(chunk_hashes)


def version_features(hits: list[dict], target: str, presence: dict, hashes: dict, cited: list[int]) -> dict:
    aligns = [chunk_alignment(list(hashes[h["chunk_id"]]), h["doc"], target, presence) for h in hits]
    w = [1 / math.log2(r + 2) for r in range(len(hits))]
    cited_idx = [c - 1 for c in cited if 1 <= c <= len(hits)]
    return {
        "ver_w": sum(a * wi for a, wi in zip(aligns, w)) / sum(w),
        "ver_top1": aligns[0],
        "ver_cited": (sum(aligns[i] for i in cited_idx) / len(cited_idx)) if cited_idx else sum(aligns) / len(aligns),
        "frac_same_version": sum(h["version"] == target for h in hits) / len(hits),
        "aligns": aligns,
    }
