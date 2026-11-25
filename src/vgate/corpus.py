"""Conformación del corpus versionado a partir de python-docs-es (.po).

Cada entrada de un archivo .po es un segmento: `msgid` (texto fuente en inglés) y
`msgstr` (traducción al español). Se conserva la traducción cuando existe y no está
marcada como *fuzzy*; en caso contrario se usa el texto fuente y se marca `lang=en`.

Salidas (data/processed):
  segments.parquet  un registro por segmento y versión
  chunks.parquet    fragmentos (agrupación de segmentos consecutivos) a indexar
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd
import polib

from . import config

_ws = re.compile(r"\s+")


def norm(text: str) -> str:
    return _ws.sub(" ", text).strip()


def h(text: str) -> str:
    return hashlib.sha1(norm(text).encode()).hexdigest()[:16]


def _line(entry: polib.POEntry) -> int:
    try:
        return int(entry.occurrences[0][1])
    except (IndexError, ValueError):
        return 0


def load_segments(sections: list[str] | None = None) -> pd.DataFrame:
    """Lee todos los .po de las versiones configuradas.

    `sections=None` lee el corpus completo (para estadísticas); en caso contrario solo
    las carpetas indicadas (p. ej. ["library"]).
    """
    rows = []
    for v in config.VERSIONS:
        base = config.RAW / f"py{v}"
        for f in sorted(base.rglob("*.po")):
            rel = f.relative_to(base).with_suffix("").as_posix()
            section = rel.split("/")[0] if "/" in rel else "_root"
            if sections is not None and section not in sections:
                continue
            for e in polib.pofile(str(f)):
                if e.obsolete or not e.msgid.strip():
                    continue
                translated = bool(e.msgstr.strip()) and "fuzzy" not in e.flags
                rows.append(
                    {
                        "version": v,
                        "doc": rel,
                        "section": section,
                        "line": _line(e),
                        "msgid": e.msgid,
                        "text": e.msgstr if translated else e.msgid,
                        "lang": "es" if translated else "en",
                        "src_hash": h(e.msgid),
                    }
                )
    return pd.DataFrame(rows)


def chunk_segments(seg: pd.DataFrame, max_chars: int = config.CHUNK_MAX_CHARS) -> pd.DataFrame:
    """Agrupa segmentos consecutivos del mismo documento y versión en fragmentos."""
    chunks = []
    for (v, doc), g in seg.sort_values(["version", "doc", "line"]).groupby(["version", "doc"], sort=False):
        buf, hashes, langs, start = [], [], [], None
        idx = 0

        def flush():
            nonlocal buf, hashes, langs, idx, start
            if not buf:
                return
            text = "\n".join(buf)
            chunks.append(
                {
                    "chunk_id": f"{v}:{doc}:{idx}",
                    "version": v,
                    "doc": doc,
                    "start_line": start,
                    "text": text,
                    "src_hashes": list(hashes),
                    "frac_es": sum(l == "es" for l in langs) / len(langs),
                }
            )
            idx += 1
            buf, hashes, langs, start = [], [], [], None

        for r in g.itertuples():
            if buf and sum(len(b) for b in buf) + len(r.text) > max_chars:
                flush()
            if start is None:
                start = r.line
            buf.append(r.text)
            hashes.append(r.src_hash)
            langs.append(r.lang)
        flush()
    out = pd.DataFrame(chunks)
    out["point_id"] = range(len(out))
    return out


def version_presence(seg: pd.DataFrame) -> dict[str, set[tuple[str, str]]]:
    """Para cada versión, el conjunto de (doc, hash del texto fuente) presentes.

    Es la base de la señal de alineación de versión: un segmento recuperado de otra
    versión sigue siendo válido para la versión solicitada si su texto fuente existe,
    idéntico, en el mismo documento de esa versión.
    """
    return {v: set(zip(g["doc"], g["src_hash"])) for v, g in seg.groupby("version")}


def build(sections: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    sections = sections or config.CORPUS_SECTIONS
    seg = load_segments(sections)
    chunks = chunk_segments(seg)
    seg.to_parquet(config.PROCESSED / "segments.parquet", index=False)
    chunks.to_parquet(config.PROCESSED / "chunks.parquet", index=False)
    return seg, chunks


def load_chunks() -> pd.DataFrame:
    return pd.read_parquet(config.PROCESSED / "chunks.parquet")


def load_presence() -> dict[str, set[tuple[str, str]]]:
    return version_presence(pd.read_parquet(config.PROCESSED / "segments.parquet"))
