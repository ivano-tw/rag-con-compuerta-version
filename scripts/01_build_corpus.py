"""Paso 1: conformación del corpus y estadísticas descriptivas.

Uso: python scripts/01_build_corpus.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgate import config, corpus  # noqa: E402

full = corpus.load_segments(sections=None)
stats = {"full_corpus": {}, "indexed_subset": {}, "cross_version": {}}

for v, g in full.groupby("version"):
    stats["full_corpus"][v] = {
        "segments": int(len(g)),
        "translated_pct": round(100 * (g["lang"] == "es").mean(), 1),
        "docs": int(g["doc"].nunique()),
    }
stats["full_corpus"]["total_segments"] = int(len(full))

seg, chunks = corpus.build()
for v in config.VERSIONS:
    s, c = seg[seg.version == v], chunks[chunks.version == v]
    stats["indexed_subset"][v] = {
        "segments": int(len(s)),
        "translated_pct": round(100 * (s["lang"] == "es").mean(), 1),
        "docs": int(s["doc"].nunique()),
        "chunks": int(len(c)),
        "avg_chunk_chars": int(c["text"].str.len().mean()),
    }

# Solapamiento entre versiones consecutivas (texto fuente idéntico en el mismo doc)
pres = corpus.version_presence(seg)
for old, new in config.VERSION_PAIRS:
    new_seg = seg[seg.version == new]
    keys = list(zip(new_seg["doc"], new_seg["src_hash"]))
    is_new = [k not in pres[old] for k in keys]
    new_only = new_seg[is_new]
    stats["cross_version"][f"{old}->{new}"] = {
        "segments_identical_pct": round(100 * (1 - sum(is_new) / len(keys)), 1),
        "segments_new_or_modified": int(sum(is_new)),
        "translated_pct_new_or_modified": round(100 * (new_only["lang"] == "es").mean(), 1),
        "translated_pct_unchanged": round(100 * (new_seg[[not x for x in is_new]]["lang"] == "es").mean(), 1),
    }

out = config.RESULTS / "corpus_stats.json"
out.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
print(json.dumps(stats, indent=2, ensure_ascii=False))
