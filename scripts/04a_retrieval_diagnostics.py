"""Paso 4a: diagnóstico de recuperación y de la señal de versión (sin LLM, sin costo).

Mide sobre el conjunto de evaluación si la recuperación ciega a la versión trae el
pasaje de oro, con qué frecuencia el primer fragmento pertenece a otra versión o está
desalineado, y cómo se distribuye S_ver. Salida: results/retrieval_only.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from vgate import config, corpus, rag, signals  # noqa: E402

qs = pd.read_csv(config.PROCESSED / "questions.csv", dtype={"target_version": str, "other_version": str})
chunks = corpus.load_chunks()
ci = chunks.set_index("chunk_id")
hashes = ci["src_hashes"].to_dict()
presence = corpus.load_presence()

rows = []
for r in qs.itertuples():
    hits = rag.retrieve(r.question, chunks)
    target = signals.requested_version(r.question)
    vf = signals.version_features(hits, target, presence, hashes, [])
    ids = [h["chunk_id"] for h in hits]
    gold_doc = r.gold_chunk_id.split(":")[1]
    rows.append(
        {
            "qid": r.qid,
            "pair": r.pair,
            "target_version": r.target_version,
            "target_is_new": r.target_version == r.pair.split("->")[1],
            "parsed_ok": target == r.target_version,
            "gold_rank": ids.index(r.gold_chunk_id) + 1 if r.gold_chunk_id in ids else None,
            "gold_doc_in_topk": any(h["doc"] == gold_doc for h in hits),
            "gold_frac_es": float(ci.at[r.gold_chunk_id, "frac_es"]),
            "top1_version": hits[0]["version"],
            "top1_other_version": hits[0]["version"] != r.target_version,
            "top1_misaligned": vf["aligns"][0] < 1,
            "any_misaligned_topk": any(a < 1 for a in vf["aligns"]),
            "ver_w": vf["ver_w"],
            "ver_top1": vf["ver_top1"],
            "frac_same_version": vf["frac_same_version"],
            "retrieved": ids,
        }
    )

df = pd.DataFrame(rows)
df.to_parquet(config.RESULTS / "retrieval_only.parquet", index=False)


def summary(d: pd.DataFrame) -> dict:
    return {
        "n": int(len(d)),
        "gold_recall_at_8": round(float(d["gold_rank"].notna().mean()), 3),
        "gold_mrr": round(float((1 / d["gold_rank"]).fillna(0).mean()), 3),
        "gold_doc_in_top8": round(float(d["gold_doc_in_topk"].mean()), 3),
        "top1_from_other_version": round(float(d["top1_other_version"].mean()), 3),
        "top1_misaligned": round(float(d["top1_misaligned"].mean()), 3),
        "any_misaligned_in_top8": round(float(d["any_misaligned_topk"].mean()), 3),
        "mean_frac_same_version_top8": round(float(d["frac_same_version"].mean()), 3),
        "mean_ver_w": round(float(d["ver_w"].mean()), 3),
    }


out = {
    "overall": summary(df),
    "version_parse_accuracy": round(float(df["parsed_ok"].mean()), 3),
    "by_pair": {k: summary(g) for k, g in df.groupby("pair")},
    "by_target": {("nueva" if k else "anterior"): summary(g) for k, g in df.groupby("target_is_new")},
    "by_gold_language": {
        "gold_mayoritariamente_es": summary(df[df.gold_frac_es >= 0.5]),
        "gold_mayoritariamente_en": summary(df[df.gold_frac_es < 0.5]),
    },
}
(config.RESULTS / "retrieval_only.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(json.dumps(out, indent=2, ensure_ascii=False))
