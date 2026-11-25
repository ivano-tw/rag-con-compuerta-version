"""Paso 4: ejecuta el RAG sobre el conjunto de evaluación y calcula las señales.

Para cada pregunta: recuperación (top-k sin filtro de versión) -> generación ->
autorater de suficiencia -> señales de versión -> juez. Guarda results/runs.parquet
y un diagnóstico de recuperación en results/retrieval_diagnostics.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from vgate import config, corpus, llm, rag, signals  # noqa: E402

qs = pd.read_csv(config.PROCESSED / "questions.csv", dtype={"target_version": str, "other_version": str})
chunks = corpus.load_chunks()
hashes = chunks.set_index("chunk_id")["src_hashes"].to_dict()
presence = corpus.load_presence()

# Recuperación (local, secuencial: el modelo de embeddings no es thread-safe en MPS)
retrieved = {r.qid: rag.retrieve(r.question, chunks) for r in qs.itertuples()}


def process(row) -> dict:
    row = row._asdict()
    hits = retrieved[row["qid"]]
    target = signals.requested_version(row["question"])
    gen = rag.generate(row["question"], hits)
    suf = rag.rate_sufficiency(row["question"], hits)
    answer = gen.answer if gen else ""
    jud = rag.judge(row, answer) if gen else None
    vf = signals.version_features(hits, target, presence, hashes, gen.cited if gen else [])
    ids = [h["chunk_id"] for h in hits]
    return {
        **row,
        "parsed_version": target,
        "retrieved": ids,
        "retrieved_versions": [h["version"] for h in hits],
        "gold_rank": ids.index(row["gold_chunk_id"]) + 1 if row["gold_chunk_id"] in ids else None,
        "answer": answer,
        "cited": gen.cited if gen else [],
        "confidence": gen.confidence if gen else 0.0,
        "suf_sufficient": bool(suf.sufficient) if suf else False,
        "suf_score": suf.score if suf else 0.0,
        "label": jud.label if jud else "no_answer",
        "judge_rationale": jud.rationale if jud else "",
        **{k: v for k, v in vf.items()},
    }


runs = pd.DataFrame(llm.pmap(process, list(qs.itertuples(index=False)), desc="RAG"))
runs.to_parquet(config.RESULTS / "runs.parquet", index=False)

top1_other = runs.apply(lambda r: r["retrieved_versions"][0] != r["target_version"], axis=1)
diag = {
    "n": int(len(runs)),
    "version_parse_accuracy": float((runs["parsed_version"] == runs["target_version"]).mean()),
    "gold_recall_at_k": float(runs["gold_rank"].notna().mean()),
    "gold_mrr": float((1 / runs["gold_rank"]).fillna(0).mean()),
    "top1_from_other_version": float(top1_other.mean()),
    "top1_misaligned": float((runs["ver_top1"] < 1).mean()),
    "mean_frac_same_version_topk": float(runs["frac_same_version"].mean()),
    "label_distribution": runs["label"].value_counts().to_dict(),
    "label_by_target": runs.groupby(["pair", "target_version"])["label"].value_counts().unstack(fill_value=0).to_dict("index"),
    "suf_sufficient_rate": float(runs["suf_sufficient"].mean()),
    "suf_sufficient_rate_when_wrong_version": float(runs.loc[runs.label == "wrong_version", "suf_sufficient"].mean())
    if (runs.label == "wrong_version").any() else None,
    "token_usage": llm.usage_summary(),
}
diag["label_by_target"] = {f"{k[0]}|{k[1]}": v for k, v in diag["label_by_target"].items()}
(config.RESULTS / "retrieval_diagnostics.json").write_text(json.dumps(diag, indent=2, ensure_ascii=False))
print(json.dumps(diag, indent=2, ensure_ascii=False))
