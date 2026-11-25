"""Paso 3: conjunto de evaluación derivado del registro oficial de cambios (whatsnew).

Uso: python scripts/03_build_evalset.py [--per-pair 120]
Requiere ANTHROPIC_API_KEY (o credenciales del SDK) y Qdrant con el índice.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vgate import config, evalset, llm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--per-pair", type=int, default=120, help="cambios candidatos por par de versiones")
args = ap.parse_args()

changes = evalset.extract_changes()
changes.to_csv(config.PROCESSED / "changes_all.csv", index=False)
sample = evalset.sample_changes(changes, per_pair=args.per_pair)
print(f"Cambios extraídos: {len(changes)} | muestreados: {len(sample)}")

qs = evalset.build_questions(sample)
qs.to_csv(config.PROCESSED / "questions_draft_all.csv", index=False)
final = evalset.balance(qs, config.TARGET_QUESTIONS)
final.to_csv(config.PROCESSED / "questions.csv", index=False)

# Plantilla para la validación por dos anotadores independientes.
tmpl = final[["qid", "change_id", "target_version", "question", "reference_answer", "gold_chunk_id"]].copy()
for a in ("anotador1", "anotador2"):
    tmpl[f"{a}_valida"] = ""
    tmpl[f"{a}_comentario"] = ""
tmpl.to_csv(config.PROCESSED / "annotation_template.csv", index=False)

print(f"Preguntas generadas: {len(qs)} | conjunto final: {len(final)}")
print(final.groupby(["pair", "change_type"]).size())
print(final.groupby(["pair", "target_version"]).size())
print("Uso de tokens:", llm.usage_summary())
