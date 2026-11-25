"""Construcción del conjunto de evaluación a partir del registro oficial de cambios.

1. Se extraen los cambios candidatos de whatsnew/<nueva>.po (rama <nueva>), que
   describe qué cambió respecto de la versión anterior.
2. Cada cambio se asocia a la página de referencia de su módulo (library/<módulo>).
3. Se recuperan pasajes candidatos de esa página en ambas versiones (búsqueda
   filtrada por versión y documento; el filtro SOLO se usa aquí, no en el sistema).
4. Un LLM constructor (Claude) decide si el cambio sirve, clasifica el tipo de
   cambio, elige el pasaje de oro y redacta preguntas en español con respuesta de
   referencia para la versión objetivo y la respuesta que daría la otra versión.

Las preguntas resultantes son un *borrador*: la tesis exige validación por dos
anotadores independientes (ver annotation_template.csv).
"""
from __future__ import annotations

import random
import re
from typing import Literal

import pandas as pd
import polib
from pydantic import BaseModel, Field

from . import config, index, llm
from .corpus import _line, load_chunks

_ROLE = re.compile(r":(?:mod|func|meth|class|data|attr|exc|const|envvar|option|pdbcmd):`!?~?([\w\.]+)")
_CONTRIB = re.compile(r"\s*\((?:Contributed|Patch|Original patch)[^)]*\)\.?\s*$", re.S)


def _library_docs(version: str) -> set[str]:
    return {p.stem for p in (config.RAW / f"py{version}" / "library").glob("*.po")}


def _map_doc(targets: list[str], heading: str, docs: set[str]) -> str | None:
    best = None
    for t in targets + ([heading] if heading else []):
        parts = t.split(".")
        for n in range(len(parts), 0, -1):
            name = ".".join(parts[:n])
            if name in docs:
                if best is None or len(name) > len(best):
                    best = name
                break
    return best


def extract_changes() -> pd.DataFrame:
    rows = []
    for old, new in config.VERSION_PAIRS:
        po = polib.pofile(str(config.RAW / f"py{new}" / "whatsnew" / f"{new}.po"))
        docs = _library_docs(old) & _library_docs(new)
        heading, section = "", ""
        for e in sorted(po, key=_line):
            txt = e.msgid.strip()
            if len(txt) < 50 and "." not in txt.rstrip(".") and " " not in txt:
                heading = txt.strip("`:!~")
                continue
            if len(txt) < 60 and not txt.endswith("."):
                section = txt
                continue
            if not (60 <= len(txt) <= 700):
                continue
            targets = _ROLE.findall(txt)
            if not targets:
                continue
            doc = _map_doc(targets, heading, docs)
            if not doc:
                continue
            line = e.occurrences[0][1] if e.occurrences else "0"
            rows.append(
                {
                    "change_id": f"{new}:{line}",
                    "pair": f"{old}->{new}",
                    "old": old,
                    "new": new,
                    "doc": f"library/{doc}",
                    "section": section,
                    "change_text": _CONTRIB.sub("", txt).strip(),
                }
            )
    return pd.DataFrame(rows).drop_duplicates("change_id")


def sample_changes(changes: pd.DataFrame, per_pair: int, max_per_doc: int = 3) -> pd.DataFrame:
    rng = random.Random(config.SEED)
    out = []
    for pair, g in changes.groupby("pair"):
        idx = list(g.index)
        rng.shuffle(idx)
        per_doc: dict[str, int] = {}
        for i in idx:
            d = g.at[i, "doc"]
            if per_doc.get(d, 0) >= max_per_doc:
                continue
            per_doc[d] = per_doc.get(d, 0) + 1
            out.append(i)
            if sum(1 for j in out if changes.at[j, "pair"] == pair) >= per_pair:
                break
    return changes.loc[out].reset_index(drop=True)


class DraftQuestion(BaseModel):
    target: Literal["new", "old"] = Field(description="Versión a la que se refiere la pregunta")
    question: str = Field(description="Pregunta en español que menciona explícitamente la versión de Python")
    reference_answer: str = Field(description="Respuesta correcta para la versión objetivo, en español, 1-2 oraciones")
    other_version_answer: str = Field(description="Lo que respondería alguien que usa la documentación de la OTRA versión")
    gold: str = Field(description="Etiqueta del pasaje de oro (p. ej. N2 u O1) de la versión objetivo")


class BuildResult(BaseModel):
    usable: bool
    reason: str
    change_type: Literal["added", "changed", "deprecated", "removed", "other"]
    questions: list[DraftQuestion]


BUILDER_SYSTEM = """Eres un anotador experto en Python que construye un conjunto de evaluación para un sistema RAG sobre la documentación oficial en español de Python.

Recibirás un cambio del registro oficial de cambios entre dos versiones de Python y pasajes candidatos de la documentación de referencia de cada versión.

Tu tarea:
1. Decide si el cambio es utilizable: debe producir una diferencia observable entre las dos versiones (una función, parámetro, comportamiento o valor que existe o funciona distinto) y al menos un pasaje candidato de la versión objetivo debe respaldar la respuesta por sí solo. Si no se cumple, marca usable=false y no generes preguntas.
2. Clasifica el tipo de cambio.
3. Redacta hasta dos preguntas en español:
   - target="new": sobre el comportamiento en la versión nueva, con pasaje de oro N*.
   - target="old": sobre el comportamiento en la versión anterior, con pasaje de oro O*. Solo si un pasaje O* respalda explícitamente la respuesta (no infieras ausencia de una función porque no aparece).
   Cada pregunta debe nombrar explícitamente la versión ("En Python 3.13, ..."), ser natural para un desarrollador, poder responderse solo con el pasaje de oro, y su respuesta debe diferir de la que daría la documentación de la otra versión.
   Escribe la respuesta de referencia y la respuesta de la otra versión en 1-2 oraciones."""


def _candidates(doc: str, version: str, query: str, prefix: str, k: int = 4) -> list[tuple[str, str]]:
    hits = index.search(query, k=k, version=version, doc=doc)
    return [(f"{prefix}{i + 1}", h["chunk_id"]) for i, h in enumerate(hits)]


def build_questions(changes: pd.DataFrame) -> pd.DataFrame:
    chunks = load_chunks().set_index("chunk_id")
    # Búsqueda secuencial: el modelo de embeddings no es seguro entre hilos en MPS.
    cands = {
        r.change_id: (
            _candidates(r.doc, r.new, r.change_text, "N"),
            _candidates(r.doc, r.old, r.change_text, "O"),
        )
        for r in changes.itertuples()
    }

    def one(row) -> list[dict]:
        cn, co = cands[row.change_id]
        label2chunk = dict(cn + co)
        blocks = "\n\n".join(f"[{lab}] (Python {cid.split(':')[0]})\n{chunks.at[cid, 'text']}" for lab, cid in cn + co)
        user = (
            f"Versión anterior: Python {row.old}\nVersión nueva: Python {row.new}\n"
            f"Módulo: {row.doc}\n\nCambio registrado (whatsnew {row.new}):\n{row.change_text}\n\n"
            f"Pasajes candidatos:\n{blocks}"
        )
        res = llm.structured(config.BUILDER_MODEL, BUILDER_SYSTEM, user, BuildResult, effort="medium")
        if res is None or not res.usable:
            return []
        out = []
        for q in res.questions:
            tv = row.new if q.target == "new" else row.old
            gold = label2chunk.get(q.gold.strip("[] "))
            if gold is None or not gold.startswith(tv + ":"):
                continue
            out.append(
                {
                    "change_id": row.change_id,
                    "pair": row.pair,
                    "doc": row.doc,
                    "change_type": res.change_type,
                    "target_version": tv,
                    "other_version": row.old if q.target == "new" else row.new,
                    "question": q.question,
                    "reference_answer": q.reference_answer,
                    "other_version_answer": q.other_version_answer,
                    "gold_chunk_id": gold,
                    "change_text": row.change_text,
                }
            )
        return out

    rows = [r for lst in llm.pmap(one, list(changes.itertuples()), desc="construyendo preguntas") for r in lst]
    df = pd.DataFrame(rows)
    df.insert(0, "qid", [f"q{i:03d}" for i in range(len(df))])
    return df


def balance(questions: pd.DataFrame, target: int) -> pd.DataFrame:
    """Recorta al tamaño objetivo manteniendo el balance por par de versiones y
    sin separar las preguntas de un mismo cambio."""
    rng = random.Random(config.SEED)
    per_pair = target // questions["pair"].nunique()
    keep = []
    for _, g in questions.groupby("pair"):
        cids = list(g["change_id"].unique())
        rng.shuffle(cids)
        n = 0
        for c in cids:
            sub = g[g.change_id == c]
            if n + len(sub) > per_pair:
                continue
            keep.append(sub)
            n += len(sub)
    return pd.concat(keep).sort_values("qid").reset_index(drop=True)
