"""Pipeline RAG compartido por la línea base y el sistema propuesto.

Ambos sistemas recuperan los mismos fragmentos y producen la misma respuesta; solo
difieren en la compuerta que decide si esa respuesta se entrega o se abstiene. Así la
diferencia observada es atribuible a la señal de versión.
"""
from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from . import config, index, llm


class GeneratedAnswer(BaseModel):
    answer: str = Field(description="Respuesta en español basada solo en el contexto")
    cited: list[int] = Field(description="Números de los fragmentos [n] en los que se basa la respuesta")
    confidence: float = Field(description="Confianza de 0 a 1 en que la respuesta es correcta")


class SufficiencyRating(BaseModel):
    sufficient: bool
    score: float = Field(description="Probabilidad de 0 a 1 de que el contexto sea suficiente")


class Judgement(BaseModel):
    label: Literal["correct", "wrong_version", "incorrect", "no_answer"]
    rationale: str


GENERATOR_SYSTEM = """Eres un asistente que responde preguntas sobre la documentación oficial de Python en español.
Responde únicamente con la información de los fragmentos de contexto numerados. Si el contexto no permite responder, dilo explícitamente.
Indica qué fragmentos usaste y tu confianza (0 a 1) en que la respuesta es correcta."""

AUTORATER_SYSTEM = """Evalúas si un contexto recuperado es SUFICIENTE para responder una pregunta.
El contexto es suficiente si contiene toda la información necesaria para dar una respuesta definitiva a la pregunta tal como está formulada, sin conocimiento externo.
No respondas la pregunta: solo evalúa la suficiencia del contexto."""

JUDGE_SYSTEM = """Eres un evaluador de respuestas sobre documentación versionada de Python.
Compara la respuesta del sistema con la respuesta de referencia para la versión solicitada.
Etiquetas:
- correct: coincide en lo sustancial con la respuesta de referencia.
- wrong_version: describe el comportamiento de otra versión (coincide con la respuesta de la otra versión o contradice la referencia de forma consistente con un cambio de versión).
- incorrect: es incorrecta por otro motivo.
- no_answer: el sistema declara que no puede responder con el contexto dado."""


def format_context(chunks: list[dict], show_version: bool = config.SHOW_VERSION_TO_GENERATOR) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        head = f"[{i}] {c['doc']}" + (f" (Python {c['version']})" if show_version else "")
        parts.append(f"{head}\n{c['text']}")
    return "\n\n".join(parts)


def retrieve(question: str, chunks_df: pd.DataFrame, k: int = config.TOP_K) -> list[dict]:
    hits = index.search(question, k=k)
    texts = chunks_df.set_index("chunk_id")["text"] if "chunk_id" in chunks_df.columns else chunks_df["text"]
    for h in hits:
        h["text"] = texts[h["chunk_id"]]
    return hits


def generate(question: str, chunks: list[dict]) -> GeneratedAnswer | None:
    user = f"Contexto:\n{format_context(chunks)}\n\nPregunta: {question}"
    return llm.structured(config.GENERATOR_MODEL, GENERATOR_SYSTEM, user, GeneratedAnswer, effort="low")


def rate_sufficiency(question: str, chunks: list[dict]) -> SufficiencyRating | None:
    user = f"Pregunta: {question}\n\nContexto:\n{format_context(chunks)}"
    return llm.structured(config.AUTORATER_MODEL, AUTORATER_SYSTEM, user, SufficiencyRating, effort="low")


def judge(row: dict, answer: str) -> Judgement | None:
    user = (
        f"Pregunta (versión solicitada: Python {row['target_version']}): {row['question']}\n\n"
        f"Respuesta de referencia (Python {row['target_version']}): {row['reference_answer']}\n"
        f"Respuesta correspondiente a otra versión (Python {row['other_version']}): {row['other_version_answer']}\n\n"
        f"Respuesta del sistema: {answer}"
    )
    return llm.structured(config.JUDGE_MODEL, JUDGE_SYSTEM, user, Judgement, effort="low")
