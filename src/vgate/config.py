"""Configuración central del prototipo (rutas, modelos y parámetros).

Todos los valores pueden sobreescribirse con variables de entorno o un archivo .env
en la raíz de `implementation/`.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
RESULTS = ROOT / "results"

VERSIONS = ["3.12", "3.13", "3.14"]
# Pares de versiones consecutivas: (anterior, nueva). El registro de cambios
# whatsnew/<nueva>.rst describe qué cambió respecto de <anterior>.
VERSION_PAIRS = [("3.12", "3.13"), ("3.13", "3.14")]
DEFAULT_VERSION = os.getenv("VGATE_DEFAULT_VERSION", "3.14")

# Subconjunto del corpus indexado en esta versión preliminar.
CORPUS_SECTIONS = os.getenv("VGATE_SECTIONS", "library").split(",")
CHUNK_MAX_CHARS = int(os.getenv("VGATE_CHUNK_MAX_CHARS", "900"))

# Recuperación
EMBED_MODEL = os.getenv("VGATE_EMBED_MODEL", "intfloat/multilingual-e5-base")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "python_docs_es")
TOP_K = int(os.getenv("VGATE_TOP_K", "8"))

# LLMs (API de Claude)
GENERATOR_MODEL = os.getenv("VGATE_GENERATOR_MODEL", "claude-haiku-5-5")
AUTORATER_MODEL = os.getenv("VGATE_AUTORATER_MODEL", "claude-haiku-5-5")
# Construcción del conjunto de evaluación y juez: modelo más capaz.
BUILDER_MODEL = os.getenv("VGATE_BUILDER_MODEL", "claude-opus-5-5")
JUDGE_MODEL = os.getenv("VGATE_JUDGE_MODEL", "claude-opus-5-5")
LLM_CONCURRENCY = int(os.getenv("VGATE_LLM_CONCURRENCY", "6"))

# Si es True, el contexto entregado al generador incluye la etiqueta de versión de
# cada fragmento. Por defecto False: reproduce un RAG "ciego a la versión".
SHOW_VERSION_TO_GENERATOR = os.getenv("VGATE_SHOW_VERSION", "0") == "1"

# Evaluación
SEED = int(os.getenv("VGATE_SEED", "13"))
CALIBRATION_FRACTION = float(os.getenv("VGATE_CALIB_FRACTION", "0.4"))
COVERAGE_LEVELS = [0.5, 0.7, 0.9]
TARGET_QUESTIONS = int(os.getenv("VGATE_TARGET_QUESTIONS", "180"))

for _d in (PROCESSED, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)
