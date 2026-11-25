# vgate — RAG con compuerta de suficiencia consciente de versión

Prototipo preliminar de la tesis *Desarrollo de un sistema RAG con compuerta de
suficiencia consciente de versión para documentación técnica* (UIDE, Grupo 6).

## Requisitos

- conda con el ambiente creado en version de python (Python 3.11)
- Docker (para Qdrant)
- Credenciales de la API de Claude en `.env` (ver `.env.example`)

## Instalación

```bash
conda activate <ambiente>
pip install -r requirements.txt
docker run -d --name tesis-qdrant -p 6333:6333 -p 6334:6334 \
  -v "$PWD/data/qdrant_storage:/qdrant/storage" qdrant/qdrant:v1.12.4
for v in 3.12 3.13 3.14; do
  git clone --depth 1 -b $v https://github.com/python/python-docs-es.git data/raw/py$v
done
cp .env.example .env   # y completar ANTHROPIC_API_KEY
```

## Ejecución

```bash
python scripts/01_build_corpus.py     # corpus + estadísticas  -> results/corpus_stats.json
python scripts/02_build_index.py      # embeddings + Qdrant
python scripts/03_build_evalset.py    # preguntas (LLM)        -> data/processed/questions.csv
python scripts/04_run_rag.py          # RAG + señales + juez   -> results/runs.parquet
python scripts/05_evaluate.py         # compuertas + métricas  -> results/evaluation.json, risk_coverage.png
```

Las llamadas al LLM se guardan en `data/processed/llm_cache.jsonl`; re-ejecutar no
vuelve a facturar las llamadas ya hechas.

## Estructura

```
src/vgate/
  config.py    rutas, modelos y parámetros (sobrescribibles por variables de entorno)
  corpus.py    lectura de .po, fragmentación, presencia de segmentos por versión
  index.py     embeddings E5 + Qdrant
  evalset.py   cambios del whatsnew -> preguntas con pasaje de oro
  llm.py       cliente Claude con salidas estructuradas y caché
  rag.py       recuperación, generación, autorater de suficiencia, juez
  signals.py   señales S_suf, S_conf, S_ver (definición operativa)
  evaluate.py  compuertas, métricas selectivas y pruebas estadísticas
scripts/       pasos 01-05
results/       salidas de los experimentos
```
