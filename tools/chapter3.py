"""Contenido del Capítulo 3 (Desarrollo), generado a partir de los artefactos del prototipo."""
from __future__ import annotations

import json

import pandas as pd


def fmt(x: float, d: int = 1) -> str:
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", " ")


def n(x: int) -> str:
    return f"{x:,}".replace(",", " ")


def blocks(root) -> list[tuple]:
    cs = json.loads((root / "results/corpus_stats.json").read_text())
    q = pd.read_csv(root / "data/processed/questions.csv")
    changes = pd.read_csv(root / "data/processed/changes_all.csv")
    drafts = pd.read_csv(root / "data/processed/questions_draft_all.csv")
    fc, sub, cv = cs["full_corpus"], cs["indexed_subset"], cs["cross_version"]
    total_chunks = sum(sub[v]["chunks"] for v in ("3.12", "3.13", "3.14"))
    by_pair_type = q.groupby(["pair", "change_type"]).size().unstack(fill_value=0)

    B: list[tuple] = []
    B.append(("p", "Este capítulo describe la implementación preliminar del sistema propuesto. Se trata de una prueba de concepto que recorre el pipeline completo, desde la conformación del corpus hasta la evaluación estadística de la compuerta, sobre un subconjunto del corpus y con un conjunto de evaluación construido de forma semiautomática que aún debe ser validado por anotadores humanos. Su propósito es verificar la viabilidad técnica del diseño, fijar la definición operativa de cada señal y obtener una primera lectura de los resultados que oriente la versión final del experimento."))
    B.append(("h4", "Entorno de desarrollo y componentes"))
    B.append(("p", "La implementación se desarrolló en Python 3.11 dentro de un ambiente conda denominado *maestria*, en un equipo Apple M1 Pro con 16 GB de memoria. El código se organizó como un paquete (vgate) con un módulo por responsabilidad y cinco scripts que ejecutan las etapas en orden, de modo que cada resultado intermedio queda persistido y el experimento es reproducible. La Tabla 1 resume los componentes empleados."))
    B.append(("caption", "Tabla 1. Componentes del prototipo"))
    B.append(("table", ([
        ["Componente", "Elección", "Justificación"],
        ["Corpus", "python-docs-es, ramas 3.12, 3.13 y 3.14 (archivos .po)", "Traducción oficial al español con metadatos de versión por rama"],
        ["Fragmentación", "Segmentos consecutivos del mismo documento, máx. 900 caracteres", "Conserva el contexto local sin mezclar documentos ni versiones"],
        ["Embeddings", "intfloat/multilingual-e5-base (local)", "Modelo multilingüe; el corpus mezcla español e inglés"],
        ["Base vectorial", "Qdrant 1.12 en Docker", "Payload con versión y documento; filtros para construir el conjunto de evaluación"],
        ["Generador", "Claude Haiku 5.5 (API)", "Bajo costo; idéntico para línea base y sistema propuesto"],
        ["Autorater de suficiencia", "Claude Haiku 5.5 (API)", "Clasificador de contexto suficiente según Joren et al. (2024)"],
        ["Constructor y juez", "Claude Opus 5.5 (API)", "Mayor capacidad para redactar preguntas y calificar respuestas"],
        ["Estadística", "scikit-learn, SciPy, statsmodels", "Regresión logística, McNemar exacto, corrección de Holm, bootstrap"],
    ], [1800, 3400, 3800])))
    B.append(("p", "Todas las llamadas al modelo de lenguaje usan salidas estructuradas validadas contra un esquema y se almacenan en una caché en disco, de forma que volver a ejecutar un paso no repite llamadas ni costos y los resultados del juez quedan auditables."))

    B.append(("h3", "Conformación del corpus versionado"))
    B.append(("p", f"Cada rama de python-docs-es contiene los archivos de traducción (.po) de la documentación oficial. Cada entrada de un archivo .po es un segmento con el texto fuente en inglés (msgid) y su traducción (msgstr). Se tomó la traducción cuando existe y no está marcada como dudosa (fuzzy); en caso contrario se conservó el texto fuente, marcándolo como segmento en inglés. Cada segmento hereda la versión de la rama y la ruta del documento. El corpus completo suma {n(fc['total_segments'])} segmentos, cifra consistente con los aproximadamente 194 000 estimados en el anteproyecto (Tabla 2)."))
    B.append(("caption", "Tabla 2. Tamaño y cobertura de traducción del corpus por versión"))
    rows = [["Versión", "Segmentos (corpus completo)", "% traducido", "Segmentos library/", "% traducido library/", "Fragmentos indexados"]]
    for v in ("3.12", "3.13", "3.14"):
        rows.append([v, n(fc[v]["segments"]), fmt(fc[v]["translated_pct"]), n(sub[v]["segments"]), fmt(sub[v]["translated_pct"]), n(sub[v]["chunks"])])
    B.append(("table", (rows, [1000, 1900, 1300, 1700, 1600, 1500])))
    B.append(("p", f"Para esta versión preliminar se indexó únicamente la sección library/ (referencia de la biblioteca estándar), que concentra la descripción de módulos, funciones y parámetros afectados por los cambios entre versiones. Los segmentos se agruparon en {n(total_chunks)} fragmentos de unos {sub['3.13']['avg_chunk_chars']} caracteres en promedio. La sección whatsnew/ se excluyó del índice porque se usa como fuente de los casos de evaluación; indexarla permitiría que el sistema respondiera leyendo directamente el registro de cambios."))
    B.append(("h4", "Hallazgo: el contenido que cambia entre versiones no está traducido"))
    a, b = cv["3.12->3.13"], cv["3.13->3.14"]
    B.append(("p", f"Al comparar versiones consecutivas por el texto fuente de cada segmento se observa que el {fmt(a['segments_identical_pct'])} % de los segmentos de library/ en 3.13 es idéntico a 3.12, y el {fmt(b['segments_identical_pct'])} % de 3.14 es idéntico a 3.13. Los segmentos nuevos o modificados ({n(a['segments_new_or_modified'])} y {n(b['segments_new_or_modified'])}, respectivamente) son justamente los que distinguen una versión de otra, y su cobertura de traducción es muy baja: {fmt(a['translated_pct_new_or_modified'])} % en 3.13 y {fmt(b['translated_pct_new_or_modified'])} % en 3.14, frente a {fmt(a['translated_pct_unchanged'])} % y {fmt(b['translated_pct_unchanged'])} % en los segmentos sin cambios. En consecuencia, en un corpus «en español» la información sensible a la versión aparece mayoritariamente en inglés. Este rasgo tiene dos implicaciones: favorece que una consulta en español recupere la descripción traducida de una versión anterior en lugar del texto nuevo en inglés, y constituye una variable de confusión que deberá controlarse en la versión final (por ejemplo, reportando resultados separados por idioma del pasaje de oro)."))
    B.append(("p", "Además, los archivos .po no contienen las firmas de las funciones ni el número de versión de las directivas versionadded y versionchanged de Sphinx: solo se conserva el cuerpo del aviso (por ejemplo, «Se agregó el parámetro include_hidden»). El texto recuperado, por tanto, no indica por sí mismo a qué versión pertenece, lo que confirma empíricamente la premisa del problema planteado en el Capítulo 1."))

    B.append(("h3", "Indexación y recuperación"))
    B.append(("p", "Los fragmentos se representaron con el modelo multilingual-e5-base (prefijo «passage:» para fragmentos y «query:» para consultas, vectores normalizados de 768 dimensiones) y se indexaron en una colección de Qdrant con distancia coseno. Cada punto almacena en su payload la versión, el documento y el identificador del fragmento. La recuperación del sistema devuelve los k = 8 fragmentos más similares a la consulta sin ningún filtro de versión. Esta decisión es deliberada: la contribución de la tesis no interviene sobre qué se recupera sino sobre si el sistema debe responder, por lo que ambos sistemas comparten exactamente la misma recuperación y la misma respuesta generada. Los filtros por versión y documento de Qdrant se emplean solo durante la construcción del conjunto de evaluación, para localizar pasajes de oro."))

    B.append(("h3", "Construcción del conjunto de evaluación"))
    B.append(("p", f"Los casos de conflicto se derivaron del registro oficial de cambios: whatsnew/3.13 (cambios de 3.12 a 3.13) y whatsnew/3.14 (cambios de 3.13 a 3.14). Estos documentos no están traducidos, por lo que se usó su texto fuente. El procedimiento fue el siguiente:"))
    for t in [
        f"**Extracción de cambios.** Se seleccionaron las entradas del registro que mencionan un objeto de la API mediante roles de Sphinx (:func:, :meth:, :class:, :mod:, etc.) y se asociaron a la página de referencia de su módulo existente en ambas versiones. Se obtuvieron {n(len(changes))} cambios candidatos ({n((changes.pair == '3.12->3.13').sum())} para 3.12→3.13 y {n((changes.pair == '3.13->3.14').sum())} para 3.13→3.14) en {changes.doc.nunique()} módulos.",
        "**Muestreo estratificado.** Se muestrearon 100 cambios por par de versiones con semilla fija, con un máximo de tres cambios por módulo para evitar que pocos módulos dominen el conjunto.",
        "**Pasajes candidatos.** Para cada cambio se recuperaron los cuatro fragmentos más similares de la página del módulo en la versión nueva y otros cuatro en la versión anterior.",
        "**Redacción asistida.** Un modelo constructor (Claude Opus 5.5) recibió el cambio y los pasajes, decidió si el cambio produce una diferencia observable respaldada por un pasaje, clasificó el tipo de cambio (agregado, modificado, obsoleto, eliminado) y redactó hasta dos preguntas en español que nombran explícitamente la versión: una sobre la versión nueva y, solo si un pasaje lo respalda de forma explícita, otra sobre la versión anterior. Cada pregunta incluye la respuesta de referencia, la respuesta que correspondería a la otra versión y el pasaje de oro.",
        f"**Resultado y agrupación.** De los 200 cambios procesados, el constructor consideró utilizables {q.change_id.nunique()}, de los que se obtuvieron {n(len(drafts))} preguntas. El balance por par de versiones (meta de 90 por par para llegar a 180) no recortó ninguna, porque el presupuesto de API asignado a esta etapa se agotó antes de procesar más cambios; el conjunto preliminar queda en {n(len(q))} preguntas. Las preguntas derivadas de un mismo cambio se mantienen juntas y el identificador del cambio se usa como conglomerado en la partición y en el bootstrap.",
    ]:
        B.append(("bullet", t))
    B.append(("caption", "Tabla 3. Distribución del conjunto de evaluación preliminar por par de versiones y tipo de cambio"))
    types = [c for c in ["added", "changed", "deprecated", "removed", "other"] if c in by_pair_type.columns]
    names = {"added": "Agregado", "changed": "Modificado", "deprecated": "Obsoleto", "removed": "Eliminado", "other": "Otro"}
    rows = [["Par de versiones"] + [names[t] for t in types] + ["Total"]]
    for pair, r in by_pair_type.iterrows():
        rows.append([pair.replace("->", " → ")] + [str(int(r[t])) for t in types] + [str(int(r.sum()))])
    rows.append(["Total"] + [str(int(by_pair_type[t].sum())) for t in types] + [str(len(q))])
    B.append(("table", (rows, None)))
    B.append(("p", "El conjunto obtenido es un borrador. Para cumplir el alcance definido, cada pregunta deberá ser validada por dos anotadores independientes; el prototipo genera para ello una plantilla (annotation_template.csv) con la pregunta, la respuesta de referencia y el pasaje de oro, y columnas para el veredicto y comentarios de cada anotador."))

    B.append(("h3", "Generación y definición operativa de las señales"))
    B.append(("p", "Para cada pregunta, el sistema ejecuta la misma secuencia: recuperación de los ocho fragmentos, generación de la respuesta y cálculo de tres señales. El generador recibe los fragmentos numerados sin la etiqueta de versión, reproduciendo un RAG ciego a la versión, y devuelve la respuesta, los números de los fragmentos que utilizó y una confianza entre 0 y 1."))
    B.append(("bullet", "**Suficiencia del contexto (S_suf).** Un autorater (Claude Haiku 5.5) recibe la pregunta y el contexto, sin la respuesta, y estima la probabilidad de que el contexto contenga la información necesaria para responder de forma definitiva, siguiendo la definición de contexto suficiente de Joren et al. (2024)."))
    B.append(("bullet", "**Confianza del generador (S_conf).** Confianza verbalizada por el generador. La API utilizada no expone probabilidades por token, por lo que se adoptó esta aproximación; la versión final evaluará la autoconsistencia entre varias muestras."))
    B.append(("bullet", "**Alineación de versión (S_ver).** La versión solicitada se extrae de la consulta con una expresión regular (por defecto, la versión más reciente). Para cada fragmento recuperado se calcula la fracción de sus segmentos cuyo texto fuente existe, idéntico, en el mismo documento de la versión solicitada. Así, un fragmento de 3.13 que no cambió respecto de 3.12 se considera válido para una consulta sobre 3.12, mientras que uno que contiene segmentos nuevos o modificados no lo es. Esta formulación evita penalizar la recuperación de texto idéntico entre versiones, que es la situación más frecuente."))
    B.append(("p", "A partir de la alineación por fragmento se derivan tres variables: ver_w, la media ponderada por rango con pesos 1/log2(r+2); ver_top1, la alineación del primer fragmento; y ver_cited, la alineación media de los fragmentos que el generador declaró haber usado. Esta última vincula la señal con la evidencia efectiva de la respuesta."))

    B.append(("h3", "Regla de integración de la compuerta y línea base"))
    B.append(("p", "Cada compuerta asigna a cada pregunta una puntuación de fiabilidad y el sistema responde a las preguntas con mayor puntuación hasta alcanzar la cobertura fijada, absteniéndose en el resto. La regla de integración es una regresión logística que estima la probabilidad de que la respuesta sea correcta:"))
    B.append(("bullet", "**Línea base:** regresión logística sobre [S_suf, S_conf]."))
    B.append(("bullet", "**Sistema propuesto:** regresión logística sobre [S_suf, S_conf, ver_w, ver_top1, ver_cited]."))
    B.append(("p", "Ambas comparten la recuperación, el generador y la respuesta; la única diferencia es la señal de versión, por lo que cualquier diferencia en las métricas es atribuible a ella. Como referencia sin entrenamiento se reportan además dos reglas multiplicativas: S_suf·S_conf y S_suf·S_conf·ver_cited."))

    B.append(("h3", "Protocolo de evaluación"))
    B.append(("p", "Un juez (Claude Opus 5.5) clasifica cada respuesta comparándola con la respuesta de referencia y con la respuesta de la otra versión en cuatro categorías: correcta, versión incorrecta, incorrecta por otro motivo y sin respuesta. Las preguntas se dividen por conglomerado de cambio en una partición de calibración (40 %), con la que se ajustan las regresiones, y una de prueba (60 %), sobre la que se reportan todas las métricas; ningún cambio aparece en ambas particiones. Se calculan la exactitud selectiva, la tasa de respuesta con versión incorrecta, la curva de cobertura frente a riesgo y su área (AURC) a coberturas de 50 %, 70 % y 90 %. La comparación con la línea base usa la prueba exacta de McNemar unilateral sobre la variable pareada «entregó una respuesta de versión incorrecta», con corrección de Holm por los tres niveles de cobertura, y un intervalo de confianza al 95 % de la diferencia obtenido por bootstrap por conglomerados de cambio (2 000 réplicas)."))
    return B
