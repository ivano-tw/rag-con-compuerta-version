"""Contenido del Capítulo 4 (Análisis de resultados).

4.1 Pruebas de concepto: diagnóstico de recuperación y de la señal de versión (sin LLM).
4.2 Análisis de resultados: si existe results/evaluation.json se incluyen las métricas
    de la compuerta; si no, se documenta el estado pendiente.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from chapter3 import fmt


def pct(x: float) -> str:
    return fmt(100 * x) + " %"


def retrieval_figure(root, ro: dict) -> None:
    groups = {
        "Global": ro["overall"],
        "Pasaje de oro\nen español": ro["by_gold_language"]["gold_mayoritariamente_es"],
        "Pasaje de oro\nen inglés": ro["by_gold_language"]["gold_mayoritariamente_en"],
        "Pregunta sobre\nversión nueva": ro["by_target"]["nueva"],
        "Pregunta sobre\nversión anterior": ro["by_target"]["anterior"],
    }
    metrics = [
        ("gold_recall_at_8", "Pasaje de oro en top-8"),
        ("top1_from_other_version", "Top-1 de otra versión"),
        ("top1_misaligned", "Top-1 desalineado (S_ver < 1)"),
    ]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    w = 0.26
    colors = ["#2a6fbb", "#e08a2c", "#c23b3b"]
    for i, (k, lab) in enumerate(metrics):
        xs = [j + (i - 1) * w for j in range(len(groups))]
        vals = [100 * g[k] for g in groups.values()]
        ax.bar(xs, vals, w, label=lab, color=colors[i])
        for x, v in zip(xs, vals):
            ax.text(x, v + 1, f"{v:.0f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(groups)), list(groups.keys()), fontsize=8)
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(root / "results/retrieval_diagnostics.png", dpi=150)
    plt.close(fig)


def blocks(root, img_risk, img_retr) -> tuple[list[tuple], list[tuple]]:
    ro = json.loads((root / "results/retrieval_only.json").read_text())
    o = ro["overall"]
    es, en = ro["by_gold_language"]["gold_mayoritariamente_es"], ro["by_gold_language"]["gold_mayoritariamente_en"]
    nu, an = ro["by_target"]["nueva"], ro["by_target"]["anterior"]
    ev_path = root / "results/evaluation.json"
    ev = json.loads(ev_path.read_text()) if ev_path.exists() else None
    diag_path = root / "results/retrieval_diagnostics.json"
    diag = json.loads(diag_path.read_text()) if diag_path.exists() else None

    A: list[tuple] = []
    A.append(("p", f"Las pruebas de concepto se ejecutaron sobre el conjunto de evaluación preliminar de {o['n']} preguntas descrito en el Capítulo 3. Esta sección analiza primero el comportamiento de la recuperación ciega a la versión y de la señal de alineación, que no dependen del modelo de lenguaje, y después las pruebas del pipeline completo."))
    A.append(("h4", "Prueba 1: recuperación ciega a la versión"))
    A.append(("p", f"La versión solicitada se extrajo correctamente de la consulta en el {pct(ro['version_parse_accuracy'])} de los casos, ya que todas las preguntas la mencionan de forma explícita. Aun así, la recuperación por similitud no la aprovecha: en el {pct(o['top1_from_other_version'])} de las preguntas el primer fragmento recuperado pertenece a otra versión, y en promedio solo el {pct(o['mean_frac_same_version_top8'])} de los ocho fragmentos recuperados proviene de la versión solicitada, una proporción cercana al tercio que se esperaría si la versión fuera irrelevante para el recuperador. Este resultado es coherente con lo reportado por Cipollone et al. (2026): declarar la versión en la consulta no basta para orientar al sistema hacia ella."))
    A.append(("p", f"El pasaje de oro aparece entre los ocho primeros fragmentos en el {pct(o['gold_recall_at_8'])} de las preguntas (MRR = {fmt(o['gold_mrr'], 3)}), aunque el documento correcto del módulo aparece en el {pct(o['gold_doc_in_top8'])}. Es decir, el recuperador casi siempre encuentra la página correcta, pero con frecuencia elige la versión equivocada de ella."))
    A.append(("caption", "Tabla 4. Diagnóstico de la recuperación ciega a la versión (k = 8)"))
    A.append(("table", ([
        ["Subconjunto", "n", "Oro en top-8", "MRR", "Top-1 de otra versión", "Top-1 desalineado", "Algún fragmento desalineado"],
        ["Global", str(o["n"]), pct(o["gold_recall_at_8"]), fmt(o["gold_mrr"], 3), pct(o["top1_from_other_version"]), pct(o["top1_misaligned"]), pct(o["any_misaligned_in_top8"])],
        ["Oro en español", str(es["n"]), pct(es["gold_recall_at_8"]), fmt(es["gold_mrr"], 3), pct(es["top1_from_other_version"]), pct(es["top1_misaligned"]), pct(es["any_misaligned_in_top8"])],
        ["Oro en inglés", str(en["n"]), pct(en["gold_recall_at_8"]), fmt(en["gold_mrr"], 3), pct(en["top1_from_other_version"]), pct(en["top1_misaligned"]), pct(en["any_misaligned_in_top8"])],
        ["Versión nueva", str(nu["n"]), pct(nu["gold_recall_at_8"]), fmt(nu["gold_mrr"], 3), pct(nu["top1_from_other_version"]), pct(nu["top1_misaligned"]), pct(nu["any_misaligned_in_top8"])],
        ["Versión anterior", str(an["n"]), pct(an["gold_recall_at_8"]), fmt(an["gold_mrr"], 3), pct(an["top1_from_other_version"]), pct(an["top1_misaligned"]), pct(an["any_misaligned_in_top8"])],
    ], [1700, 600, 1200, 900, 1500, 1500, 1600])))
    A.append(("image", img_retr))
    A.append(("caption", "Figura 1. Recuperación del pasaje de oro y desalineación de versión del primer fragmento, por subconjunto"))
    A.append(("p", f"El hallazgo del Capítulo 3 sobre la traducción se refleja directamente en la recuperación. Cuando el pasaje de oro está mayoritariamente en español, se recupera en el {pct(es['gold_recall_at_8'])} de los casos; cuando está en inglés (contenido nuevo aún no traducido), solo en el {pct(en['gold_recall_at_8'])} (MRR {fmt(en['gold_mrr'], 3)} frente a {fmt(es['gold_mrr'], 3)}). La consulta en español se parece más a la descripción traducida de la versión anterior que al texto nuevo en inglés, lo que empuja sistemáticamente al sistema hacia la versión equivocada."))
    A.append(("h4", "Prueba 2: comportamiento de la señal de alineación de versión"))
    A.append(("p", f"El {pct(o['top1_misaligned'])} de las preguntas tiene como primer fragmento uno que contiene segmentos ausentes o modificados en la versión solicitada (S_ver < 1 para ese fragmento), y el {pct(o['any_misaligned_in_top8'])} tiene al menos un fragmento desalineado entre los ocho recuperados. Que el primer fragmento sea de otra versión ({pct(o['top1_from_other_version'])}) no implica por sí solo un error: muchos fragmentos de otra versión tienen texto idéntico y siguen siendo válidos. La señal propuesta distingue ambos casos y marca como riesgo solo el subconjunto en el que el contenido efectivamente difiere. Esto es lo que se espera de ella: que no penalice la coincidencia inocua entre versiones y que se active donde hay conflicto."))
    A.append(("p", "Dos ejemplos ilustran el mecanismo. Ante la pregunta «En Python 3.14, ¿en cuántas generaciones clasifica el recolector de basura del módulo gc a los objetos?», el primer fragmento recuperado es el de 3.12, traducido, que afirma que son tres generaciones, mientras que la respuesta correcta para 3.14 es dos. De forma análoga, para «En Python 3.13, ¿qué backends usa el módulo genérico dbm?» se recupera primero la página de 3.12, que no menciona el nuevo backend dbm.sqlite3. En ambos casos el contexto es tópicamente impecable y un autorater de suficiencia lo consideraría suficiente; solo la señal de versión detecta que esos fragmentos contienen texto que no existe en la versión solicitada."))

    B: list[tuple] = []
    if ev is None:
        B.append(("p", "La evaluación del pipeline completo (generación, autorater de suficiencia, juez y comparación de compuertas) está implementada y probada con datos sintéticos, pero no se ejecutó sobre el conjunto real en esta entrega preliminar porque el presupuesto de API asignado se consumió durante la construcción del conjunto de evaluación. Su ejecución tiene un costo estimado de 2 USD para las 127 preguntas y no requiere cambios de código (scripts 04_run_rag.py y 05_evaluate.py). Esta sección se completará con esos resultados; a continuación se resumen el análisis disponible y lo que se espera medir."))
        B.append(("h4", "Lectura de los resultados disponibles"))
    else:
        runs = pd.read_parquet(root / "results/runs.parquet")
        B.extend(gate_blocks(ev, diag, img_risk, runs))
        B.append(("h4", "Discusión"))
        B.append(("bullet", "**El clasificador de suficiencia es ciego a la versión, como se anticipó.** Puntúa casi igual las respuestas de versión incorrecta y las correctas, porque el contexto que recibe es tópicamente completo. La señal de alineación sí separa ambos grupos, y por eso aporta información que la suficiencia no contiene."))
        B.append(("bullet", "**El riesgo se concentra en las preguntas sobre la versión anterior.** El texto nuevo y el anterior compiten en la recuperación. Cuando se pregunta por la versión anterior, el sistema tiende a responder con la documentación más reciente; en el sentido inverso, la falta de traducción del texto nuevo hace que el sistema se abstenga más a menudo de lo que se equivoca."))
        B.append(("bullet", "**La evidencia es direccionalmente favorable pero no concluyente.** La compuerta con señal de versión reduce el área bajo la curva de versión incorrecta, pero con solo siete casos de conflicto en la partición de prueba las diferencias a cobertura fija no son significativas. El conjunto completo de 180 preguntas validadas, con más preguntas sobre la versión anterior, es necesario para resolver la hipótesis."))
    B.append(("bullet", "**El problema existe en el corpus real y es frecuente.** Con una consulta que nombra la versión, más de la mitad de los primeros fragmentos recuperados son de otra versión y casi un tercio contiene texto que no existe en la versión solicitada. Esto confirma la premisa del trabajo para la documentación de Python en español."))
    B.append(("bullet", "**La señal de versión es computable sin costo y sin entrenamiento.** Se obtiene de la comparación del texto fuente de los segmentos entre ramas, no requiere llamadas adicionales al modelo y no penaliza los fragmentos idénticos entre versiones."))
    B.append(("bullet", "**El idioma es una variable de confusión.** El contenido que cambia entre versiones está mayoritariamente sin traducir y se recupera peor. La versión final deberá reportar los resultados separados por idioma del pasaje de oro, o bien completar la traducción de los segmentos afectados, para que la mejora atribuida a la señal de versión no se confunda con un efecto de idioma."))
    B.append(("h4", "Limitaciones de la versión preliminar"))
    for t in [
        "El conjunto de evaluación tiene 127 preguntas generadas con asistencia de un modelo de lenguaje y aún no validadas por dos anotadores; la meta del anteproyecto es 180 preguntas validadas.",
        "Las preguntas sobre la versión anterior son minoría (33 de 127), porque solo se aceptaron cuando un pasaje las respaldaba explícitamente. El par 3.12→3.13 está subrepresentado frente a 3.13→3.14.",
        "Solo se indexó la sección library/ del corpus; la versión final debe incluir el corpus completo.",
        "La confianza del generador es verbalizada; se evaluará la autoconsistencia como alternativa.",
        "El juez automático debe contrastarse con una muestra calificada por humanos para estimar su exactitud.",
    ]:
        B.append(("bullet", t))
    return A, B


LABELS = {"correct": "Correcta", "wrong_version": "Versión incorrecta", "incorrect": "Incorrecta (otro motivo)", "no_answer": "Sin respuesta"}


def gate_blocks(ev: dict, diag: dict | None, img_risk, runs: pd.DataFrame) -> list[tuple]:
    B: list[tuple] = []
    fc = ev["full_coverage"]
    runs = runs.copy()
    runs["is_new"] = runs.apply(lambda x: x.target_version == x.pair.split("->")[1], axis=1)
    runs["gold_in"] = runs["gold_rank"].notna()
    lab = runs["label"].value_counts()
    B.append(("h4", "Prueba 3: pipeline completo sobre las 127 preguntas"))
    B.append(("p", f"El generador respondió las {len(runs)} preguntas con la misma recuperación de la Prueba 1. El juez clasificó {lab.get('correct', 0)} respuestas como correctas, {lab.get('wrong_version', 0)} como de versión incorrecta, {lab.get('incorrect', 0)} como incorrectas por otro motivo y {lab.get('no_answer', 0)} como sin respuesta, porque el generador declaró que el contexto no le permitía responder. La abstención del generador se explica casi por completo por la recuperación: {int(((runs.label == 'no_answer') & ~runs.gold_in).sum())} de las {lab.get('no_answer', 0)} abstenciones ocurren cuando el pasaje de oro no está entre los ocho fragmentos recuperados."))
    old, new = runs[~runs.is_new], runs[runs.is_new]
    B.append(("p", f"Las respuestas de versión incorrecta se concentran en las preguntas sobre la versión anterior: {int((old.label == 'wrong_version').sum())} de {len(old)} ({pct((old.label == 'wrong_version').mean())}), frente a {int((new.label == 'wrong_version').sum())} de {len(new)} ({pct((new.label == 'wrong_version').mean())}) en las preguntas sobre la versión nueva. Cuando se pregunta por una versión anterior, el fragmento más reciente, que describe el comportamiento nuevo, compite con el correcto y el generador lo adopta sin advertir el conflicto."))
    if diag and diag.get("suf_sufficient_rate_when_wrong_version") is not None:
        B.append(("p", f"El autorater de suficiencia consideró suficiente el contexto en el {pct(diag['suf_sufficient_rate'])} de todas las preguntas y en el {pct(diag['suf_sufficient_rate_when_wrong_version'])} de las que terminaron en una respuesta de versión incorrecta. La Tabla 5 muestra el valor medio de cada señal por categoría: la suficiencia y la confianza son prácticamente iguales en las respuestas correctas y en las de versión incorrecta, mientras que la alineación de los fragmentos citados (ver_cited) es claramente menor en estas últimas. Esta es la evidencia más directa de que la señal de versión aporta información que el clasificador de suficiencia no captura."))
    B.append(("caption", "Tabla 5. Valor medio de las señales por categoría de respuesta (127 preguntas)"))
    g = runs.groupby("label")[["suf_score", "confidence", "ver_w", "ver_top1", "ver_cited"]].mean()
    rows = [["Categoría", "n", "S_suf", "S_conf", "ver_w", "ver_top1", "ver_cited"]]
    for k in ["correct", "wrong_version", "incorrect", "no_answer"]:
        if k in g.index:
            r = g.loc[k]
            rows.append([LABELS[k], str(int(lab.get(k, 0)))] + [fmt(r[c], 2) for c in g.columns])
    B.append(("table", (rows, [2400, 600, 1150, 1150, 1150, 1150, 1300])))
    dist = ev["test_label_distribution"]
    B.append(("h4", "Prueba 4: comparación de compuertas"))
    B.append(("p", f"La partición por conglomerados de cambio dejó {ev['n_calibration']} preguntas ({ev['changes_calibration']} cambios) para calibrar las compuertas y {ev['n_test']} preguntas ({ev['changes_test']} cambios) para la prueba. Sin compuerta, es decir con cobertura del 100 %, el sistema responde correctamente el {pct(fc['selective_accuracy'])} de las preguntas de prueba y entrega una respuesta de versión incorrecta en el {pct(fc['wrong_version_rate'])}. Distribución de etiquetas en la partición de prueba: " + ", ".join(f"{LABELS[k].lower()} {v}" for k, v in dist.items()) + "."))
    B.append(("caption", "Tabla 6. Exactitud selectiva y tasa de versión incorrecta por nivel de cobertura (partición de prueba)"))
    rows = [["Cobertura", "Compuerta", "Respondidas", "Exactitud selectiva", "Tasa versión incorrecta"]]
    names = {"baseline": "Línea base (S_suf, S_conf)", "proposed": "Propuesta (+ S_ver)", "rule_suf_conf": "Regla S_suf·S_conf", "rule_suf_conf_ver": "Regla S_suf·S_conf·ver_cited"}
    for c, d in ev["by_coverage"].items():
        for g in ["baseline", "proposed", "rule_suf_conf", "rule_suf_conf_ver"]:
            m = d[g]
            rows.append([f"{int(float(c) * 100)} %", names[g], str(m["n_answered"]), pct(m["selective_accuracy"]), pct(m["wrong_version_rate"])])
    B.append(("table", (rows, [1100, 2900, 1300, 1800, 1900])))
    B.append(("image", img_risk))
    B.append(("caption", "Figura 2. Curvas de cobertura frente a riesgo y de tasa de versión incorrecta frente a cobertura"))
    B.append(("caption", "Tabla 7. Área bajo las curvas (menor es mejor)"))
    rows = [["Compuerta", "AURC (riesgo)", "AURC (versión incorrecta)"]]
    for g in ["baseline", "proposed", "rule_suf_conf", "rule_suf_conf_ver"]:
        rows.append([names[g], fmt(ev["aurc"][g], 3), fmt(ev["aurc_wrong_version"][g], 3)])
    B.append(("table", (rows, [3600, 2400, 2800])))
    a, aw = ev["aurc"], ev["aurc_wrong_version"]
    B.append(("p", f"Integrada sobre todo el rango de coberturas, la compuerta con señal de versión reduce el área bajo la curva de riesgo de {fmt(a['baseline'], 3)} a {fmt(a['proposed'], 3)} y la de versión incorrecta de {fmt(aw['baseline'], 3)} a {fmt(aw['proposed'], 3)}, una reducción relativa del {fmt(100 * (1 - aw['proposed'] / aw['baseline']), 0)} %. La regla multiplicativa sin entrenamiento S_suf·S_conf·ver_cited obtiene un resultado similar ({fmt(aw['rule_suf_conf_ver'], 3)}), lo que indica que la mejora proviene de la señal y no del ajuste de la regresión. En los tres puntos de cobertura fijados por la hipótesis, en cambio, las diferencias son pequeñas: a 50 % de cobertura la tasa de versión incorrecta baja de {pct(ev['by_coverage']['0.5']['baseline']['wrong_version_rate'])} a {pct(ev['by_coverage']['0.5']['proposed']['wrong_version_rate'])}, y a 70 % y 90 % ambas compuertas entregan el mismo conjunto de respuestas de versión incorrecta."))
    B.append(("caption", "Tabla 8. Prueba de hipótesis: propuesta frente a línea base"))
    rows = [["Cobertura", "Solo línea base falla", "Solo propuesta falla", "p (McNemar, unilateral)", "p (Holm)", "Δ tasa (IC 95 % bootstrap)"]]
    for c, t in ev["tests"].items():
        bs = t["bootstrap"]
        rows.append([f"{int(float(c) * 100)} %", str(t["baseline_only"]), str(t["proposed_only"]), fmt(t["p_value_one_sided"], 3), fmt(t["p_holm"], 3), f"{fmt(100 * bs['diff_mean'])} pp [{fmt(100 * bs['ci95'][0])}; {fmt(100 * bs['ci95'][1])}]"])
    B.append(("table", (rows, [1100, 1500, 1500, 1700, 1100, 2100])))
    sig = [c for c, t in ev["tests"].items() if t["significant"]]
    if sig:
        B.append(("p", f"La reducción de la tasa de respuesta con versión incorrecta es estadísticamente significativa tras la corrección de Holm en las coberturas {', '.join(str(int(float(c) * 100)) + ' %' for c in sig)}. Con la cautela que impone un conjunto preliminar y no validado, el resultado es consistente con la hipótesis."))
    else:
        B.append(("p", "Ninguna de las diferencias alcanza significación estadística tras la corrección de Holm. Con el tamaño de la partición de prueba de esta versión preliminar, la potencia es limitada; el resultado no permite confirmar ni refutar la hipótesis y debe repetirse con el conjunto completo y validado."))
    coef = ev["coefficients"]["proposed"]
    B.append(("p", "Coeficientes de la regresión logística de la compuerta propuesta (variables en escala 0-1): " + ", ".join(f"{k} = {fmt(v, 2)}" for k, v in coef.items()) + "."))
    return B
