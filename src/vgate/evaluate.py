"""Compuertas, métricas de generación selectiva y pruebas estadísticas.

Compuertas (todas puntúan; se responde a las preguntas con mayor puntuación hasta
alcanzar la cobertura fijada):
  baseline   regresión logística sobre [S_suf, S_conf]           (sin señal de versión)
  proposed   regresión logística sobre [S_suf, S_conf, S_ver*]   (consciente de versión)
  rule_*     reglas sin entrenamiento (producto de señales), como referencia
Los modelos se ajustan en la partición de calibración y se evalúan en la de prueba;
la partición se hace por cambio (change_id) para que un mismo cambio no aparezca en ambas.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from statsmodels.stats.multitest import multipletests

from . import config

DISPLAY = {
    "baseline": "Línea base (S_suf, S_conf)",
    "proposed": "Propuesta (+ S_ver)",
    "rule_suf_conf": "Regla S_suf·S_conf",
    "rule_suf_conf_ver": "Regla S_suf·S_conf·ver_cited",
}
BASE_FEATS = ["suf_score", "confidence"]
VER_FEATS = ["ver_w", "ver_top1", "ver_cited"]


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    gss = GroupShuffleSplit(n_splits=1, train_size=config.CALIBRATION_FRACTION, random_state=config.SEED)
    cal_idx, test_idx = next(gss.split(df, groups=df["change_id"]))
    return df.iloc[cal_idx].copy(), df.iloc[test_idx].copy()


def fit_gate(cal: pd.DataFrame, feats: list[str]) -> LogisticRegression:
    return LogisticRegression(C=1.0, max_iter=1000).fit(cal[feats], cal["correct"])


def answer_mask(scores: np.ndarray, coverage: float) -> np.ndarray:
    n = len(scores)
    k = int(np.ceil(coverage * n))
    order = np.lexsort((np.arange(n), -scores))  # desempate determinista
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def selective_metrics(df: pd.DataFrame, mask: np.ndarray) -> dict:
    ans = df[mask]
    return {
        "coverage": float(mask.mean()),
        "selective_accuracy": float(ans["correct"].mean()) if len(ans) else float("nan"),
        "wrong_version_rate": float(ans["wrong_version"].mean()) if len(ans) else float("nan"),
        "n_answered": int(mask.sum()),
    }


def risk_coverage(df: pd.DataFrame, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.lexsort((np.arange(len(scores)), -scores))
    corr = df["correct"].to_numpy()[order]
    wv = df["wrong_version"].to_numpy()[order]
    n = np.arange(1, len(scores) + 1)
    return n / len(scores), 1 - np.cumsum(corr) / n, np.cumsum(wv) / n


def aurc(cov: np.ndarray, risk: np.ndarray) -> float:
    return float(np.trapezoid(risk, cov))


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    """a, b: vectores binarios pareados (1 = respuesta con versión incorrecta entregada)."""
    n01 = int(((a == 1) & (b == 0)).sum())  # solo la línea base falla
    n10 = int(((a == 0) & (b == 1)).sum())  # solo el propuesto falla
    n = n01 + n10
    p = binomtest(n10, n, 0.5, alternative="less").pvalue if n else 1.0
    return {"baseline_only": n01, "proposed_only": n10, "p_value_one_sided": float(p)}


def cluster_bootstrap_diff(df: pd.DataFrame, mb: np.ndarray, mp: np.ndarray, reps: int = 2000) -> dict:
    """IC 95 % de (WVR propuesto - WVR línea base) remuestreando cambios completos."""
    rng = np.random.default_rng(config.SEED)
    groups = df["change_id"].to_numpy()
    uniq = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in uniq}
    wv = df["wrong_version"].to_numpy()
    diffs = []
    for _ in range(reps):
        idx = np.concatenate([idx_by[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        b, p = mb[idx], mp[idx]
        if b.sum() == 0 or p.sum() == 0:
            continue
        diffs.append(wv[idx][p].mean() - wv[idx][b].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff_mean": float(np.mean(diffs)), "ci95": [float(lo), float(hi)]}


def run(df: pd.DataFrame, out_prefix: str = "") -> dict:
    df = df.copy()
    df["correct"] = (df["label"] == "correct").astype(int)
    df["wrong_version"] = (df["label"] == "wrong_version").astype(int)
    cal, test = split(df)
    gates = {
        "baseline": fit_gate(cal, BASE_FEATS),
        "proposed": fit_gate(cal, BASE_FEATS + VER_FEATS),
    }
    scores = {name: m.predict_proba(test[m.feature_names_in_])[:, 1] for name, m in gates.items()}
    scores["rule_suf_conf"] = (test["suf_score"] * test["confidence"]).to_numpy()
    scores["rule_suf_conf_ver"] = (test["suf_score"] * test["confidence"] * test["ver_cited"]).to_numpy()

    report: dict = {
        "n_total": int(len(df)),
        "n_calibration": int(len(cal)),
        "n_test": int(len(test)),
        "changes_calibration": int(cal["change_id"].nunique()),
        "changes_test": int(test["change_id"].nunique()),
        "test_label_distribution": test["label"].value_counts().to_dict(),
        "full_coverage": selective_metrics(test, np.ones(len(test), dtype=bool)),
        "coefficients": {
            n: dict(zip(m.feature_names_in_, np.round(m.coef_[0], 3).tolist())) for n, m in gates.items()
        },
        "aurc": {},
        "aurc_wrong_version": {},
        "by_coverage": {},
        "tests": {},
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, s in scores.items():
        cov, risk, wvr = risk_coverage(test, s)
        report["aurc"][name] = aurc(cov, risk)
        report["aurc_wrong_version"][name] = aurc(cov, wvr)
        axes[0].plot(cov, risk, label=DISPLAY.get(name, name))
        axes[1].plot(cov, wvr, label=DISPLAY.get(name, name))
    axes[0].set(xlabel="Cobertura", ylabel="Riesgo (1 - exactitud selectiva)", title="Curva cobertura-riesgo")
    axes[1].set(xlabel="Cobertura", ylabel="Tasa de versión incorrecta", title="Versión incorrecta vs cobertura")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.RESULTS / f"{out_prefix}risk_coverage.png", dpi=150)
    plt.close(fig)

    pvals = []
    for c in config.COVERAGE_LEVELS:
        masks = {n: answer_mask(s, c) for n, s in scores.items()}
        report["by_coverage"][str(c)] = {n: selective_metrics(test, m) for n, m in masks.items()}
        wv = test["wrong_version"].to_numpy()
        mb, mp = masks["baseline"], masks["proposed"]
        t = mcnemar_exact(wv * mb, wv * mp)
        t["bootstrap"] = cluster_bootstrap_diff(test, mb, mp)
        report["tests"][str(c)] = t
        pvals.append(t["p_value_one_sided"])
    holm = multipletests(pvals, alpha=0.05, method="holm")
    for c, rej, padj in zip(config.COVERAGE_LEVELS, holm[0], holm[1]):
        report["tests"][str(c)]["p_holm"] = float(padj)
        report["tests"][str(c)]["significant"] = bool(rej)

    (config.RESULTS / f"{out_prefix}evaluation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    test.assign(**{f"score_{k}": v for k, v in scores.items()}).to_csv(
        config.RESULTS / f"{out_prefix}test_scores.csv", index=False
    )
    return report
