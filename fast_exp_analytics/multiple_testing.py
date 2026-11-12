"""Поправки на множественные сравнения (FWER и FDR)."""

from __future__ import annotations

import numpy as np
from statsmodels.stats.multitest import multipletests

# Псевдонимы → имена методов statsmodels.
PVALUE_ADJUST_METHODS = {
    "bonferroni": "bonferroni",
    "sidak": "sidak",
    "holm": "holm",
    "holm-sidak": "holm-sidak",
    "holm_sidak": "holm-sidak",
    "hochberg": "simes-hochberg",
    "simes-hochberg": "simes-hochberg",
    "hommel": "hommel",
    "bh": "fdr_bh",
    "fdr_bh": "fdr_bh",
    "benjamini-hochberg": "fdr_bh",
    "by": "fdr_by",
    "fdr_by": "fdr_by",
    "benjamini-yekutieli": "fdr_by",
    "fdr_tsbh": "fdr_tsbh",
}

FWER_METHODS = ("bonferroni", "sidak", "holm", "holm-sidak", "hochberg", "hommel")
FDR_METHODS = ("bh", "by", "fdr_tsbh")


def adjust_pvalues(p_values, method: str | None = "holm", alpha: float = 0.05):
    """Скорректировать p-value на множественные сравнения.

    Возвращает ``(p_adjusted, reject)``; NaN игнорируются и остаются NaN.

    FWER (подтверждающий анализ, мало гипотез): ``bonferroni``, ``sidak``,
    ``holm`` (рекомендуется), ``holm-sidak``, ``hochberg``, ``hommel``.
    FDR (много метрик/сегментов, исследовательский анализ): ``bh``, ``by``
    (при произвольной зависимости), ``fdr_tsbh``.
    """
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(p.shape, np.nan)
    reject = np.zeros(p.shape, dtype=bool)
    mask = np.isfinite(p)
    if method is None:
        adjusted[mask] = p[mask]
        reject[mask] = p[mask] < alpha
        return adjusted, reject
    key = str(method).lower()
    # неизвестные имена передаются в statsmodels как есть (он сам проверит метод)
    sm_method = PVALUE_ADJUST_METHODS.get(key, method)
    if mask.any():
        rej, adj, _, _ = multipletests(p[mask], alpha=alpha, method=sm_method)
        adjusted[mask] = adj
        reject[mask] = rej
    return adjusted, reject
