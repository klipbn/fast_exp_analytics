"""Размеры эффекта и доверительные интервалы относительного изменения."""

from __future__ import annotations

import numpy as np
from scipy import stats


def _clean(values) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    return array[np.isfinite(array)]


def cohens_d(control, experiment) -> float:
    """d Коэна: разница средних в единицах объединённого стандартного отклонения."""
    c, e = _clean(control), _clean(experiment)
    if len(c) < 2 or len(e) < 2:
        return np.nan
    df = len(c) + len(e) - 2
    pooled = np.sqrt(((len(c) - 1) * c.var(ddof=1) + (len(e) - 1) * e.var(ddof=1)) / df)
    if pooled == 0:
        return np.nan
    return float((e.mean() - c.mean()) / pooled)


def hedges_g(control, experiment) -> float:
    """g Хеджеса: d Коэна с поправкой на смещение при малых выборках."""
    d = cohens_d(control, experiment)
    df = len(_clean(control)) + len(_clean(experiment)) - 2
    if not np.isfinite(d) or df <= 1:
        return np.nan
    return float(d * (1 - 3 / (4 * df - 1)))


def glass_delta(control, experiment) -> float:
    """Дельта Гласса: разница средних в единицах стандартного отклонения контроля."""
    c, e = _clean(control), _clean(experiment)
    if len(c) < 2 or len(e) < 1 or c.std(ddof=1) == 0:
        return np.nan
    return float((e.mean() - c.mean()) / c.std(ddof=1))


def vargha_delaney_a(control, experiment) -> float:
    """A Варга—Делани = ``P(E > C) + 0.5 P(E = C)`` (common language effect size)."""
    c, e = _clean(control), _clean(experiment)
    if len(c) == 0 or len(e) == 0:
        return np.nan
    ranks = stats.rankdata(np.concatenate([c, e]))
    u_e = ranks[len(c) :].sum() - len(e) * (len(e) + 1) / 2
    return float(u_e / (len(c) * len(e)))


def cliffs_delta(control, experiment) -> float:
    """Дельта Клиффа = ``P(E > C) - P(E < C)``; совпадает с ранг-бисериальной корреляцией."""
    a = vargha_delaney_a(control, experiment)
    return float(2 * a - 1) if np.isfinite(a) else np.nan


rank_biserial = cliffs_delta


def cohens_h(p_control: float, p_experiment: float) -> float:
    """h Коэна для двух долей (арксинус-преобразование)."""
    if not (0 <= p_control <= 1 and 0 <= p_experiment <= 1):
        return np.nan
    return float(2 * np.arcsin(np.sqrt(p_experiment)) - 2 * np.arcsin(np.sqrt(p_control)))


def relative_lift_ci(
    mean_control: float,
    var_mean_control: float,
    mean_experiment: float,
    var_mean_experiment: float,
    alpha: float = 0.05,
    method: str = "delta",
) -> tuple[float, float]:
    """CI относительного изменения ``mean_experiment / mean_control - 1`` (в долях).

    ``var_mean_*`` — дисперсии оценок (квадрат стандартной ошибки), группы
    независимы. ``method``:

    * ``delta`` — дельта-метод, симметричный интервал;
    * ``fieller`` — интервал Филлера, точнее при шумном знаменателе; если
      контрольное среднее неотличимо от нуля, интервал не ограничен → NaN.
    """
    values = (mean_control, var_mean_control, mean_experiment, var_mean_experiment)
    if not all(np.isfinite(v) for v in values) or mean_control == 0:
        return np.nan, np.nan
    if var_mean_control < 0 or var_mean_experiment < 0:
        return np.nan, np.nan
    z = float(stats.norm.ppf(1 - alpha / 2))
    ratio = mean_experiment / mean_control
    if method == "delta":
        var = var_mean_experiment / mean_control**2 + (
            mean_experiment**2 * var_mean_control / mean_control**4
        )
        margin = z * np.sqrt(var)
        return float(ratio - 1 - margin), float(ratio - 1 + margin)
    if method == "fieller":
        a = mean_control**2 - z**2 * var_mean_control
        if a <= 0:
            return np.nan, np.nan
        b = mean_experiment * mean_control
        c = mean_experiment**2 - z**2 * var_mean_experiment
        disc = b**2 - a * c
        if disc < 0:
            return np.nan, np.nan
        root = np.sqrt(disc)
        lo, hi = sorted(((b - root) / a, (b + root) / a))
        return float(lo - 1), float(hi - 1)
    raise ValueError("method must be 'delta' or 'fieller'")
