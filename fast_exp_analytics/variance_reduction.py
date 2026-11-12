"""Снижение дисперсии и работа с тяжёлыми хвостами: CUPED, винзоризация, пост-стратификация."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats


def cuped_theta(values_by_group: Sequence, covariates_by_group: Sequence) -> float:
    """Коэффициент CUPED ``theta = cov(Y, X) / var(X)`` по внутригрупповой ковариации.

    Центрирование внутри каждой группы исключает вклад самого эффекта
    эксперимента в оценку ``theta``.
    """
    cov_sum, var_sum = 0.0, 0.0
    for y, x in zip(values_by_group, covariates_by_group):
        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 2:
            continue
        y_c, x_c = y[mask] - y[mask].mean(), x[mask] - x[mask].mean()
        cov_sum += float(np.sum(y_c * x_c))
        var_sum += float(np.sum(x_c**2))
    if var_sum == 0:
        return 0.0
    return cov_sum / var_sum


def cuped_adjust(control, control_covariate, experiment, experiment_covariate):
    """CUPED-корректировка двух групп.

    ``Y_adj = Y - theta * (X - mean(X))``, где ``X`` — ковариата из
    пред-экспериментального периода (обычно та же метрика до эксперимента).
    Пропуски ковариаты (новые пользователи) заменяются общим средним — такие
    наблюдения не корректируются. Ковариата не должна зависеть от воздействия.

    Возвращает ``(control_adj, experiment_adj, info)``, где ``info`` содержит
    ``theta``, ``correlation`` и ``variance_reduction`` (доля снятой дисперсии).
    """
    y_c = np.asarray(control, dtype=float)
    y_e = np.asarray(experiment, dtype=float)
    x_c = np.asarray(control_covariate, dtype=float).copy()
    x_e = np.asarray(experiment_covariate, dtype=float).copy()
    if len(y_c) != len(x_c) or len(y_e) != len(x_e):
        raise ValueError("values and covariates must have the same length within a group")
    x_c[~np.isfinite(x_c)] = np.nan
    x_e[~np.isfinite(x_e)] = np.nan
    x_all = np.concatenate([x_c, x_e])
    info = {"theta": 0.0, "correlation": np.nan, "variance_reduction": 0.0}
    if np.isfinite(x_all).sum() < 2:
        return y_c.copy(), y_e.copy(), info
    x_mean = float(np.nanmean(x_all))
    x_c = np.where(np.isnan(x_c), x_mean, x_c)
    x_e = np.where(np.isnan(x_e), x_mean, x_e)
    theta = cuped_theta([y_c, y_e], [x_c, x_e])
    adj_c = y_c - theta * (x_c - x_mean)
    adj_e = y_e - theta * (x_e - x_mean)

    def _pooled_var(a, b):
        parts = [v[np.isfinite(v)] for v in (a, b)]
        ss = sum(float(np.sum((p - p.mean()) ** 2)) for p in parts if len(p) > 1)
        dof = sum(len(p) - 1 for p in parts if len(p) > 1)
        return ss / dof if dof > 0 else np.nan

    var_raw, var_adj = _pooled_var(y_c, y_e), _pooled_var(adj_c, adj_e)
    if np.isfinite(var_raw) and var_raw > 0 and np.isfinite(var_adj):
        reduction = 1 - var_adj / var_raw
        info = {
            "theta": float(theta),
            "correlation": float(np.sign(theta) * np.sqrt(max(reduction, 0.0))),
            "variance_reduction": float(reduction),
        }
    else:
        info["theta"] = float(theta)
    return adj_c, adj_e, info


def add_cuped_column(
    df: pd.DataFrame,
    value_col: str,
    covariate_col: str,
    group_col: str = "exp_group",
    out_col: str | None = None,
) -> pd.DataFrame:
    """Добавить в DataFrame CUPED-скорректированную колонку (для любого числа групп)."""
    out_col = out_col or f"{value_col}_cuped"
    result = df.copy()
    y = pd.to_numeric(result[value_col], errors="coerce").to_numpy(dtype=float)
    x = pd.to_numeric(result[covariate_col], errors="coerce").to_numpy(dtype=float)
    x[~np.isfinite(x)] = np.nan
    if np.isfinite(x).sum() < 2:
        result[out_col] = y
        return result
    x_mean = float(np.nanmean(x))
    x = np.where(np.isnan(x), x_mean, x)
    codes = result[group_col].to_numpy()
    groups = pd.unique(codes)
    theta = cuped_theta([y[codes == g] for g in groups], [x[codes == g] for g in groups])
    result[out_col] = y - theta * (x - x_mean)
    return result


def winsorize_values(
    values,
    upper_quantile: float | None = 0.99,
    lower_quantile: float | None = None,
    reference=None,
):
    """Винзоризация: значения за порогом заменяются порогом.

    Пороги считаются по ``reference`` (по умолчанию — по самим ``values``). В A/B
    порог нужно считать по объединённой выборке всех групп, иначе обрезка
    по-разному затронет группы и сместит оценку. Возвращает
    ``(clipped, lower_threshold, upper_threshold)``.
    """
    array = np.asarray(values, dtype=float)
    ref = array if reference is None else np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    lower = upper = None
    if len(ref):
        if upper_quantile is not None:
            if not 0 < upper_quantile <= 1:
                raise ValueError("upper_quantile must be in (0, 1]")
            upper = float(np.quantile(ref, upper_quantile))
        if lower_quantile is not None:
            if not 0 <= lower_quantile < 1:
                raise ValueError("lower_quantile must be in [0, 1)")
            lower = float(np.quantile(ref, lower_quantile))
    clipped = np.clip(array, lower, upper) if (lower is not None or upper is not None) else array
    return clipped, lower, upper


def winsorize_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    upper_quantile: float | None = 0.99,
    lower_quantile: float | None = None,
) -> pd.DataFrame:
    """Винзоризовать колонки DataFrame по общим (для всех групп) порогам."""
    result = df.copy()
    for col in columns:
        values = pd.to_numeric(result[col], errors="coerce")
        result[col], _, _ = winsorize_values(values, upper_quantile, lower_quantile)
    return result


def post_stratified_difference(
    values,
    groups,
    strata,
    group_control,
    group_experiment,
    alpha: float = 0.05,
) -> dict:
    """Пост-стратифицированная оценка разницы средних.

    Эффект считается внутри каждой страты и взвешивается долей страты в общей
    выборке; дисперсия между стратами исключается. Страты должны определяться
    до эксперимента (страна, платформа, сегмент расхода). Страты, где в одной
    из групп меньше двух наблюдений, исключаются, веса перенормируются.
    """
    frame = pd.DataFrame(
        {
            "value": pd.to_numeric(pd.Series(np.asarray(values)), errors="coerce"),
            "group": np.asarray(groups),
            "stratum": np.asarray(strata),
        }
    ).dropna(subset=["value"])
    frame = frame[frame["group"].isin([group_control, group_experiment])]
    estimate, variance, weight_sum, used = 0.0, 0.0, 0.0, 0
    parts = []
    for stratum, part in frame.groupby("stratum"):
        c = part.loc[part["group"] == group_control, "value"].to_numpy()
        e = part.loc[part["group"] == group_experiment, "value"].to_numpy()
        if len(c) < 2 or len(e) < 2:
            continue
        parts.append((len(part), e.mean() - c.mean(), c.var(ddof=1) / len(c) + e.var(ddof=1) / len(e)))
        weight_sum += len(part)
        used += 1
    if used == 0:
        return {
            "estimate": np.nan,
            "se": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_strata_used": 0,
        }
    for size, diff, var in parts:
        w = size / weight_sum
        estimate += w * diff
        variance += w**2 * var
    se = float(np.sqrt(variance))
    z = float(stats.norm.ppf(1 - alpha / 2))
    p_value = float(2 * stats.norm.sf(abs(estimate / se))) if se > 0 else np.nan
    return {
        "estimate": float(estimate),
        "se": se,
        "p_value": p_value,
        "ci_lower": float(estimate - z * se),
        "ci_upper": float(estimate + z * se),
        "n_strata_used": used,
    }
