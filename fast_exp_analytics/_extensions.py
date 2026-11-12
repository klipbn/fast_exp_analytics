"""Общая логика расширенных возможностей пайплайнов AB / ABC / ABCD.

Опциональные колонки конфига метрик:

* ``test`` — критерий вместо дефолтного для типа метрики (см. ``available_tests``);
* ``covariate`` — колонка с пред-экспериментальным значением для CUPED
  (типы ``additive`` и ``average``);
* ``winsorize`` — верхний квантиль винзоризации, например ``0.99``
  (типы ``additive`` и ``average``); порог считается по всем сравниваемым группам.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .confidence_intervals import metric_confidence_interval
from .effect_sizes import cliffs_delta, cohens_d, cohens_h, relative_lift_ci
from .stat_tests import (
    ALLOWED_TESTS_BY_TYPE,
    DEFAULT_TEST_BY_TYPE,
    ESTIMAND_BY_TYPE,
    JOINT_POSTHOC_TESTS,
    available_tests,
    ratio_group_variance,
    run_metric_test,
)
from .variance_reduction import cuped_adjust

OPTIONAL_METRIC_COLUMNS = ("test", "covariate", "winsorize")

EXTENDED_COLUMNS = [
    "test_name",
    "statistic",
    "effect_size",
    "effect_size_name",
    "rel_ci_lower",
    "rel_ci_upper",
    "delta_cuped",
    "cuped_variance_reduction",
    "winsorize_threshold",
]

_VALUE_ADJUSTABLE_TYPES = ("additive", "average")


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def metric_options(metrics_df: pd.DataFrame, metric) -> dict:
    """Прочитать опциональные настройки метрики (``test``, ``covariate``, ``winsorize``)."""
    out = {"test": None, "covariate": None, "winsorize": None}
    for key in OPTIONAL_METRIC_COLUMNS:
        if key not in metrics_df.columns:
            continue
        value = metrics_df.loc[metric, key]
        if _is_missing(value):
            continue
        if key == "test":
            out[key] = str(value).strip().lower()
        elif key == "covariate":
            out[key] = str(value)
        else:
            out[key] = float(value)
    return out


def validate_metric_options(metrics_df: pd.DataFrame, multi_group: bool = False) -> None:
    """Проверить опциональные колонки конфига метрик; при ошибке — ``ValueError``."""
    if not any(col in metrics_df.columns for col in OPTIONAL_METRIC_COLUMNS):
        return
    for metric in metrics_df.index:
        metric_type = str(metrics_df.loc[metric, "type"]).lower()
        opts = metric_options(metrics_df, metric)
        test = opts["test"]
        if test is not None:
            if metric_type not in ALLOWED_TESTS_BY_TYPE:
                raise ValueError(f"Metric {metric!r}: unsupported metric type {metric_type!r}")
            allowed = available_tests(metric_type, multi_group=multi_group)
            if test not in allowed:
                hint = ""
                if test in JOINT_POSTHOC_TESTS and not multi_group:
                    hint = " (post-hoc процедуры доступны только для 3+ групп)"
                raise ValueError(
                    f"Metric {metric!r}: test {test!r} is not supported for type "
                    f"{metric_type!r}{hint}; allowed: {list(allowed)}"
                )
        for key in ("covariate", "winsorize"):
            if opts[key] is not None and metric_type not in _VALUE_ADJUSTABLE_TYPES:
                raise ValueError(
                    f"Metric {metric!r}: '{key}' is supported only for metric types "
                    f"{list(_VALUE_ADJUSTABLE_TYPES)}, got {metric_type!r}"
                )
        if opts["winsorize"] is not None and not 0 < opts["winsorize"] < 1:
            raise ValueError(f"Metric {metric!r}: 'winsorize' must be a quantile in (0, 1)")
        if opts["covariate"] is not None and test in JOINT_POSTHOC_TESTS:
            raise ValueError(
                f"Metric {metric!r}: 'covariate' (CUPED) cannot be combined with joint "
                f"post-hoc test {test!r}"
            )


def build_metric_frame(df: pd.DataFrame, num: str, den: str, covariate: str | None = None):
    """Собрать фрейм метрики с колонками ``user_id, exp_group, num, den[, cov]``."""
    if covariate is not None and covariate not in df.columns:
        raise ValueError(f"Input data is missing covariate column: {covariate!r}")
    data = {"user_id": df["user_id"], "exp_group": df["exp_group"], "num": df[num]}
    data["den"] = df[den] if num != den else 1.0
    if covariate is not None:
        data["cov"] = pd.to_numeric(df[covariate], errors="coerce")
    return pd.DataFrame(data)


def winsorize_metric_frame(df_metric: pd.DataFrame, metric_type: str, quantile: float, groups):
    """Обрезать ``num`` сверху по квантилю, посчитанному по всем сравниваемым группам."""
    df_metric = df_metric.copy()
    num = pd.to_numeric(df_metric["num"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    in_scope = df_metric["exp_group"].isin(list(groups)) & num.notna()
    if metric_type == "average":
        in_scope &= pd.to_numeric(df_metric["den"], errors="coerce") > 0
    reference = num[in_scope]
    if reference.empty:
        return df_metric, np.nan
    threshold = float(np.quantile(reference.to_numpy(dtype=float), quantile))
    df_metric["num"] = num.clip(upper=threshold)
    return df_metric, threshold


def cuped_stat_frames(df_control: pd.DataFrame, df_pilot: pd.DataFrame):
    """Вернуть копии фреймов с CUPED-скорректированной ``value`` и сводку корректировки."""
    adj_c, adj_e, info = cuped_adjust(
        df_control["value"].to_numpy(dtype=float),
        df_control["cov"].to_numpy(dtype=float),
        df_pilot["value"].to_numpy(dtype=float),
        df_pilot["cov"].to_numpy(dtype=float),
    )
    stat_c = df_control.assign(value=adj_c)
    stat_e = df_pilot.assign(value=adj_e)
    info = dict(info)
    info["delta_cuped"] = float(np.nanmean(adj_e) - np.nanmean(adj_c)) if len(adj_c) and len(adj_e) else np.nan
    return stat_c, stat_e, info


def resolve_inference(
    stat_control: pd.DataFrame,
    stat_pilot: pd.DataFrame,
    metric_type: str,
    test: str | None,
    ci_alpha: float,
    n_resamples: int,
    random_state,
    default_p_value_fn: Callable,
    need_statistic: bool = False,
    override: dict | None = None,
) -> dict:
    """p-value, CI, имя критерия и статистика с учётом выбранного ``test``.

    Без ``test`` (или с дефолтным для типа) поведение идентично исходному
    пайплайну. Для альтернативного критерия CI берётся из него, только если
    критерий оценивает ту же величину, что и дельта метрики; иначе остаётся
    дефолтный CI типа метрики.
    """
    default_name = DEFAULT_TEST_BY_TYPE.get(metric_type)

    def _default_ci():
        return metric_confidence_interval(
            stat_control, stat_pilot, metric_type, ci_alpha, n_resamples, random_state
        )

    if override is not None:
        ci = (override.get("ci_lower", np.nan), override.get("ci_upper", np.nan))
        if not (np.isfinite(ci[0]) and np.isfinite(ci[1])):
            ci = _default_ci()
        return {
            "p_value": override.get("p_value", np.nan),
            "ci_lower": ci[0],
            "ci_upper": ci[1],
            "test_name": override.get("test_name", ""),
            "statistic": override.get("statistic", np.nan),
        }

    if test is None or test == default_name:
        statistic = np.nan
        if need_statistic and default_name is not None:
            statistic = run_metric_test(
                stat_control, stat_pilot, metric_type, None, ci_alpha, n_resamples, random_state
            ).statistic
        ci = _default_ci()
        return {
            "p_value": default_p_value_fn(stat_control, stat_pilot, metric_type),
            "ci_lower": ci[0],
            "ci_upper": ci[1],
            "test_name": default_name or "",
            "statistic": statistic,
        }

    res = run_metric_test(
        stat_control, stat_pilot, metric_type, test, ci_alpha, n_resamples, random_state
    )
    ci = (res.ci_lower, res.ci_upper)
    same_estimand = res.estimand == ESTIMAND_BY_TYPE.get(metric_type)
    if not (same_estimand and np.isfinite(ci[0]) and np.isfinite(ci[1])):
        ci = _default_ci()
    p_value = float(res.p_value) if np.isfinite(res.p_value) else np.nan
    return {
        "p_value": p_value,
        "ci_lower": ci[0],
        "ci_upper": ci[1],
        "test_name": test,
        "statistic": res.statistic,
    }


def _mean_and_var(values) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    return float(x.mean()), float(x.var(ddof=1) / len(x))


def _share_mean_and_var(frame: pd.DataFrame) -> tuple[float, float]:
    valid = (frame["den"] > 0) & frame["num"].notna()
    n = int(valid.sum())
    if n == 0:
        return np.nan, np.nan
    p = float((valid & (frame["num"] > 0)).sum() / n)
    return p, p * (1 - p) / n


def extended_stats(
    df_control: pd.DataFrame,
    df_pilot: pd.DataFrame,
    stat_control: pd.DataFrame,
    stat_pilot: pd.DataFrame,
    metric_type: str,
    alpha: float,
) -> dict:
    """Размер эффекта и CI относительного изменения для расширенного вывода.

    * ``additive`` / ``average`` / ``ratio`` — d Коэна (для ratio по
      линеаризованным значениям);
    * ``share`` — h Коэна;
    * ``median`` — дельта Клиффа (ранг-бисериальная корреляция).

    ``rel_ci_*`` — CI относительного изменения в долях (дельта-метод): для
    ``additive`` / ``average`` — среднего на юнит, для ``share`` — доли, для
    ``ratio`` — самой ratio-метрики.
    """
    out = {
        "effect_size": np.nan,
        "effect_size_name": "",
        "rel_ci_lower": np.nan,
        "rel_ci_upper": np.nan,
    }
    if metric_type in ("additive", "average"):
        out["effect_size"] = cohens_d(df_control["value"], df_pilot["value"])
        out["effect_size_name"] = "cohens_d"
        m_c, v_c = _mean_and_var(stat_control["value"])
        m_e, v_e = _mean_and_var(stat_pilot["value"])
    elif metric_type == "ratio":
        out["effect_size"] = cohens_d(df_control["value"], df_pilot["value"])
        out["effect_size_name"] = "cohens_d_linearized"
        m_c, v_c, _ = ratio_group_variance(df_control["num"], df_control["den"])
        m_e, v_e, _ = ratio_group_variance(df_pilot["num"], df_pilot["den"])
    elif metric_type == "share":
        m_c, v_c = _share_mean_and_var(df_control)
        m_e, v_e = _share_mean_and_var(df_pilot)
        if np.isfinite(m_c) and np.isfinite(m_e):
            out["effect_size"] = cohens_h(m_c, m_e)
        out["effect_size_name"] = "cohens_h"
    elif metric_type == "median":
        out["effect_size"] = cliffs_delta(df_control["value"], df_pilot["value"])
        out["effect_size_name"] = "cliffs_delta"
        return out
    else:
        return out
    out["rel_ci_lower"], out["rel_ci_upper"] = relative_lift_ci(m_c, v_c, m_e, v_e, alpha)
    return out
