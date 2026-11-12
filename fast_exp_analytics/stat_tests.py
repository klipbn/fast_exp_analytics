"""Библиотека статистических критериев для сравнения двух групп.

Все функции — чистые и stateless: принимают массивы (или счётчики) и возвращают
``StatTestResult``. Порядок аргументов везде одинаковый: сначала контроль, затем
эксперимент; эффект считается как ``experiment - control``.

``alternative``:
    * ``"two-sided"`` — эффект отличается от нуля;
    * ``"greater"``  — эксперимент больше контроля;
    * ``"less"``     — эксперимент меньше контроля.

Доверительные интервалы всегда двусторонние уровня ``1 - alpha``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np
from scipy import stats

from .confidence_intervals import proportion_difference_ci

_ALTERNATIVES = ("two-sided", "greater", "less")


@dataclass(frozen=True)
class StatTestResult:
    """Результат статистического критерия.

    ``estimand`` описывает, что именно оценивают ``estimate`` и CI:
    ``mean_diff``, ``trimmed_mean_diff``, ``median_diff``, ``quantile_diff``,
    ``proportion_diff``, ``ratio_diff``, ``prob_superiority``, ``distribution``,
    ``rate_ratio``.
    """

    test: str
    statistic: float = np.nan
    p_value: float = np.nan
    estimate: float = np.nan
    ci_lower: float = np.nan
    ci_upper: float = np.nan
    estimand: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        extra = out.pop("extra")
        return {**out, **extra}


# --------------------------------------------------------------------------- helpers


def _clean(values) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    return array[np.isfinite(array)]


def _check_alternative(alternative: str) -> None:
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"alternative must be one of {_ALTERNATIVES}, got {alternative!r}")


def _p_from_t(t_stat: float, df: float, alternative: str) -> float:
    if not np.isfinite(t_stat):
        return np.nan
    if np.isinf(df):
        dist = stats.norm
        args: tuple = ()
    else:
        dist = stats.t
        args = (df,)
    if alternative == "greater":
        return float(dist.sf(t_stat, *args))
    if alternative == "less":
        return float(dist.cdf(t_stat, *args))
    return float(min(1.0, 2 * dist.sf(abs(t_stat), *args)))


def _crit(df: float, alpha: float) -> float:
    if np.isinf(df):
        return float(stats.norm.ppf(1 - alpha / 2))
    return float(stats.t.ppf(1 - alpha / 2, df))


def _mean_diff_result(
    test: str,
    delta: float,
    se: float,
    df: float,
    alpha: float,
    alternative: str,
    estimand: str = "mean_diff",
    extra: dict[str, Any] | None = None,
) -> StatTestResult:
    extra = dict(extra or {})
    extra.setdefault("se", se)
    extra.setdefault("df", df)
    if not np.isfinite(se) or not (np.isinf(df) or (np.isfinite(df) and df > 0)):
        return StatTestResult(test=test, estimate=delta, estimand=estimand, extra=extra)
    if se == 0:
        return StatTestResult(
            test=test, estimate=delta, ci_lower=delta, ci_upper=delta, estimand=estimand, extra=extra
        )
    t_stat = delta / se
    margin = _crit(df, alpha) * se
    return StatTestResult(
        test=test,
        statistic=float(t_stat),
        p_value=_p_from_t(t_stat, df, alternative),
        estimate=float(delta),
        ci_lower=float(delta - margin),
        ci_upper=float(delta + margin),
        estimand=estimand,
        extra=extra,
    )


def _welch_df(var_mean_c: float, n_c: int, var_mean_e: float, n_e: int) -> float:
    den = var_mean_c**2 / (n_c - 1) + var_mean_e**2 / (n_e - 1)
    if den <= 0:
        return np.nan
    return float((var_mean_c + var_mean_e) ** 2 / den)


def _rng(random_state) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _resample_p_value(null_or_boot: np.ndarray, observed: float, alternative: str, kind: str) -> float:
    """p-value по перестановочному (``kind='perm'``) или bootstrap-распределению."""
    n = len(null_or_boot)
    if kind == "perm":
        eps = 1e-12 * max(1.0, abs(observed))
        if alternative == "greater":
            count = np.sum(null_or_boot >= observed - eps)
        elif alternative == "less":
            count = np.sum(null_or_boot <= observed + eps)
        else:
            count = np.sum(np.abs(null_or_boot) >= abs(observed) - eps)
        return float((count + 1) / (n + 1))
    # bootstrap: обращение percentile-интервала относительно нуля
    p_le = (np.sum(null_or_boot <= 0) + 1) / (n + 1)
    p_ge = (np.sum(null_or_boot >= 0) + 1) / (n + 1)
    if alternative == "greater":
        return float(min(1.0, p_le))
    if alternative == "less":
        return float(min(1.0, p_ge))
    return float(min(1.0, 2 * min(p_le, p_ge)))


def _statistic_fn(statistic: str | Callable, q: float | None = None) -> tuple[Callable, str]:
    if callable(statistic):
        return statistic, "custom_diff"
    if statistic == "mean":
        return (lambda a, axis: np.mean(a, axis=axis)), "mean_diff"
    if statistic == "median":
        return (lambda a, axis: np.median(a, axis=axis)), "median_diff"
    if statistic == "quantile":
        if q is None or not 0 < q < 1:
            raise ValueError("statistic='quantile' requires 0 < q < 1")
        return (lambda a, axis: np.quantile(a, q, axis=axis)), "quantile_diff"
    raise ValueError("statistic must be 'mean', 'median', 'quantile' or a callable(a, axis)")


def _batch_size(n_resamples: int, row_len: int, max_cells: int = 4_000_000) -> int:
    return int(max(1, min(n_resamples, max_cells // max(row_len, 1))))


# --------------------------------------------------------------------------- средние


def welch_ttest(control, experiment, alpha: float = 0.05, alternative: str = "two-sided"):
    """t-тест Уэлча (неравные дисперсии) для разницы средних. Дефолт для A/B."""
    _check_alternative(alternative)
    c, e = _clean(control), _clean(experiment)
    if len(c) < 2 or len(e) < 2:
        return StatTestResult(test="welch", estimand="mean_diff")
    v_c, v_e = c.var(ddof=1) / len(c), e.var(ddof=1) / len(e)
    se = float(np.sqrt(v_c + v_e))
    df = _welch_df(v_c, len(c), v_e, len(e)) if se > 0 else np.nan
    return _mean_diff_result("welch", float(e.mean() - c.mean()), se, df, alpha, alternative)


def student_ttest(control, experiment, alpha: float = 0.05, alternative: str = "two-sided"):
    """Классический t-тест Стьюдента (равные дисперсии). Обычно лучше ``welch_ttest``."""
    _check_alternative(alternative)
    c, e = _clean(control), _clean(experiment)
    if len(c) < 2 or len(e) < 2:
        return StatTestResult(test="student", estimand="mean_diff")
    df = len(c) + len(e) - 2
    pooled = ((len(c) - 1) * c.var(ddof=1) + (len(e) - 1) * e.var(ddof=1)) / df
    se = float(np.sqrt(pooled * (1 / len(c) + 1 / len(e))))
    return _mean_diff_result("student", float(e.mean() - c.mean()), se, float(df), alpha, alternative)


def z_test_means(control, experiment, alpha: float = 0.05, alternative: str = "two-sided"):
    """z-тест разницы средних (нормальная аппроксимация, большие выборки)."""
    _check_alternative(alternative)
    c, e = _clean(control), _clean(experiment)
    if len(c) < 2 or len(e) < 2:
        return StatTestResult(test="ztest", estimand="mean_diff")
    se = float(np.sqrt(c.var(ddof=1) / len(c) + e.var(ddof=1) / len(e)))
    return _mean_diff_result("ztest", float(e.mean() - c.mean()), se, np.inf, alpha, alternative)


def _trimmed_parts(x: np.ndarray, trim: float):
    n = len(x)
    g = int(np.floor(trim * n))
    h = n - 2 * g
    xs = np.sort(x)
    trimmed_mean = xs[g : n - g].mean()
    wins = xs.copy()
    if g > 0:
        wins[:g] = xs[g]
        wins[n - g :] = xs[n - g - 1]
    d = (n - 1) * wins.var(ddof=1) / (h * (h - 1)) if h > 1 else np.nan
    return trimmed_mean, d, h


def yuen_trimmed_ttest(
    control, experiment, trim: float = 0.1, alpha: float = 0.05, alternative: str = "two-sided"
):
    """Тест Юэна: Welch-тест для усечённых средних (устойчив к тяжёлым хвостам).

    ``trim`` — доля наблюдений, отрезаемая с каждого хвоста. Оценивает разницу
    усечённых средних, а не обычных.
    """
    _check_alternative(alternative)
    if not 0 <= trim < 0.5:
        raise ValueError("trim must be in [0, 0.5)")
    c, e = _clean(control), _clean(experiment)
    if len(c) < 2 or len(e) < 2:
        return StatTestResult(test="yuen", estimand="trimmed_mean_diff")
    tm_c, d_c, h_c = _trimmed_parts(c, trim)
    tm_e, d_e, h_e = _trimmed_parts(e, trim)
    if h_c < 2 or h_e < 2:
        return StatTestResult(test="yuen", estimand="trimmed_mean_diff")
    se = float(np.sqrt(d_c + d_e))
    den = d_c**2 / (h_c - 1) + d_e**2 / (h_e - 1)
    df = float((d_c + d_e) ** 2 / den) if den > 0 else np.nan
    return _mean_diff_result(
        "yuen",
        float(tm_e - tm_c),
        se,
        df,
        alpha,
        alternative,
        estimand="trimmed_mean_diff",
        extra={"trim": trim},
    )


def bootstrap_test(
    control,
    experiment,
    statistic: str | Callable = "mean",
    q: float | None = None,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    n_resamples: int = 10_000,
    random_state=0,
    ci_method: str = "percentile",
):
    """Bootstrap разницы статистик (среднее, медиана, квантиль, произвольная функция).

    Группы ресемплируются независимо с возвращением. ``ci_method``:
    ``percentile`` или ``basic`` (reverse percentile). p-value — обращение
    percentile-интервала относительно нуля.
    """
    _check_alternative(alternative)
    if ci_method not in ("percentile", "basic"):
        raise ValueError("ci_method must be 'percentile' or 'basic'")
    fn, estimand = _statistic_fn(statistic, q)
    c, e = _clean(control), _clean(experiment)
    name = "bootstrap"
    if len(c) < 2 or len(e) < 2 or n_resamples < 1:
        return StatTestResult(test=name, estimand=estimand)
    rng = _rng(random_state)
    observed = float(fn(e, None) - fn(c, None))
    diffs = np.empty(n_resamples)
    batch = _batch_size(n_resamples, max(len(c), len(e)))
    done = 0
    while done < n_resamples:
        b = min(batch, n_resamples - done)
        boot_c = c[rng.integers(0, len(c), size=(b, len(c)))]
        boot_e = e[rng.integers(0, len(e), size=(b, len(e)))]
        diffs[done : done + b] = fn(boot_e, 1) - fn(boot_c, 1)
        done += b
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    if ci_method == "basic":
        lo, hi = 2 * observed - hi, 2 * observed - lo
    return StatTestResult(
        test=name,
        statistic=observed,
        p_value=_resample_p_value(diffs, observed, alternative, "boot"),
        estimate=observed,
        ci_lower=float(lo),
        ci_upper=float(hi),
        estimand=estimand,
        extra={"n_resamples": n_resamples, "se": float(diffs.std(ddof=1)), "ci_method": ci_method},
    )


def quantile_test(control, experiment, q: float = 0.5, **kwargs):
    """Bootstrap-тест разницы квантилей (``q=0.5`` — медиана, ``q=0.9`` — p90)."""
    return bootstrap_test(control, experiment, statistic="quantile", q=q, **kwargs)


def permutation_test(
    control,
    experiment,
    statistic: str | Callable = "mean",
    q: float | None = None,
    alternative: str = "two-sided",
    n_resamples: int = 10_000,
    random_state=0,
):
    """Перестановочный тест разницы статистик (точный контроль ошибки I рода).

    Нулевая гипотеза — взаимозаменяемость групп (одинаковые распределения).
    p-value считается с поправкой ``(k + 1) / (B + 1)``.
    """
    _check_alternative(alternative)
    fn, estimand = _statistic_fn(statistic, q)
    c, e = _clean(control), _clean(experiment)
    if len(c) < 1 or len(e) < 1 or n_resamples < 1:
        return StatTestResult(test="permutation", estimand=estimand)
    rng = _rng(random_state)
    observed = float(fn(e, None) - fn(c, None))
    pooled = np.concatenate([c, e])
    n_c = len(c)
    null = np.empty(n_resamples)
    batch = _batch_size(n_resamples, len(pooled))
    done = 0
    while done < n_resamples:
        b = min(batch, n_resamples - done)
        shuffled = rng.permuted(np.tile(pooled, (b, 1)), axis=1)
        null[done : done + b] = fn(shuffled[:, n_c:], 1) - fn(shuffled[:, :n_c], 1)
        done += b
    return StatTestResult(
        test="permutation",
        statistic=observed,
        p_value=_resample_p_value(null, observed, alternative, "perm"),
        estimate=observed,
        estimand=estimand,
        extra={"n_resamples": n_resamples},
    )


# --------------------------------------------------------------------------- ранги и распределения


def _prob_superiority(c: np.ndarray, e: np.ndarray) -> float:
    """P(E > C) + 0.5 * P(E == C) через ранги."""
    ranks = stats.rankdata(np.concatenate([c, e]))
    u_e = ranks[len(c) :].sum() - len(e) * (len(e) + 1) / 2
    return float(u_e / (len(c) * len(e)))


def mann_whitney_test(control, experiment, alternative: str = "two-sided", **_):
    """U-тест Манна—Уитни. Сравнивает ранговое положение распределений, не медианы.

    ``estimate`` — вероятность превосходства ``P(E > C) + 0.5 P(E = C)``;
    в ``extra`` — ранг-бисериальная корреляция (= дельта Клиффа).
    """
    _check_alternative(alternative)
    c, e = _clean(control), _clean(experiment)
    if len(c) == 0 or len(e) == 0:
        return StatTestResult(test="mannwhitney", estimand="prob_superiority")
    res = stats.mannwhitneyu(e, c, alternative=alternative)
    ps = float(res.statistic / (len(c) * len(e)))
    return StatTestResult(
        test="mannwhitney",
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        estimate=ps,
        estimand="prob_superiority",
        extra={"rank_biserial": 2 * ps - 1, "median_diff": float(np.median(e) - np.median(c))},
    )


def brunner_munzel_test(control, experiment, alpha: float = 0.05, alternative: str = "two-sided"):
    """Тест Бруннера—Мунцеля: H0 ``P(E > C) + 0.5 P(E = C) = 0.5``.

    В отличие от Манна—Уитни не требует равенства дисперсий/форм распределений
    (обобщённая проблема Беренса—Фишера). Возвращает CI для вероятности
    превосходства.
    """
    _check_alternative(alternative)
    c, e = _clean(control), _clean(experiment)
    name, estimand = "brunner_munzel", "prob_superiority"
    n_c, n_e = len(c), len(e)
    if n_c < 2 or n_e < 2:
        return StatTestResult(test=name, estimand=estimand)
    rank_all = stats.rankdata(np.concatenate([c, e]))
    rank_all_c, rank_all_e = rank_all[:n_c], rank_all[n_c:]
    rank_c, rank_e = stats.rankdata(c), stats.rankdata(e)
    s_c = np.sum((rank_all_c - rank_c - rank_all_c.mean() + rank_c.mean()) ** 2) / (n_c - 1)
    s_e = np.sum((rank_all_e - rank_e - rank_all_e.mean() + rank_e.mean()) ** 2) / (n_e - 1)
    p_hat = float((rank_all_e.mean() - rank_all_c.mean()) / (n_c + n_e) + 0.5)
    var_sum = n_c * s_c + n_e * s_e
    if var_sum <= 0:
        return StatTestResult(test=name, estimate=p_hat, estimand=estimand)
    se = float(np.sqrt(var_sum) / (n_c * n_e))
    df = float(var_sum**2 / ((n_c * s_c) ** 2 / (n_c - 1) + (n_e * s_e) ** 2 / (n_e - 1)))
    stat = (p_hat - 0.5) / se
    margin = _crit(df, alpha) * se
    return StatTestResult(
        test=name,
        statistic=float(stat),
        p_value=_p_from_t(stat, df, alternative),
        estimate=p_hat,
        ci_lower=float(max(0.0, p_hat - margin)),
        ci_upper=float(min(1.0, p_hat + margin)),
        estimand=estimand,
        extra={"se": se, "df": df},
    )


def ks_test(control, experiment, **_):
    """Двухвыборочный тест Колмогорова—Смирнова: любые различия распределений."""
    c, e = _clean(control), _clean(experiment)
    if len(c) == 0 or len(e) == 0:
        return StatTestResult(test="ks", estimand="distribution")
    res = stats.ks_2samp(c, e)
    return StatTestResult(
        test="ks", statistic=float(res.statistic), p_value=float(res.pvalue), estimand="distribution"
    )


def cramer_von_mises_test(control, experiment, **_):
    """Двухвыборочный тест Крамера—фон Мизеса (чувствительнее KS к сдвигам в центре)."""
    c, e = _clean(control), _clean(experiment)
    if len(c) < 2 or len(e) < 2:
        return StatTestResult(test="cvm", estimand="distribution")
    res = stats.cramervonmises_2samp(c, e)
    return StatTestResult(
        test="cvm", statistic=float(res.statistic), p_value=float(res.pvalue), estimand="distribution"
    )


def mood_median_test(control, experiment, **_):
    """Медианный тест Муда: доли наблюдений выше общей медианы (хи-квадрат)."""
    c, e = _clean(control), _clean(experiment)
    name = "mood"
    if len(c) == 0 or len(e) == 0:
        return StatTestResult(test=name, estimand="median_diff")
    delta = float(np.median(e) - np.median(c))
    try:
        res = stats.median_test(c, e)
    except ValueError:
        return StatTestResult(test=name, estimate=delta, estimand="median_diff")
    return StatTestResult(
        test=name,
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        estimate=delta,
        estimand="median_diff",
        extra={"grand_median": float(res.median)},
    )


# --------------------------------------------------------------------------- доли

_PROPORTION_METHODS = ("ztest", "chi2", "chi2_yates", "g_test", "fisher", "boschloo", "barnard")
_EXACT_UNCONDITIONAL_MAX_N = 2_000


def proportion_test(
    control_success: int,
    control_n: int,
    experiment_success: int,
    experiment_n: int,
    method: str = "ztest",
    alpha: float = 0.05,
    alternative: str = "two-sided",
):
    """Сравнение двух долей.

    ``method``: ``ztest`` (pooled z-тест), ``chi2``, ``chi2_yates``, ``g_test``
    (likelihood-ratio), ``fisher`` (точный условный), ``boschloo`` / ``barnard``
    (точные безусловные, только для малых выборок). CI разницы долей —
    Newcombe—Wilson. В ``extra``: ``risk_ratio`` с лог-CI и ``odds_ratio``.
    """
    _check_alternative(alternative)
    if method not in _PROPORTION_METHODS:
        raise ValueError(f"method must be one of {_PROPORTION_METHODS}, got {method!r}")
    s_c, n_c, s_e, n_e = int(control_success), int(control_n), int(experiment_success), int(experiment_n)
    if n_c <= 0 or n_e <= 0:
        return StatTestResult(test=method, estimand="proportion_diff")
    if not (0 <= s_c <= n_c and 0 <= s_e <= n_e):
        raise ValueError("successes must be within [0, n]")
    p_c, p_e = s_c / n_c, s_e / n_e
    delta = p_e - p_c
    table = np.array([[s_e, n_e - s_e], [s_c, n_c - s_c]])
    statistic, p_value = np.nan, np.nan
    degenerate = (s_c + s_e == 0) or (s_c + s_e == n_c + n_e)

    if method == "ztest":
        p_pool = (s_c + s_e) / (n_c + n_e)
        se0 = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_e))
        if se0 > 0:
            statistic = delta / se0
            p_value = _p_from_t(statistic, np.inf, alternative)
    elif method in ("chi2", "chi2_yates", "g_test"):
        if alternative != "two-sided":
            raise ValueError(f"{method} supports only alternative='two-sided'")
        if not degenerate:
            res = stats.chi2_contingency(
                table,
                correction=(method == "chi2_yates"),
                lambda_="log-likelihood" if method == "g_test" else None,
            )
            statistic, p_value = float(res[0]), float(res[1])
    elif method == "fisher":
        statistic, p_value = stats.fisher_exact(table, alternative=alternative)
    else:
        if n_c + n_e > _EXACT_UNCONDITIONAL_MAX_N:
            raise ValueError(
                f"{method} is an exact unconditional test and is practical only for "
                f"n_total <= {_EXACT_UNCONDITIONAL_MAX_N}; use 'ztest' or 'fisher'"
            )
        fn = stats.boschloo_exact if method == "boschloo" else stats.barnard_exact
        res = fn(table.T, alternative=alternative)
        statistic, p_value = float(res.statistic), float(res.pvalue)

    ci_lower, ci_upper = proportion_difference_ci(s_c, n_c, s_e, n_e, alpha)
    extra: dict[str, Any] = {"p_control": p_c, "p_experiment": p_e}
    if s_c > 0 and s_e > 0:
        rr = p_e / p_c
        se_log = np.sqrt(1 / s_e - 1 / n_e + 1 / s_c - 1 / n_c)
        z = stats.norm.ppf(1 - alpha / 2)
        extra.update(
            risk_ratio=float(rr),
            risk_ratio_ci_lower=float(np.exp(np.log(rr) - z * se_log)),
            risk_ratio_ci_upper=float(np.exp(np.log(rr) + z * se_log)),
        )
    if 0 < s_c < n_c and 0 < s_e < n_e:
        extra["odds_ratio"] = float((s_e / (n_e - s_e)) / (s_c / (n_c - s_c)))
    return StatTestResult(
        test=method,
        statistic=float(statistic) if np.isfinite(statistic) else np.nan,
        p_value=float(p_value) if np.isfinite(p_value) else np.nan,
        estimate=float(delta),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        estimand="proportion_diff",
        extra=extra,
    )


# --------------------------------------------------------------------------- ratio-метрики


def _ratio_inputs(num, den):
    num = np.asarray(num, dtype=float).ravel()
    den = np.asarray(den, dtype=float).ravel()
    mask = np.isfinite(num) & np.isfinite(den)
    return num[mask], den[mask]


def ratio_group_variance(num, den) -> tuple[float, float, int]:
    """Оценка ``sum(num)/sum(den)`` и её дисперсия дельта-методом для одной группы."""
    num, den = _ratio_inputs(num, den)
    n = len(num)
    if n < 2 or den.mean() == 0:
        return np.nan, np.nan, n
    ratio = float(num.mean() / den.mean())
    influence = num - ratio * den
    return ratio, float(influence.var(ddof=1) / (n * den.mean() ** 2)), n


def ratio_delta_test(
    control_num,
    control_den,
    experiment_num,
    experiment_den,
    alpha: float = 0.05,
    alternative: str = "two-sided",
):
    """Дельта-метод для разницы ratio-метрик (CTR, CPC, CPM, CPA: сумма/сумма).

    Юнит рандомизации — строка (пользователь); числитель и знаменатель
    агрегированы до него. Дисперсия учитывает ковариацию числителя и знаменателя.
    """
    _check_alternative(alternative)
    r_c, v_c, n_c = ratio_group_variance(control_num, control_den)
    r_e, v_e, n_e = ratio_group_variance(experiment_num, experiment_den)
    if not all(np.isfinite(x) for x in (r_c, v_c, r_e, v_e)):
        return StatTestResult(test="delta", estimand="ratio_diff")
    se = float(np.sqrt(v_c + v_e))
    df = _welch_df(v_c, n_c, v_e, n_e) if se > 0 else np.nan
    return _mean_diff_result(
        "delta",
        r_e - r_c,
        se,
        df,
        alpha,
        alternative,
        estimand="ratio_diff",
        extra={"ratio_control": r_c, "ratio_experiment": r_e},
    )


def ratio_linearization_test(
    control_num,
    control_den,
    experiment_num,
    experiment_den,
    alpha: float = 0.05,
    alternative: str = "two-sided",
):
    """Линеаризация ratio-метрики: Welch-тест по ``num - R_control * den``.

    CI переведён в шкалу ratio делением на средний знаменатель эксперимента
    (приближение первого порядка).
    """
    _check_alternative(alternative)
    c_num, c_den = _ratio_inputs(control_num, control_den)
    e_num, e_den = _ratio_inputs(experiment_num, experiment_den)
    name = "linearization"
    if len(c_num) < 2 or len(e_num) < 2 or c_den.sum() == 0 or e_den.mean() == 0:
        return StatTestResult(test=name, estimand="ratio_diff")
    coeff = c_num.sum() / c_den.sum()
    res = welch_ttest(c_num - coeff * c_den, e_num - coeff * e_den, alpha, alternative)
    scale = float(e_den.mean())
    return StatTestResult(
        test=name,
        statistic=res.statistic,
        p_value=res.p_value,
        estimate=float(e_num.sum() / e_den.sum() - coeff),
        ci_lower=res.ci_lower / scale,
        ci_upper=res.ci_upper / scale,
        estimand="ratio_diff",
        extra={"linearization_coeff": float(coeff), "df": res.extra.get("df")},
    )


def ratio_bootstrap_test(
    control_num,
    control_den,
    experiment_num,
    experiment_den,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    n_resamples: int = 10_000,
    random_state=0,
):
    """Bootstrap разницы ratio-метрик: ресемплируются пользователи (пары num, den)."""
    _check_alternative(alternative)
    c_num, c_den = _ratio_inputs(control_num, control_den)
    e_num, e_den = _ratio_inputs(experiment_num, experiment_den)
    name = "bootstrap"
    if len(c_num) < 2 or len(e_num) < 2 or c_den.sum() == 0 or e_den.sum() == 0:
        return StatTestResult(test=name, estimand="ratio_diff")
    rng = _rng(random_state)
    observed = float(e_num.sum() / e_den.sum() - c_num.sum() / c_den.sum())
    diffs = np.empty(n_resamples)
    batch = _batch_size(n_resamples, max(len(c_num), len(e_num)))
    done = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        while done < n_resamples:
            b = min(batch, n_resamples - done)
            idx_c = rng.integers(0, len(c_num), size=(b, len(c_num)))
            idx_e = rng.integers(0, len(e_num), size=(b, len(e_num)))
            diffs[done : done + b] = e_num[idx_e].sum(axis=1) / e_den[idx_e].sum(axis=1) - c_num[
                idx_c
            ].sum(axis=1) / c_den[idx_c].sum(axis=1)
            done += b
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) < 2:
        return StatTestResult(test=name, estimate=observed, estimand="ratio_diff")
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return StatTestResult(
        test=name,
        statistic=observed,
        p_value=_resample_p_value(diffs, observed, alternative, "boot"),
        estimate=observed,
        ci_lower=float(lo),
        ci_upper=float(hi),
        estimand="ratio_diff",
        extra={"n_resamples": int(len(diffs)), "se": float(diffs.std(ddof=1))},
    )


# --------------------------------------------------------------------------- связанные выборки


def _paired(before, after) -> np.ndarray:
    before = np.asarray(before, dtype=float).ravel()
    after = np.asarray(after, dtype=float).ravel()
    if len(before) != len(after):
        raise ValueError("paired samples must have the same length")
    diff = after - before
    return diff[np.isfinite(diff)]


def paired_ttest(before, after, alpha: float = 0.05, alternative: str = "two-sided"):
    """Парный t-тест (один и тот же юнит до/после, switchback, matched pairs)."""
    _check_alternative(alternative)
    d = _paired(before, after)
    if len(d) < 2:
        return StatTestResult(test="paired_t", estimand="mean_diff")
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return _mean_diff_result("paired_t", float(d.mean()), se, float(len(d) - 1), alpha, alternative)


def wilcoxon_signed_rank_test(before, after, alternative: str = "two-sided", **_):
    """Критерий знаковых рангов Уилкоксона для связанных выборок."""
    _check_alternative(alternative)
    d = _paired(before, after)
    if len(d) == 0 or np.all(d == 0):
        return StatTestResult(test="wilcoxon", estimand="median_diff")
    res = stats.wilcoxon(d, alternative=alternative)
    return StatTestResult(
        test="wilcoxon",
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        estimate=float(np.median(d)),
        estimand="median_diff",
    )


def sign_test(before, after, alternative: str = "two-sided", **_):
    """Критерий знаков для связанных выборок (нулевые разности отбрасываются)."""
    _check_alternative(alternative)
    d = _paired(before, after)
    pos, neg = int(np.sum(d > 0)), int(np.sum(d < 0))
    if pos + neg == 0:
        return StatTestResult(test="sign", estimand="prob_superiority")
    res = stats.binomtest(pos, pos + neg, 0.5, alternative=alternative)
    return StatTestResult(
        test="sign",
        statistic=float(pos),
        p_value=float(res.pvalue),
        estimate=pos / (pos + neg),
        estimand="prob_superiority",
        extra={"n_positive": pos, "n_negative": neg},
    )


def mcnemar_test(n_only_before: int, n_only_after: int, exact: bool | None = None):
    """Тест Мак-Немара для связанных бинарных исходов.

    ``n_only_before`` — юнитов с успехом только «до», ``n_only_after`` — только
    «после». ``exact=None`` выбирает точный биномиальный вариант при < 25
    дискордантных пар, иначе хи-квадрат с поправкой на непрерывность.
    """
    b, c = int(n_only_before), int(n_only_after)
    n = b + c
    if n == 0:
        return StatTestResult(test="mcnemar", estimand="proportion_diff")
    if exact is None:
        exact = n < 25
    if exact:
        statistic = float(min(b, c))
        p_value = float(stats.binomtest(c, n, 0.5).pvalue)
    else:
        statistic = (abs(b - c) - 1) ** 2 / n
        p_value = float(stats.chi2.sf(statistic, 1))
    return StatTestResult(
        test="mcnemar",
        statistic=float(statistic),
        p_value=p_value,
        estimand="proportion_diff",
        extra={"exact": bool(exact), "discordant_pairs": n},
    )


# --------------------------------------------------------------------------- эквивалентность


def _welch_parts(control, experiment):
    c, e = _clean(control), _clean(experiment)
    if len(c) < 2 or len(e) < 2:
        return None
    v_c, v_e = c.var(ddof=1) / len(c), e.var(ddof=1) / len(e)
    se = float(np.sqrt(v_c + v_e))
    if se == 0:
        return None
    return float(e.mean() - c.mean()), se, _welch_df(v_c, len(c), v_e, len(e))


def tost_equivalence_test(control, experiment, low: float, upp: float, alpha: float = 0.05):
    """TOST: проверка эквивалентности ``low < mean(E) - mean(C) < upp`` (Welch).

    H0 — эффект вне коридора. Малый p-value означает, что эффект
    статистически укладывается в границы практической незначимости.
    CI уровня ``1 - 2 * alpha``, как принято для TOST.
    """
    if not low < upp:
        raise ValueError("low must be less than upp")
    parts = _welch_parts(control, experiment)
    if parts is None:
        return StatTestResult(test="tost", estimand="mean_diff")
    delta, se, df = parts
    p_low = float(stats.t.sf((delta - low) / se, df))
    p_upp = float(stats.t.cdf((delta - upp) / se, df))
    margin = float(stats.t.ppf(1 - alpha, df) * se)
    return StatTestResult(
        test="tost",
        statistic=float(min((delta - low) / se, (upp - delta) / se)),
        p_value=max(p_low, p_upp),
        estimate=delta,
        ci_lower=delta - margin,
        ci_upper=delta + margin,
        estimand="mean_diff",
        extra={"p_lower": p_low, "p_upper": p_upp, "low": low, "upp": upp},
    )


def noninferiority_test(
    control, experiment, margin: float, higher_is_better: bool = True, alpha: float = 0.05
):
    """Тест «не хуже чем» (non-inferiority) для разницы средних (Welch).

    ``margin > 0`` — допустимое ухудшение в абсолютных единицах. H0: эксперимент
    хуже контроля более чем на ``margin``.
    """
    if margin <= 0:
        raise ValueError("margin must be positive")
    parts = _welch_parts(control, experiment)
    if parts is None:
        return StatTestResult(test="noninferiority", estimand="mean_diff")
    delta, se, df = parts
    if higher_is_better:
        stat = (delta + margin) / se
        p_value = float(stats.t.sf(stat, df))
    else:
        stat = (delta - margin) / se
        p_value = float(stats.t.cdf(stat, df))
    half = float(stats.t.ppf(1 - alpha, df) * se)
    return StatTestResult(
        test="noninferiority",
        statistic=float(stat),
        p_value=p_value,
        estimate=delta,
        ci_lower=delta - half,
        ci_upper=delta + half,
        estimand="mean_diff",
        extra={"margin": margin, "higher_is_better": higher_is_better},
    )


# --------------------------------------------------------------------------- счётчики


def poisson_rate_test(
    control_events: int,
    control_exposure: float,
    experiment_events: int,
    experiment_exposure: float,
    alternative: str = "two-sided",
):
    """Точный условный тест отношения пуассоновских интенсивностей.

    Корректен только для независимых событий (без кластеризации по
    пользователю); для user-level метрик используйте ``ratio_delta_test``.
    """
    _check_alternative(alternative)
    k_c, k_e = int(control_events), int(experiment_events)
    if control_exposure <= 0 or experiment_exposure <= 0 or k_c + k_e == 0:
        return StatTestResult(test="poisson_rate", estimand="rate_ratio")
    share = experiment_exposure / (control_exposure + experiment_exposure)
    res = stats.binomtest(k_e, k_c + k_e, share, alternative=alternative)
    rate_c, rate_e = k_c / control_exposure, k_e / experiment_exposure
    return StatTestResult(
        test="poisson_rate",
        statistic=float(k_e),
        p_value=float(res.pvalue),
        estimate=float(rate_e / rate_c) if rate_c > 0 else np.nan,
        estimand="rate_ratio",
        extra={"rate_control": rate_c, "rate_experiment": rate_e},
    )


# --------------------------------------------------------------------------- диспетчер по типам метрик

DEFAULT_TEST_BY_TYPE = {
    "additive": "welch",
    "average": "welch",
    "ratio": "linearization",
    "share": "ztest",
    "median": "mannwhitney",
}

_VALUE_TESTS: dict[str, Callable] = {
    "welch": welch_ttest,
    "student": student_ttest,
    "ztest": z_test_means,
    "yuen": yuen_trimmed_ttest,
    "mannwhitney": mann_whitney_test,
    "brunner_munzel": brunner_munzel_test,
    "ks": ks_test,
    "cvm": cramer_von_mises_test,
}

# Совместные post-hoc процедуры (нужны все группы сразу) — только для ABC/ABCD/N групп.
JOINT_POSTHOC_TESTS = ("dunn", "tukey", "games_howell", "dunnett")

ALLOWED_TESTS_BY_TYPE = {
    "additive": (*_VALUE_TESTS, "permutation", "bootstrap"),
    "average": (*_VALUE_TESTS, "permutation", "bootstrap"),
    "median": ("mannwhitney", "brunner_munzel", "mood", "bootstrap", "permutation", "ks", "cvm"),
    "share": _PROPORTION_METHODS,
    "ratio": ("linearization", "delta", "bootstrap"),
}

ESTIMAND_BY_TYPE = {
    "additive": "mean_diff",
    "average": "mean_diff",
    "median": "median_diff",
    "share": "proportion_diff",
    "ratio": "ratio_diff",
}


def available_tests(metric_type: str | None = None, multi_group: bool = False):
    """Список допустимых значений колонки ``test`` конфига метрик."""

    def _with_joint(mtype: str) -> tuple[str, ...]:
        tests = tuple(ALLOWED_TESTS_BY_TYPE[mtype])
        if not multi_group:
            return tests
        if mtype in ("additive", "average"):
            return tests + JOINT_POSTHOC_TESTS
        if mtype == "median":
            return tests + ("dunn",)
        return tests

    if metric_type is None:
        return {mtype: _with_joint(mtype) for mtype in ALLOWED_TESTS_BY_TYPE}
    if metric_type not in ALLOWED_TESTS_BY_TYPE:
        raise ValueError(f"Unsupported metric type: {metric_type}")
    return _with_joint(metric_type)


def run_metric_test(
    control_df,
    experiment_df,
    metric_type: str,
    test: str | None = None,
    alpha: float = 0.05,
    n_resamples: int = 10_000,
    random_state=0,
) -> StatTestResult:
    """Запустить критерий ``test`` на фреймах, подготовленных ``_prep_metric_pair_frames``.

    Фреймы содержат колонки ``num``, ``den``, ``value``.
    """
    if metric_type not in ALLOWED_TESTS_BY_TYPE:
        raise ValueError(f"Unsupported metric type: {metric_type}")
    test = DEFAULT_TEST_BY_TYPE[metric_type] if test is None else str(test).lower()
    if test not in ALLOWED_TESTS_BY_TYPE[metric_type]:
        raise ValueError(
            f"Test {test!r} is not supported for metric type {metric_type!r}; "
            f"allowed: {list(ALLOWED_TESTS_BY_TYPE[metric_type])}"
        )

    if metric_type == "share":
        c_valid = (control_df["den"] > 0) & control_df["num"].notna()
        e_valid = (experiment_df["den"] > 0) & experiment_df["num"].notna()
        return proportion_test(
            int((c_valid & (control_df["num"] > 0)).sum()),
            int(c_valid.sum()),
            int((e_valid & (experiment_df["num"] > 0)).sum()),
            int(e_valid.sum()),
            method=test,
            alpha=alpha,
        )

    if metric_type == "ratio":
        args = (control_df["num"], control_df["den"], experiment_df["num"], experiment_df["den"])
        if test == "delta":
            return ratio_delta_test(*args, alpha=alpha)
        if test == "bootstrap":
            return ratio_bootstrap_test(
                *args, alpha=alpha, n_resamples=n_resamples, random_state=random_state
            )
        return ratio_linearization_test(*args, alpha=alpha)

    control, experiment = control_df["value"], experiment_df["value"]
    statistic = "median" if metric_type == "median" else "mean"
    if test == "bootstrap":
        return bootstrap_test(
            control,
            experiment,
            statistic=statistic,
            alpha=alpha,
            n_resamples=n_resamples,
            random_state=random_state,
        )
    if test == "permutation":
        return permutation_test(
            control, experiment, statistic=statistic, n_resamples=n_resamples, random_state=random_state
        )
    if test == "mood":
        return mood_median_test(control, experiment)
    return _VALUE_TESTS[test](control, experiment, alpha=alpha)
