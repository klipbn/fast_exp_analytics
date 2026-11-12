"""Проверки качества эксперимента и выбор критерия.

SRM, пересечения групп, описательная статистика, нормальность, рекомендация
теста и A/A-симуляции (эмпирическая проверка ошибки I рода и мощности критерия
на реальном распределении метрики).
"""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

from . import stat_tests


def _clean(values) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    return array[np.isfinite(array)]


# --------------------------------------------------------------------------- SRM и дизайн


def srm_check(
    observed: Mapping[str, int],
    expected_shares: Mapping[str, float] | None = None,
    alpha: float = 0.001,
) -> dict:
    """Sample Ratio Mismatch: хи-квадрат согласия размеров групп с планом сплита.

    ``observed`` — число юнитов рандомизации в каждой группе;
    ``expected_shares`` — плановые доли (по умолчанию равные). Порог ``alpha``
    по умолчанию 0.001: SRM проверяется в каждом эксперименте, и ложные
    срабатывания дороги. При ``is_srm=True`` результатам эксперимента доверять
    нельзя, пока не найдена причина (бакетизация, логирование, фильтры).
    """
    names = list(observed)
    counts = np.array([float(observed[n]) for n in names])
    if len(names) < 2 or counts.sum() <= 0:
        raise ValueError("srm_check requires at least two groups with observations")
    if expected_shares is None:
        shares = np.full(len(names), 1 / len(names))
    else:
        missing = [n for n in names if n not in expected_shares]
        if missing:
            raise ValueError(f"expected_shares is missing groups: {missing}")
        shares = np.array([float(expected_shares[n]) for n in names])
        if (shares <= 0).any():
            raise ValueError("expected shares must be positive")
        shares = shares / shares.sum()
    expected = shares * counts.sum()
    statistic, p_value = stats.chisquare(counts, expected)
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "is_srm": bool(p_value < alpha),
        "alpha": alpha,
        "observed": dict(zip(names, counts.astype(int).tolist())),
        "expected": dict(zip(names, expected.tolist())),
        "observed_shares": dict(zip(names, (counts / counts.sum()).tolist())),
        "expected_shares": dict(zip(names, shares.tolist())),
    }


def srm_check_df(
    df: pd.DataFrame,
    group_col: str = "exp_group",
    unit_col: str | None = "user_id",
    expected_shares: Mapping[str, float] | None = None,
    alpha: float = 0.001,
) -> dict:
    """SRM по DataFrame: считает уникальные юниты (``unit_col``) в каждой группе."""
    if unit_col is None:
        counts = df[group_col].value_counts()
    else:
        counts = df.groupby(group_col)[unit_col].nunique()
    if expected_shares is not None:
        counts = counts.reindex(list(expected_shares)).fillna(0)
    return srm_check(counts.sort_index().to_dict(), expected_shares, alpha)


def check_unit_overlap(
    df: pd.DataFrame, unit_col: str = "user_id", group_col: str = "exp_group"
) -> dict:
    """Проверка независимости: юниты в нескольких группах и повторные строки на юнит.

    Статистические тесты предполагают одну строку на юнит рандомизации. Если
    ``n_duplicated_rows > 0`` — данные нужно агрегировать до юнита; если
    ``n_units_in_multiple_groups > 0`` — сломана бакетизация или джойн.
    """
    groups_per_unit = df.groupby(unit_col)[group_col].nunique()
    n_units = int(len(groups_per_unit))
    n_multi = int((groups_per_unit > 1).sum())
    n_dup = int(df.duplicated(subset=[unit_col, group_col]).sum())
    return {
        "n_rows": int(len(df)),
        "n_units": n_units,
        "n_units_in_multiple_groups": n_multi,
        "share_units_in_multiple_groups": n_multi / n_units if n_units else np.nan,
        "n_duplicated_rows": n_dup,
        "is_ok": n_multi == 0 and n_dup == 0,
    }


# --------------------------------------------------------------------------- описательная статистика


def _top_share(x: np.ndarray, share: float = 0.01) -> float:
    total = x.sum()
    if len(x) == 0 or total <= 0 or (x < 0).any():
        return np.nan
    k = max(1, int(np.ceil(len(x) * share)))
    return float(np.sort(x)[-k:].sum() / total)


def describe_groups(
    df: pd.DataFrame, value_col: str, group_col: str = "exp_group"
) -> pd.DataFrame:
    """Описательная статистика метрики по группам: квантили, доля нулей, хвост.

    ``top1pct_share`` — доля суммы метрики, приходящаяся на верхний 1% юнитов
    (индикатор того, что результат определяется несколькими крупными юнитами).
    """
    rows = []
    for group, part in df.groupby(group_col):
        x = _clean(pd.to_numeric(part[value_col], errors="coerce"))
        if len(x) == 0:
            rows.append({group_col: group, "n": 0})
            continue
        q = np.quantile(x, [0.25, 0.5, 0.75, 0.9, 0.99])
        rows.append(
            {
                group_col: group,
                "n": len(x),
                "mean": x.mean(),
                "std": x.std(ddof=1) if len(x) > 1 else np.nan,
                "median": q[1],
                "p25": q[0],
                "p75": q[2],
                "p90": q[3],
                "p99": q[4],
                "max": x.max(),
                "share_zeros": float(np.mean(x == 0)),
                "skewness": float(stats.skew(x)) if len(x) > 2 else np.nan,
                "top1pct_share": _top_share(x),
            }
        )
    return pd.DataFrame(rows).set_index(group_col)


def normality_report(values, max_shapiro_n: int = 5_000, random_state: int | None = 0) -> dict:
    """Асимметрия, эксцесс и тесты нормальности.

    На больших выборках тесты нормальности отвергают H0 почти всегда — выбирать
    критерий только по ним нельзя. Практичнее смотреть ``skewness`` и
    ``required_n_for_clt``: правило Кохави ``n >= 355 * skewness^2`` для
    применимости t-теста к среднему.
    """
    x = _clean(values)
    out = {
        "n": int(len(x)),
        "skewness": np.nan,
        "excess_kurtosis": np.nan,
        "shapiro_p_value": np.nan,
        "dagostino_p_value": np.nan,
        "required_n_for_clt": np.nan,
    }
    if len(x) < 3 or np.all(x == x[0]):
        return out
    skew = float(stats.skew(x))
    out["skewness"] = skew
    out["excess_kurtosis"] = float(stats.kurtosis(x))
    out["required_n_for_clt"] = float(np.ceil(355 * skew**2))
    sample = x
    if len(x) > max_shapiro_n:
        sample = np.random.default_rng(random_state).choice(x, size=max_shapiro_n, replace=False)
    out["shapiro_p_value"] = float(stats.shapiro(sample).pvalue)
    if len(x) >= 20:
        out["dagostino_p_value"] = float(stats.normaltest(x).pvalue)
    return out


# --------------------------------------------------------------------------- выбор критерия


def recommend_test(
    control,
    experiment,
    metric_type: str = "average",
    small_n: int = 30,
) -> dict:
    """Рекомендовать критерий по типу метрики, размеру выборки и форме распределения.

    Возвращает ``{"test", "reason", "alternatives", "checks"}``; ``test`` можно
    подставить в колонку ``test`` конфига метрик. Для ``share`` в ``control`` и
    ``experiment`` передаются бинарные (0/1) значения.

    Логика: бизнес-вопрос о среднем → Welch, пока выборка достаточна для ЦПТ
    при наблюдаемой асимметрии (``n >= 355 * skewness^2``); иначе bootstrap или
    перестановочный тест. Ранговые тесты отвечают на другой вопрос (ранговое
    положение распределений), поэтому предлагаются только как альтернатива.
    """
    if metric_type not in stat_tests.ALLOWED_TESTS_BY_TYPE:
        raise ValueError(f"Unsupported metric type: {metric_type}")
    c, e = _clean(control), _clean(experiment)
    n_min = int(min(len(c), len(e)))
    checks: dict = {"n_control": len(c), "n_experiment": len(e)}

    if metric_type == "share":
        s_c, s_e = float((c > 0).sum()), float((e > 0).sum())
        total = len(c) + len(e)
        p_pool = (s_c + s_e) / total if total else np.nan
        min_expected = n_min * min(p_pool, 1 - p_pool) if total else np.nan
        checks["min_expected_count"] = min_expected
        if np.isfinite(min_expected) and min_expected < 5:
            return {
                "test": "fisher",
                "reason": "ожидаемое число успехов/неуспехов в ячейке < 5 — нормальная "
                "аппроксимация ненадёжна, нужен точный тест",
                "alternatives": ["boschloo"],
                "checks": checks,
            }
        return {
            "test": "ztest",
            "reason": "бинарная метрика с достаточными ожидаемыми частотами — z-тест долей",
            "alternatives": ["chi2", "g_test"],
            "checks": checks,
        }

    if metric_type == "ratio":
        test = "delta" if n_min >= 200 else "bootstrap"
        reason = (
            "ratio-метрика (сумма/сумма): дельта-метод учитывает ковариацию числителя и знаменателя"
            if test == "delta"
            else "ratio-метрика на малой выборке: асимптотика дельта-метода ненадёжна, bootstrap по юнитам"
        )
        return {
            "test": test,
            "reason": reason,
            "alternatives": ["linearization", "bootstrap"],
            "checks": checks,
        }

    skew = max(
        (abs(float(stats.skew(x))) for x in (c, e) if len(x) > 2 and not np.all(x == x[0])),
        default=0.0,
    )
    required_n = float(np.ceil(355 * skew**2))
    checks.update(
        max_abs_skewness=skew,
        required_n_for_clt=required_n,
        top1pct_share_control=_top_share(c),
        top1pct_share_experiment=_top_share(e),
    )

    if metric_type == "median":
        iqr = [np.subtract(*np.quantile(x, [0.75, 0.25])) for x in (c, e) if len(x)]
        spread_ratio = max(iqr) / min(iqr) if len(iqr) == 2 and min(iqr) > 0 else np.nan
        checks["iqr_ratio"] = spread_ratio
        if np.isfinite(spread_ratio) and spread_ratio > 1.5:
            return {
                "test": "brunner_munzel",
                "reason": "разброс групп заметно различается — Манн—Уитни теряет контроль "
                "ошибки I рода, Бруннер—Мунцель устойчив к разным формам",
                "alternatives": ["bootstrap", "mannwhitney"],
                "checks": checks,
            }
        return {
            "test": "mannwhitney",
            "reason": "ранговое сравнение при сопоставимых формах распределений",
            "alternatives": ["brunner_munzel", "bootstrap"],
            "checks": checks,
        }

    if n_min < small_n:
        return {
            "test": "permutation",
            "reason": f"малая выборка (n < {small_n}): перестановочный тест точно "
            "контролирует ошибку I рода без нормальной аппроксимации",
            "alternatives": ["mannwhitney", "welch"],
            "checks": checks,
        }
    if n_min >= required_n:
        return {
            "test": "welch",
            "reason": "выборка достаточна для ЦПТ при наблюдаемой асимметрии "
            f"(n >= 355 * skew^2 = {required_n:.0f}) — Welch t-test для разницы средних",
            "alternatives": ["bootstrap", "permutation"],
            "checks": checks,
        }
    return {
        "test": "bootstrap",
        "reason": "тяжёлый хвост: выборки недостаточно для нормальной аппроксимации "
        f"среднего (нужно n >= {required_n:.0f}). Bootstrap для разницы средних; "
        "дополнительно стоит ограничить выбросы (winsorize) и применить CUPED",
        "alternatives": ["permutation", "yuen", "mannwhitney"],
        "checks": checks,
    }


# --------------------------------------------------------------------------- симуляции


def _resolve_test_callable(test) -> Callable:
    if callable(test):
        return test
    name = str(test).lower()
    registry = {
        **stat_tests._VALUE_TESTS,
        "permutation": stat_tests.permutation_test,
        "bootstrap": stat_tests.bootstrap_test,
        "mood": stat_tests.mood_median_test,
    }
    if name not in registry:
        raise ValueError(f"Unknown test {test!r}; allowed: {sorted(registry)} or a callable")
    return registry[name]


def _p_value_of(result) -> float:
    if isinstance(result, stat_tests.StatTestResult):
        return result.p_value
    if isinstance(result, Mapping):
        return float(result["p_value"])
    return float(result)


def simulate_aa(
    values,
    test="welch",
    n_simulations: int = 1_000,
    alpha: float = 0.05,
    experiment_share: float = 0.5,
    effect_rel: float = 0.0,
    effect_abs: float = 0.0,
    random_state: int | None = 0,
) -> dict:
    """A/A (и A/B) симуляции критерия на реальном распределении метрики.

    Значения юнитов случайно делятся на две группы ``n_simulations`` раз. При
    нулевом эффекте доля ``p < alpha`` — эмпирическая ошибка I рода: она должна
    быть близка к ``alpha`` (``is_fpr_ok``), а p-value — равномерны
    (``ks_uniform_p_value``). С ``effect_rel`` / ``effect_abs`` в экспериментальную
    группу добавляется эффект, и та же доля — эмпирическая мощность.

    ``test`` — имя критерия или ``callable(control, experiment)``, возвращающий
    ``StatTestResult``, словарь с ``p_value`` или число.
    """
    x = _clean(values)
    if len(x) < 4:
        raise ValueError("simulate_aa requires at least four observations")
    if not 0 < experiment_share < 1:
        raise ValueError("experiment_share must be in (0, 1)")
    fn = _resolve_test_callable(test)
    rng = np.random.default_rng(random_state)
    n_exp = int(round(len(x) * experiment_share))
    n_exp = min(max(n_exp, 2), len(x) - 2)
    p_values = np.empty(n_simulations)
    for i in range(n_simulations):
        perm = rng.permutation(len(x))
        experiment = x[perm[:n_exp]] * (1 + effect_rel) + effect_abs
        control = x[perm[n_exp:]]
        p_values[i] = _p_value_of(fn(control, experiment))
    finite = p_values[np.isfinite(p_values)]
    rejections = int(np.sum(finite < alpha))
    rate = rejections / len(finite) if len(finite) else np.nan
    ci_low, ci_high = (
        proportion_confint(rejections, len(finite), alpha=0.05, method="wilson")
        if len(finite)
        else (np.nan, np.nan)
    )
    is_null = effect_rel == 0 and effect_abs == 0
    out = {
        "rejection_rate": rate,
        "ci_lower": float(ci_low),
        "ci_upper": float(ci_high),
        "alpha": alpha,
        "n_simulations": int(len(finite)),
        "p_values": p_values,
    }
    if is_null:
        out["is_fpr_ok"] = bool(ci_low <= alpha <= ci_high) if len(finite) else False
        out["ks_uniform_p_value"] = (
            float(stats.kstest(finite, "uniform").pvalue) if len(finite) else np.nan
        )
    return out
