"""Критерии для трёх и более групп: omnibus-тесты и post-hoc попарные сравнения.

Группы передаются словарём ``{имя_группы: значения}``. Порядок ключей задаёт
порядок групп; первая группа считается контролем там, где это важно.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.oneway import anova_oneway

from .multiple_testing import adjust_pvalues

OMNIBUS_TESTS = ("anova", "welch_anova", "kruskal", "chi2")


def _clean_groups(groups: Mapping[str, Sequence[float]]) -> dict[str, np.ndarray]:
    if len(groups) < 2:
        raise ValueError("At least two groups are required")
    out = {}
    for name, values in groups.items():
        array = np.asarray(values, dtype=float).ravel()
        out[name] = array[np.isfinite(array)]
    return out


def _omnibus_row(test, statistic=np.nan, p_value=np.nan, df=np.nan, effect=np.nan, effect_name=""):
    def _f(x):
        return float(x) if x is not None and np.isfinite(x) else np.nan

    return {
        "test": test,
        "statistic": _f(statistic),
        "p_value": _f(p_value),
        "df": _f(df),
        "effect_size": _f(effect),
        "effect_size_name": effect_name,
    }


# --------------------------------------------------------------------------- omnibus


def anova_test(groups: Mapping[str, Sequence[float]]) -> dict:
    """Однофакторный ANOVA (равные дисперсии). Эффект — эта-квадрат."""
    data = _clean_groups(groups)
    arrays = list(data.values())
    if any(len(a) < 2 for a in arrays):
        return _omnibus_row("anova", effect_name="eta_squared")
    stat, p_value = stats.f_oneway(*arrays)
    pooled = np.concatenate(arrays)
    ss_total = np.sum((pooled - pooled.mean()) ** 2)
    ss_between = sum(len(a) * (a.mean() - pooled.mean()) ** 2 for a in arrays)
    eta = ss_between / ss_total if ss_total > 0 else np.nan
    return _omnibus_row("anova", stat, p_value, len(arrays) - 1, eta, "eta_squared")


def welch_anova_test(groups: Mapping[str, Sequence[float]]) -> dict:
    """Welch ANOVA: сравнение средних без предположения о равенстве дисперсий."""
    data = _clean_groups(groups)
    arrays = list(data.values())
    if any(len(a) < 2 or a.var(ddof=1) == 0 for a in arrays):
        return _omnibus_row("welch_anova", effect_name="eta_squared")
    res = anova_oneway(arrays, use_var="unequal", welch_correction=True)
    pooled = np.concatenate(arrays)
    ss_total = np.sum((pooled - pooled.mean()) ** 2)
    ss_between = sum(len(a) * (a.mean() - pooled.mean()) ** 2 for a in arrays)
    eta = ss_between / ss_total if ss_total > 0 else np.nan
    return _omnibus_row("welch_anova", res.statistic, res.pvalue, res.df_num, eta, "eta_squared")


def kruskal_wallis_test(groups: Mapping[str, Sequence[float]]) -> dict:
    """Краскел—Уоллис: ранговый аналог ANOVA (с поправкой на связки). Эффект — эпсилон-квадрат."""
    data = _clean_groups(groups)
    arrays = list(data.values())
    n_total = sum(len(a) for a in arrays)
    if any(len(a) == 0 for a in arrays) or len(np.unique(np.concatenate(arrays))) < 2:
        return _omnibus_row("kruskal", effect_name="epsilon_squared")
    stat, p_value = stats.kruskal(*arrays)
    k = len(arrays)
    eps = (stat - k + 1) / (n_total - k) if n_total > k else np.nan
    return _omnibus_row("kruskal", stat, p_value, k - 1, max(eps, 0.0), "epsilon_squared")


def chi2_homogeneity_test(successes: Mapping[str, int], totals: Mapping[str, int]) -> dict:
    """Хи-квадрат однородности долей в k группах. Эффект — V Крамера."""
    names = list(totals)
    s = np.array([int(successes[n]) for n in names])
    t = np.array([int(totals[n]) for n in names])
    if len(names) < 2 or (t <= 0).any() or s.sum() == 0 or s.sum() == t.sum():
        return _omnibus_row("chi2", effect_name="cramers_v")
    table = np.vstack([s, t - s])
    stat, p_value, dof, _ = stats.chi2_contingency(table, correction=False)
    cramers_v = np.sqrt(stat / t.sum())
    return _omnibus_row("chi2", stat, p_value, dof, cramers_v, "cramers_v")


def levene_test(groups: Mapping[str, Sequence[float]], center: str = "median") -> dict:
    """Проверка равенства дисперсий: ``center='median'`` — Браун—Форсайт, ``'mean'`` — Левене."""
    arrays = list(_clean_groups(groups).values())
    name = "brown_forsythe" if center == "median" else "levene"
    if any(len(a) < 2 for a in arrays):
        return _omnibus_row(name)
    stat, p_value = stats.levene(*arrays, center=center)
    return _omnibus_row(name, stat, p_value, len(arrays) - 1)


def friedman_test(measurements: Mapping[str, Sequence[float]]) -> dict:
    """Критерий Фридмана для 3+ связанных измерений одних и тех же юнитов."""
    arrays = [np.asarray(v, dtype=float).ravel() for v in measurements.values()]
    if len(arrays) < 3:
        raise ValueError("Friedman test requires at least three related samples")
    if len({len(a) for a in arrays}) != 1:
        raise ValueError("Related samples must have the same length")
    mask = np.all(np.isfinite(np.vstack(arrays)), axis=0)
    arrays = [a[mask] for a in arrays]
    if mask.sum() < 2:
        return _omnibus_row("friedman")
    stat, p_value = stats.friedmanchisquare(*arrays)
    return _omnibus_row("friedman", stat, p_value, len(arrays) - 1)


def omnibus_test(groups: Mapping[str, Sequence[float]], test: str = "welch_anova") -> dict:
    """Общий тест «есть ли хоть одно различие между группами» для числовой метрики."""
    if test == "anova":
        return anova_test(groups)
    if test == "welch_anova":
        return welch_anova_test(groups)
    if test == "kruskal":
        return kruskal_wallis_test(groups)
    raise ValueError("test must be 'anova', 'welch_anova' or 'kruskal' (для долей — chi2_homogeneity_test)")


# --------------------------------------------------------------------------- post-hoc

_POSTHOC_COLUMNS = [
    "group_base",
    "group_exp",
    "estimate",
    "statistic",
    "p_value",
    "p_value_adj",
    "ci_lower",
    "ci_upper",
]


def _resolve_pairs(names: list[str], pairs) -> list[tuple[str, str]]:
    if pairs is None:
        return list(combinations(names, 2))
    out = []
    for base, exp in pairs:
        if base not in names or exp not in names:
            raise ValueError(f"Unknown group in pair ({base!r}, {exp!r})")
        out.append((base, exp))
    return out


def dunn_test(
    groups: Mapping[str, Sequence[float]],
    p_adjust: str | None = "holm",
    pairs: Sequence[tuple[str, str]] | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Тест Данна: непараметрические попарные сравнения по средним рангам.

    Ранги считаются по объединённой выборке всех групп, дисперсия — с поправкой
    на связки (как в Краскеле—Уоллисе). ``pairs`` ограничивает набор заранее
    заданных сравнений (например, только с контролем) — это уменьшает число
    проверок и повышает мощность. ``estimate`` — разница средних рангов
    ``exp - base``; тест говорит о ранговом положении распределений, а не о
    средних или медианах.
    """
    data = _clean_groups(groups)
    names = list(data)
    pair_list = _resolve_pairs(names, pairs)
    pooled = np.concatenate([data[n] for n in names])
    n_total = len(pooled)
    rows = []
    if n_total == 0:
        return pd.DataFrame(columns=_POSTHOC_COLUMNS)
    ranks = stats.rankdata(pooled)
    _, counts = np.unique(pooled, return_counts=True)
    tie_term = float(np.sum(counts**3 - counts)) / (12 * (n_total - 1)) if n_total > 1 else 0.0
    base_var = n_total * (n_total + 1) / 12 - tie_term
    mean_ranks, sizes, start = {}, {}, 0
    for name in names:
        n = len(data[name])
        sizes[name] = n
        mean_ranks[name] = ranks[start : start + n].mean() if n else np.nan
        start += n
    for base, exp in pair_list:
        diff = mean_ranks[exp] - mean_ranks[base]
        z, p_value = np.nan, np.nan
        if sizes[base] > 0 and sizes[exp] > 0 and base_var > 0:
            se = np.sqrt(base_var * (1 / sizes[base] + 1 / sizes[exp]))
            z = diff / se
            p_value = min(1.0, 2 * stats.norm.sf(abs(z)))
        rows.append(
            {
                "group_base": base,
                "group_exp": exp,
                "estimate": diff,
                "statistic": z,
                "p_value": p_value,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "mean_rank_base": mean_ranks[base],
                "mean_rank_exp": mean_ranks[exp],
            }
        )
    result = pd.DataFrame(rows)
    result["p_value_adj"], _ = adjust_pvalues(result["p_value"].to_numpy(), p_adjust, alpha)
    return result[_POSTHOC_COLUMNS + ["mean_rank_base", "mean_rank_exp"]]


def tukey_hsd_test(groups: Mapping[str, Sequence[float]], alpha: float = 0.05) -> pd.DataFrame:
    """Tukey HSD: все попарные сравнения средних (равные дисперсии).

    ``p_value`` уже скорректирован на все пары (совпадает с ``p_value_adj``),
    CI — одновременные.
    """
    data = _clean_groups(groups)
    names = list(data)
    if any(len(a) < 2 for a in data.values()):
        raise ValueError("Tukey HSD requires at least two observations per group")
    res = stats.tukey_hsd(*data.values())
    ci = res.confidence_interval(confidence_level=1 - alpha)
    rows = []
    for i, j in combinations(range(len(names)), 2):
        rows.append(
            {
                "group_base": names[i],
                "group_exp": names[j],
                "estimate": float(res.statistic[j, i]),
                "statistic": float(res.statistic[j, i]),
                "p_value": float(res.pvalue[j, i]),
                "p_value_adj": float(res.pvalue[j, i]),
                "ci_lower": float(ci.low[j, i]),
                "ci_upper": float(ci.high[j, i]),
            }
        )
    return pd.DataFrame(rows)[_POSTHOC_COLUMNS]


def games_howell_test(groups: Mapping[str, Sequence[float]], alpha: float = 0.05) -> pd.DataFrame:
    """Games—Howell: все попарные сравнения средних при неравных дисперсиях и размерах.

    Аналог Tukey HSD для ситуации Welch; ``p_value`` уже скорректирован на все
    пары, CI — одновременные.
    """
    data = _clean_groups(groups)
    names = list(data)
    k = len(names)
    if any(len(a) < 2 for a in data.values()):
        raise ValueError("Games-Howell requires at least two observations per group")
    rows = []
    for base, exp in combinations(names, 2):
        c, e = data[base], data[exp]
        v_c, v_e = c.var(ddof=1) / len(c), e.var(ddof=1) / len(e)
        diff = float(e.mean() - c.mean())
        se = np.sqrt(v_c + v_e)
        row = {"group_base": base, "group_exp": exp, "estimate": diff}
        if se == 0:
            row.update(statistic=np.nan, p_value=np.nan, ci_lower=diff, ci_upper=diff)
        else:
            df = (v_c + v_e) ** 2 / (v_c**2 / (len(c) - 1) + v_e**2 / (len(e) - 1))
            q = abs(diff) / (se / np.sqrt(2))
            p_value = float(stats.studentized_range.sf(q, k, df))
            margin = float(stats.studentized_range.ppf(1 - alpha, k, df)) * se / np.sqrt(2)
            row.update(
                statistic=float(q), p_value=p_value, ci_lower=diff - margin, ci_upper=diff + margin
            )
        row["p_value_adj"] = row["p_value"]
        rows.append(row)
    return pd.DataFrame(rows)[_POSTHOC_COLUMNS]


def dunnett_test(
    groups: Mapping[str, Sequence[float]], control: str | None = None, alpha: float = 0.05
) -> pd.DataFrame:
    """Тест Даннетта: каждая тестовая группа против одного контроля.

    Мощнее Tukey/Holm, когда интересны только сравнения с контролем. ``control``
    по умолчанию — первая группа. ``p_value`` уже скорректирован.
    """
    if not hasattr(stats, "dunnett"):
        raise RuntimeError("dunnett_test requires scipy>=1.11")
    data = _clean_groups(groups)
    names = list(data)
    control = names[0] if control is None else control
    if control not in data:
        raise ValueError(f"Unknown control group {control!r}")
    treatments = [n for n in names if n != control]
    if any(len(a) < 2 for a in data.values()):
        raise ValueError("Dunnett test requires at least two observations per group")
    res = stats.dunnett(*[data[n] for n in treatments], control=data[control])
    ci = res.confidence_interval(confidence_level=1 - alpha)
    rows = []
    for idx, name in enumerate(treatments):
        rows.append(
            {
                "group_base": control,
                "group_exp": name,
                "estimate": float(data[name].mean() - data[control].mean()),
                "statistic": float(res.statistic[idx]),
                "p_value": float(res.pvalue[idx]),
                "p_value_adj": float(res.pvalue[idx]),
                "ci_lower": float(ci.low[idx]),
                "ci_upper": float(ci.high[idx]),
            }
        )
    return pd.DataFrame(rows)[_POSTHOC_COLUMNS]


def posthoc_test(
    groups: Mapping[str, Sequence[float]],
    test: str = "dunn",
    alpha: float = 0.05,
    p_adjust: str | None = "holm",
    pairs: Sequence[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Диспетчер post-hoc процедур: ``dunn``, ``tukey``, ``games_howell``, ``dunnett``."""
    if test == "dunn":
        return dunn_test(groups, p_adjust=p_adjust, pairs=pairs, alpha=alpha)
    if test == "tukey":
        return tukey_hsd_test(groups, alpha=alpha)
    if test == "games_howell":
        return games_howell_test(groups, alpha=alpha)
    if test == "dunnett":
        return dunnett_test(groups, alpha=alpha)
    raise ValueError("test must be 'dunn', 'tukey', 'games_howell' or 'dunnett'")
