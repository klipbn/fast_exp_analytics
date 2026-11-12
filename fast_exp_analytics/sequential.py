"""Последовательный анализ: как подглядывать в результаты, не раздувая ошибку I рода.

Два подхода:

* **Group sequential** (Lan—DeMets alpha-spending): заранее известно число
  промежуточных просмотров; на каждом своя граница z-статистики.
* **mSPRT / always-valid inference**: смотреть можно когда угодно; p-value и
  доверительные интервалы остаются валидными при любом правиле остановки.

Обычный p-value при ежедневном подглядывании даёт ошибку I рода 20–30% вместо 5%.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import optimize, stats

SPENDING_FUNCTIONS = ("obf", "pocock", "linear")
_Z_CAP = 8.0


def alpha_spending(information_fraction, alpha: float = 0.05, kind: str = "obf"):
    """Кумулятивно потраченная альфа при доле информации ``t`` (двусторонний тест).

    ``obf`` — О'Брайен—Флеминг (почти не тратит альфу в начале, финальная
    граница близка к обычной; рекомендуется), ``pocock`` — равномерные границы,
    ``linear`` — ``alpha * t``.
    """
    t = np.clip(np.asarray(information_fraction, dtype=float), 0.0, 1.0)
    if kind == "obf":
        with np.errstate(divide="ignore"):
            # по alpha/2 на каждую сторону — стандартная симметричная конвенция (ldbounds, gsDesign)
            spent = 2 * (2 - 2 * stats.norm.cdf(stats.norm.ppf(1 - alpha / 4) / np.sqrt(t)))
    elif kind == "pocock":
        spent = alpha * np.log(1 + (np.e - 1) * t)
    elif kind == "linear":
        spent = alpha * t
    else:
        raise ValueError(f"kind must be one of {SPENDING_FUNCTIONS}")
    return spent


def _exit_probability(c: float, t: float, grid, weights, density, step_sd: float) -> float:
    b = c * np.sqrt(t)
    tail = stats.norm.cdf((-b - grid) / step_sd) + stats.norm.sf((b - grid) / step_sd)
    return float(np.sum(weights * density * tail))


def group_sequential_boundaries(
    information_fractions: Sequence[float],
    alpha: float = 0.05,
    spending: str = "obf",
    n_grid: int = 1_001,
) -> pd.DataFrame:
    """Границы group sequential теста (Lan—DeMets) для двустороннего z-теста.

    ``information_fractions`` — возрастающие доли накопленной информации на
    каждом просмотре, последняя равна 1 (например, доля набранной выборки:
    ``[0.25, 0.5, 0.75, 1]``). На просмотре ``k`` нулевая гипотеза отвергается,
    если ``|z| >= z_boundary``; эквивалентно ``p_value <= nominal_p_value``.

    Границы считаются рекурсивным численным интегрированием плотности
    накопленной статистики (Armitage—McPherson—Rowe).
    """
    t = np.asarray(information_fractions, dtype=float)
    if t.ndim != 1 or len(t) == 0:
        raise ValueError("information_fractions must be a non-empty 1D sequence")
    if (t <= 0).any() or (t > 1 + 1e-12).any() or (np.diff(t) <= 0).any():
        raise ValueError("information_fractions must be increasing values in (0, 1]")
    spent = alpha_spending(t, alpha, spending)
    increments = np.diff(np.concatenate([[0.0], spent]))

    boundaries: list[float] = []
    grid = weights = density = None
    for k, t_k in enumerate(t):
        inc = float(increments[k])
        if k == 0:
            c = float(stats.norm.isf(inc / 2)) if inc > 0 else np.inf
        else:
            step_sd = float(np.sqrt(t_k - t[k - 1]))
            args = (t_k, grid, weights, density, step_sd)
            if inc <= 0 or _exit_probability(_Z_CAP * 2, *args) >= inc:
                c = np.inf
            elif _exit_probability(0.0, *args) <= inc:
                c = 0.0
            else:
                c = float(
                    optimize.brentq(
                        lambda v, args=args, inc=inc: _exit_probability(v, *args) - inc, 0.0, _Z_CAP * 2, xtol=1e-10
                    )
                )
        boundaries.append(c)
        if k == len(t) - 1:
            break
        # плотность накопленной статистики S_k = Z_k * sqrt(t_k) в области продолжения
        b = min(c, _Z_CAP) * np.sqrt(t_k)
        new_grid = np.linspace(-b, b, n_grid)
        new_weights = np.full(n_grid, new_grid[1] - new_grid[0])
        new_weights[[0, -1]] *= 0.5
        if k == 0:
            new_density = stats.norm.pdf(new_grid, scale=np.sqrt(t_k))
        else:
            kernel = stats.norm.pdf(new_grid[:, None] - grid[None, :], scale=step_sd)
            new_density = kernel @ (weights * density)
        grid, weights, density = new_grid, new_weights, new_density

    z = np.array(boundaries)
    return pd.DataFrame(
        {
            "look": np.arange(1, len(t) + 1),
            "information_fraction": t,
            "alpha_spent": spent,
            "z_boundary": z,
            "nominal_p_value": 2 * stats.norm.sf(z),
        }
    )


def group_sequential_test(
    z_statistics: Sequence[float],
    information_fractions: Sequence[float],
    alpha: float = 0.05,
    spending: str = "obf",
) -> pd.DataFrame:
    """Применить group sequential границы к наблюдённым z-статистикам.

    ``z_statistics`` — значения на уже проведённых просмотрах (может быть короче
    плана ``information_fractions``). Колонка ``stop`` — первый просмотр, на
    котором можно остановить тест с отвержением H0.
    """
    plan = group_sequential_boundaries(information_fractions, alpha, spending)
    z = np.asarray(z_statistics, dtype=float)
    if len(z) > len(plan):
        raise ValueError("more z-statistics than planned looks")
    result = plan.iloc[: len(z)].copy()
    result["z_statistic"] = z
    result["reject"] = np.abs(z) >= result["z_boundary"].to_numpy()
    result["stop"] = result["reject"] & (result["reject"].cumsum() == 1)
    return result


# --------------------------------------------------------------------------- mSPRT


def always_valid_p_value(delta, se, tau: float):
    """Always-valid p-value mSPRT с нормальной смесью ``N(0, tau^2)`` по эффекту.

    ``delta`` — оценка эффекта, ``se`` — её стандартная ошибка на текущий
    момент, ``tau`` — ожидаемый масштаб эффекта в тех же единицах (разумный
    выбор — абсолютный MDE). ``tau`` фиксируется до начала теста.

    Это p-value одного момента времени; для последовательности просмотров
    берётся накопленный минимум — см. ``msprt_sequence``.
    """
    delta = np.asarray(delta, dtype=float)
    v = np.asarray(se, dtype=float) ** 2
    if tau <= 0:
        raise ValueError("tau must be positive")
    t2 = tau**2
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_lr = 0.5 * np.log(v / (v + t2)) + t2 * delta**2 / (2 * v * (v + t2))
        p = np.minimum(1.0, np.exp(-log_lr))
    return p if p.ndim else float(p)


def always_valid_ci(delta, se, tau: float, alpha: float = 0.05):
    """Always-valid доверительный интервал (confidence sequence) для эффекта."""
    delta = np.asarray(delta, dtype=float)
    v = np.asarray(se, dtype=float) ** 2
    if tau <= 0:
        raise ValueError("tau must be positive")
    t2 = tau**2
    with np.errstate(divide="ignore", invalid="ignore"):
        half = np.sqrt((2 * v * (v + t2) / t2) * (np.log(1 / alpha) + 0.5 * np.log((v + t2) / v)))
    lower, upper = delta - half, delta + half
    if lower.ndim == 0:
        return float(lower), float(upper)
    return lower, upper


def msprt_sequence(
    deltas: Sequence[float], ses: Sequence[float], tau: float, alpha: float = 0.05
) -> pd.DataFrame:
    """mSPRT по последовательности просмотров (например, по дням эксперимента).

    ``deltas`` и ``ses`` — накопленные оценки эффекта и их стандартные ошибки на
    каждом просмотре. p-value — накопленный минимум, CI — накопленное
    пересечение; ``stop`` — первый просмотр со значимым результатом.
    Ошибка I рода не превышает ``alpha`` при любом числе просмотров.
    """
    deltas = np.asarray(deltas, dtype=float)
    ses = np.asarray(ses, dtype=float)
    if deltas.shape != ses.shape or deltas.ndim != 1:
        raise ValueError("deltas and ses must be 1D sequences of the same length")
    p_point = np.asarray(always_valid_p_value(deltas, ses, tau), dtype=float).reshape(-1)
    lower, upper = always_valid_ci(deltas, ses, tau, alpha)
    p_point = np.where(np.isfinite(p_point), p_point, 1.0)
    lower = np.where(np.isfinite(lower), lower, -np.inf)
    upper = np.where(np.isfinite(upper), upper, np.inf)
    p_running = np.minimum.accumulate(p_point)
    reject = p_running < alpha
    return pd.DataFrame(
        {
            "look": np.arange(1, len(deltas) + 1),
            "delta": deltas,
            "se": ses,
            "p_value_always_valid": p_running,
            "ci_lower": np.maximum.accumulate(lower),
            "ci_upper": np.minimum.accumulate(upper),
            "reject": reject,
            "stop": reject & (np.cumsum(reject) == 1),
        }
    )
