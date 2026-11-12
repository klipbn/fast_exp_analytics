"""Размер выборки, MDE и мощность для средних, долей и ratio-метрик.

Общие параметры:

* ``ratio`` — отношение размеров групп ``n_experiment / n_control``;
* ``alternative`` — ``"two-sided"`` или ``"one-sided"``;
* ``n_comparisons`` — число сравнений для поправки Бонферрони (например, число
  тестовых групп в ABC/ABCD или число ключевых метрик).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _z_alpha(alpha: float, alternative: str, n_comparisons: int) -> float:
    if alternative not in ("two-sided", "one-sided"):
        raise ValueError("alternative must be 'two-sided' or 'one-sided'")
    if n_comparisons < 1:
        raise ValueError("n_comparisons must be >= 1")
    alpha = alpha / n_comparisons
    return float(stats.norm.ppf(1 - alpha / 2 if alternative == "two-sided" else 1 - alpha))


def _power_from_z(z_effect: float, z_alpha: float, alternative: str) -> float:
    power = stats.norm.sf(z_alpha - z_effect)
    if alternative == "two-sided":
        power += stats.norm.cdf(-z_alpha - z_effect)
    return float(power)


def sample_size_means(
    std: float,
    mde_abs: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    std_experiment: float | None = None,
    alternative: str = "two-sided",
    n_comparisons: int = 1,
) -> dict:
    """Размер групп для обнаружения абсолютной разницы средних ``mde_abs``."""
    if mde_abs == 0 or std < 0 or ratio <= 0:
        raise ValueError("mde_abs must be non-zero, std non-negative and ratio positive")
    std_e = std if std_experiment is None else std_experiment
    z_a, z_b = _z_alpha(alpha, alternative, n_comparisons), float(stats.norm.ppf(power))
    n_c = (z_a + z_b) ** 2 * (std**2 + std_e**2 / ratio) / mde_abs**2
    n_c = int(np.ceil(n_c))
    n_e = int(np.ceil(n_c * ratio))
    return {"n_control": n_c, "n_experiment": n_e, "n_total": n_c + n_e}


def sample_size_proportions(
    p_base: float,
    mde_abs: float | None = None,
    mde_rel: float | None = None,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    alternative: str = "two-sided",
    n_comparisons: int = 1,
) -> dict:
    """Размер групп для долей; эффект задаётся ``mde_abs`` (п.п. в долях) или ``mde_rel``."""
    if (mde_abs is None) == (mde_rel is None):
        raise ValueError("provide exactly one of mde_abs or mde_rel")
    delta = mde_abs if mde_abs is not None else p_base * mde_rel
    p2 = p_base + delta
    if not (0 < p_base < 1 and 0 < p2 < 1) or delta == 0 or ratio <= 0:
        raise ValueError("p_base and p_base + effect must be in (0, 1), effect non-zero")
    z_a, z_b = _z_alpha(alpha, alternative, n_comparisons), float(stats.norm.ppf(power))
    p_bar = (p_base + ratio * p2) / (1 + ratio)
    num = (
        z_a * np.sqrt(p_bar * (1 - p_bar) * (1 + 1 / ratio))
        + z_b * np.sqrt(p_base * (1 - p_base) + p2 * (1 - p2) / ratio)
    ) ** 2
    n_c = int(np.ceil(num / delta**2))
    n_e = int(np.ceil(n_c * ratio))
    return {"n_control": n_c, "n_experiment": n_e, "n_total": n_c + n_e}


def mde_means(
    std: float,
    n_control: int,
    n_experiment: int,
    alpha: float = 0.05,
    power: float = 0.8,
    std_experiment: float | None = None,
    alternative: str = "two-sided",
    n_comparisons: int = 1,
) -> float:
    """Минимальный обнаружимый абсолютный эффект для разницы средних."""
    if n_control <= 0 or n_experiment <= 0:
        return np.nan
    std_e = std if std_experiment is None else std_experiment
    z_a, z_b = _z_alpha(alpha, alternative, n_comparisons), float(stats.norm.ppf(power))
    return float((z_a + z_b) * np.sqrt(std**2 / n_control + std_e**2 / n_experiment))


def mde_proportions(
    p_base: float,
    n_control: int,
    n_experiment: int,
    alpha: float = 0.05,
    power: float = 0.8,
    alternative: str = "two-sided",
    n_comparisons: int = 1,
) -> float:
    """Минимальный обнаружимый абсолютный эффект для доли (дисперсия по базовой доле)."""
    if n_control <= 0 or n_experiment <= 0 or not 0 < p_base < 1:
        return np.nan
    z_a, z_b = _z_alpha(alpha, alternative, n_comparisons), float(stats.norm.ppf(power))
    return float((z_a + z_b) * np.sqrt(p_base * (1 - p_base) * (1 / n_control + 1 / n_experiment)))


def power_means(
    delta: float,
    std: float,
    n_control: int,
    n_experiment: int,
    alpha: float = 0.05,
    std_experiment: float | None = None,
    alternative: str = "two-sided",
    n_comparisons: int = 1,
) -> float:
    """Мощность теста разницы средних для заданного (планового, не наблюдённого) эффекта."""
    if n_control <= 0 or n_experiment <= 0:
        return np.nan
    std_e = std if std_experiment is None else std_experiment
    se = np.sqrt(std**2 / n_control + std_e**2 / n_experiment)
    if se == 0:
        return np.nan
    return _power_from_z(abs(delta) / se, _z_alpha(alpha, alternative, n_comparisons), alternative)


def power_proportions(
    p_control: float,
    p_experiment: float,
    n_control: int,
    n_experiment: int,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    n_comparisons: int = 1,
) -> float:
    """Мощность z-теста долей для заданных долей в группах."""
    if n_control <= 0 or n_experiment <= 0:
        return np.nan
    se = np.sqrt(
        p_control * (1 - p_control) / n_control + p_experiment * (1 - p_experiment) / n_experiment
    )
    if se == 0:
        return np.nan
    z_eff = abs(p_experiment - p_control) / se
    return _power_from_z(z_eff, _z_alpha(alpha, alternative, n_comparisons), alternative)


def ratio_metric_std(num, den) -> tuple[float, float]:
    """``(ratio, std)`` ratio-метрики на юнит: std линеаризованной величины / средний знаменатель.

    Результат подставляется в ``sample_size_means`` / ``mde_means`` как ``std``,
    а эффект задаётся в единицах самой ratio-метрики.
    """
    num = np.asarray(num, dtype=float).ravel()
    den = np.asarray(den, dtype=float).ravel()
    mask = np.isfinite(num) & np.isfinite(den)
    num, den = num[mask], den[mask]
    if len(num) < 2 or den.mean() == 0:
        return np.nan, np.nan
    ratio = num.sum() / den.sum()
    return float(ratio), float((num - ratio * den).std(ddof=1) / den.mean())


def cuped_sample_size_factor(correlation: float) -> float:
    """Во сколько раз CUPED уменьшает нужную выборку: ``1 - rho^2``."""
    if not -1 <= correlation <= 1:
        raise ValueError("correlation must be in [-1, 1]")
    return float(1 - correlation**2)


def optimal_control_share(n_treatments: int) -> float:
    """Оптимальная доля контроля при сравнении ``k`` тестовых групп с одним контролем.

    Минимизирует дисперсию сравнений с контролем: ``n_control / n_treatment = sqrt(k)``.
    """
    if n_treatments < 1:
        raise ValueError("n_treatments must be >= 1")
    root = np.sqrt(n_treatments)
    return float(root / (root + n_treatments))
