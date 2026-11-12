"""Байесовская оценка A/B: вероятность выигрыша и ожидаемые потери.

Дополняет частотный анализ ответом на вопрос «с какой вероятностью B лучше A и
сколько мы теряем, если ошибёмся». Не отменяет дизайн эксперимента и не решает
проблему подглядывания.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def bayes_proportions(
    control_success: int,
    control_n: int,
    experiment_success: int,
    experiment_n: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    credible_level: float = 0.95,
    n_draws: int = 200_000,
    random_state: int | None = 0,
) -> dict:
    """Beta-Binomial модель для двух долей (сопряжённый prior ``Beta(a, b)``).

    Возвращает вероятность, что доля в эксперименте выше, ожидаемые потери при
    выборе каждой группы (в абсолютных пунктах доли) и credible interval для
    разницы и относительного изменения.
    """
    if not (0 <= control_success <= control_n and 0 <= experiment_success <= experiment_n):
        raise ValueError("successes must be within [0, n]")
    if control_n <= 0 or experiment_n <= 0:
        raise ValueError("group sizes must be positive")
    rng = np.random.default_rng(random_state)
    p_c = rng.beta(prior_alpha + control_success, prior_beta + control_n - control_success, n_draws)
    p_e = rng.beta(
        prior_alpha + experiment_success, prior_beta + experiment_n - experiment_success, n_draws
    )
    diff = p_e - p_c
    tail = (1 - credible_level) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = diff / p_c
    lift = lift[np.isfinite(lift)]
    return {
        "prob_experiment_better": float(np.mean(diff > 0)),
        "expected_loss_experiment": float(np.mean(np.maximum(-diff, 0))),
        "expected_loss_control": float(np.mean(np.maximum(diff, 0))),
        "diff_mean": float(diff.mean()),
        "diff_ci_lower": float(np.quantile(diff, tail)),
        "diff_ci_upper": float(np.quantile(diff, 1 - tail)),
        "rel_lift_ci_lower": float(np.quantile(lift, tail)) if len(lift) else np.nan,
        "rel_lift_ci_upper": float(np.quantile(lift, 1 - tail)) if len(lift) else np.nan,
    }


def bayes_means(control, experiment, credible_level: float = 0.95) -> dict:
    """Нормальная аппроксимация постериора разницы средних (плоский prior).

    Применима на больших выборках (ЦПТ). ``expected_loss_*`` — в единицах метрики.
    """
    c = np.asarray(control, dtype=float).ravel()
    e = np.asarray(experiment, dtype=float).ravel()
    c, e = c[np.isfinite(c)], e[np.isfinite(e)]
    nan = {
        "prob_experiment_better": np.nan,
        "expected_loss_experiment": np.nan,
        "expected_loss_control": np.nan,
        "diff_mean": np.nan,
        "diff_ci_lower": np.nan,
        "diff_ci_upper": np.nan,
    }
    if len(c) < 2 or len(e) < 2:
        return nan
    m = float(e.mean() - c.mean())
    s = float(np.sqrt(c.var(ddof=1) / len(c) + e.var(ddof=1) / len(e)))
    if s == 0:
        return {**nan, "diff_mean": m, "diff_ci_lower": m, "diff_ci_upper": m}
    z = m / s
    q = float(stats.norm.ppf(1 - (1 - credible_level) / 2))
    # E[max(-d, 0)] и E[max(d, 0)] для d ~ N(m, s^2)
    loss_exp = s * stats.norm.pdf(z) - m * stats.norm.cdf(-z)
    loss_ctrl = s * stats.norm.pdf(z) + m * stats.norm.cdf(z)
    return {
        "prob_experiment_better": float(stats.norm.cdf(z)),
        "expected_loss_experiment": float(loss_exp),
        "expected_loss_control": float(loss_ctrl),
        "diff_mean": m,
        "diff_ci_lower": m - q * s,
        "diff_ci_upper": m + q * s,
    }
