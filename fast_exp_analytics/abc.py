from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW

from .config import validate_metrics_config

ALPHA = 0.05
POWER = 0.80

AGGREGATE_COLUMN_NAMES_ABC = [
    "metric_name",
    "metric_type",
    "pair",
    "group_base",
    "group_exp",
    "value_base",
    "value_exp",
    "abs_delta",
    "rel_delta",
    "p_value",
    "p_value_adj",
    "result",
    "result_adj",
    "mde",
    "number_samples",
    "number_samples_base",
    "number_samples_exp",
    "avg_value_base",
    "avg_value_exp",
    "avg_abs_delta",
    "avg_rel_delta",
    "days_more_if_same_delta",
    "days_more_base",
    "days_more_exp",
    "n_required_per_group",
    "power_now",
    "direction",
]

def _is_finite(x) -> bool:
    try:
        return x is not None and np.isfinite(float(x))
    except Exception:
        return False

def _safe_div(a, b):
    if not _is_finite(a) or not _is_finite(b) or float(b) == 0:
        return np.nan
    return float(a) / float(b)

def _z_alpha_beta(alpha=ALPHA, power=POWER):
    return stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)

def _days_elapsed(exp_start_date, exp_end_date):
    start = pd.to_datetime(exp_start_date).normalize()
    end = pd.to_datetime(exp_end_date).normalize()
    return int(max((end - start).days + 1, 1))

def _harmonic_mean_n(n1, n2):
    if n1 <= 0 or n2 <= 0:
        return np.nan
    return float(2 / (1 / n1 + 1 / n2))

def _days_to_reach_required_n(n_current, n_required, days_running):
    if not _is_finite(n_required) or n_required <= 0:
        return np.nan
    if days_running is None or days_running <= 0:
        return np.nan
    if n_current >= n_required:
        return 0
    rate_per_day = n_current / days_running
    if rate_per_day <= 0:
        return np.nan
    return int(np.ceil((n_required - n_current) / rate_per_day))

def _prep_metric_pair_frames(df_metric, group_ctrl="A", group_tst="B", metric_type="additive"):
    df_metric = df_metric.copy()
    for col in ["num", "den"]:
        df_metric[col] = pd.to_numeric(df_metric[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    df_ctrl = df_metric.loc[df_metric["exp_group"] == group_ctrl, ["user_id", "num", "den"]].copy()
    df_tst = df_metric.loc[df_metric["exp_group"] == group_tst, ["user_id", "num", "den"]].copy()

    if metric_type == "additive":
        df_ctrl["value"] = df_ctrl["num"]
        df_tst["value"] = df_tst["num"]
    elif metric_type in ["average", "median"]:
        df_ctrl["value"] = np.where(df_ctrl["den"] > 0, df_ctrl["num"], np.nan)
        df_tst["value"] = np.where(df_tst["den"] > 0, df_tst["num"], np.nan)
    elif metric_type == "share":
        df_ctrl["value"] = np.where(df_ctrl["den"] > 0, (df_ctrl["num"] > 0).astype(float), np.nan)
        df_tst["value"] = np.where(df_tst["den"] > 0, (df_tst["num"] > 0).astype(float), np.nan)
    elif metric_type == "ratio":
        den_sum_ctrl = df_ctrl["den"].sum()
        linearization_coeff = df_ctrl["num"].sum() / den_sum_ctrl if _is_finite(den_sum_ctrl) and float(den_sum_ctrl) != 0 else np.nan
        df_ctrl["value"] = df_ctrl["num"] - linearization_coeff * df_ctrl["den"]
        df_tst["value"] = df_tst["num"] - linearization_coeff * df_tst["den"]
    else:
        raise ValueError(f"Unsupported metric type: {metric_type}")
    return df_ctrl, df_tst

def _share_success_obs(df_):
    obs = int((df_["den"] > 0).sum())
    if obs == 0:
        return 0, 0, np.nan
    success = int(((df_["den"] > 0) & (df_["num"] > 0)).sum())
    return success, obs, float(success / obs)

def calculate_number_samples(df_control_metrics, df_pilot_metrics):
    n_c = int(df_control_metrics["value"].notna().sum())
    n_t = int(df_pilot_metrics["value"].notna().sum())
    return n_c + n_t, n_c, n_t

def _mde_mean_two_sample(std_c, std_t, n_per_group, alpha=ALPHA, power=POWER):
    z_alpha, z_beta = _z_alpha_beta(alpha, power)
    return float(np.sqrt(((z_alpha + z_beta) ** 2) * (float(std_c) ** 2 + float(std_t) ** 2) / n_per_group))

def _mde_share_two_sample(p, n_per_group, alpha=ALPHA, power=POWER):
    z_alpha, z_beta = _z_alpha_beta(alpha, power)
    return float((z_alpha + z_beta) * np.sqrt(2 * p * (1 - p) / n_per_group))

def calculate_mde(df_control_metrics, df_pilot_metrics, metric_type, alpha=ALPHA, power=POWER):
    _, n_c, n_t = calculate_number_samples(df_control_metrics, df_pilot_metrics)
    n_per_group = _harmonic_mean_n(n_c, n_t)
    if not _is_finite(n_per_group) or n_per_group <= 0:
        return np.nan, np.nan
    if metric_type == "median":
        return np.nan, np.nan
    if metric_type == "share":
        _, _, p1 = _share_success_obs(df_control_metrics)
        _, _, p2 = _share_success_obs(df_pilot_metrics)
        if not _is_finite(p1) and not _is_finite(p2):
            return np.nan, np.nan
        p_pool = 0.5 * (p1 + p2) if _is_finite(p1) and _is_finite(p2) else (p1 if _is_finite(p1) else p2)
        return _mde_share_two_sample(p_pool, n_per_group, alpha, power), np.nan

    std_c = df_control_metrics["value"].std(ddof=1)
    std_t = df_pilot_metrics["value"].std(ddof=1)
    if not _is_finite(std_c) or not _is_finite(std_t):
        return np.nan, np.nan

    if metric_type == "ratio":
        den_mean = df_control_metrics["den"].mean()
        if not _is_finite(den_mean) or float(den_mean) == 0:
            return np.nan, np.nan
        mde_lin = _mde_mean_two_sample(std_c, std_t, n_per_group, alpha, power)
        return float(mde_lin / float(den_mean)), np.nan

    mde = _mde_mean_two_sample(std_c, std_t, n_per_group, alpha, power)
    effect = (df_control_metrics["value"].mean() / std_c) if (std_c != 0 and _is_finite(std_c)) else np.nan
    return float(mde), float(effect) if _is_finite(effect) else np.nan

def _required_n_per_group_for_observed_delta(df_control_metrics, df_pilot_metrics, metric_type, observed_delta, alpha=ALPHA, power=POWER):
    z_alpha, z_beta = _z_alpha_beta(alpha, power)
    if not _is_finite(observed_delta) or float(observed_delta) == 0:
        return np.inf
    if metric_type == "median":
        return np.inf
    if metric_type == "share":
        _, _, p1 = _share_success_obs(df_control_metrics)
        _, _, p2 = _share_success_obs(df_pilot_metrics)
        if not _is_finite(p1) or not _is_finite(p2):
            return np.inf
        p_bar = 0.5 * (p1 + p2)
        delta = abs(float(observed_delta))
        num = (z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        return float(np.ceil(num / (delta ** 2)))
    std_c = df_control_metrics["value"].std(ddof=1)
    std_t = df_pilot_metrics["value"].std(ddof=1)
    if not _is_finite(std_c) or not _is_finite(std_t):
        return np.inf
    disp_sum = float(std_c) ** 2 + float(std_t) ** 2
    if metric_type == "ratio":
        den_mean = df_control_metrics["den"].mean()
        if not _is_finite(den_mean) or float(den_mean) == 0:
            return np.inf
        delta_lin = float(observed_delta) * float(den_mean)
        return float(np.ceil(((z_alpha + z_beta) ** 2) * disp_sum / (delta_lin ** 2)))
    return float(np.ceil(((z_alpha + z_beta) ** 2) * disp_sum / (float(observed_delta) ** 2)))

def _power_for_delta(df_control_metrics, df_pilot_metrics, metric_type, observed_delta, alpha=ALPHA):
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    _, n_c, n_t = calculate_number_samples(df_control_metrics, df_pilot_metrics)
    if n_c <= 0 or n_t <= 0:
        return np.nan
    if not _is_finite(observed_delta) or float(observed_delta) == 0:
        return 0.0
    if metric_type == "median":
        return np.nan
    if metric_type == "share":
        _, _, p1 = _share_success_obs(df_control_metrics)
        _, _, p2 = _share_success_obs(df_pilot_metrics)
        if not _is_finite(p1) or not _is_finite(p2):
            return np.nan
        se = np.sqrt(p1 * (1 - p1) / n_c + p2 * (1 - p2) / n_t)
        if se == 0 or not np.isfinite(se):
            return np.nan
        z_eff = abs(float(observed_delta)) / se
        return float(1 - stats.norm.cdf(z_alpha - z_eff) + stats.norm.cdf(-z_alpha - z_eff))
    std_c = df_control_metrics["value"].std(ddof=1)
    std_t = df_pilot_metrics["value"].std(ddof=1)
    if not _is_finite(std_c) or not _is_finite(std_t):
        return np.nan
    se = np.sqrt((float(std_c) ** 2) / n_c + (float(std_t) ** 2) / n_t)
    if se == 0 or not np.isfinite(se):
        return np.nan
    if metric_type == "ratio":
        den_mean = df_control_metrics["den"].mean()
        if not _is_finite(den_mean) or float(den_mean) == 0:
            return np.nan
        z_eff = abs(float(observed_delta) * float(den_mean)) / se
    else:
        z_eff = abs(float(observed_delta)) / se
    return float(1 - stats.norm.cdf(z_alpha - z_eff) + stats.norm.cdf(-z_alpha - z_eff))

def calculate_base_exp_values(df_control_metrics, df_pilot_metrics, metric_type):
    if metric_type == "additive":
        base_value = df_control_metrics["value"].sum()
        exp_value = df_pilot_metrics["value"].sum()
    elif metric_type == "ratio":
        base_value = _safe_div(df_control_metrics["num"].sum(), df_control_metrics["den"].sum())
        exp_value = _safe_div(df_pilot_metrics["num"].sum(), df_pilot_metrics["den"].sum())
    elif metric_type == "median":
        base_value = df_control_metrics["value"].median()
        exp_value = df_pilot_metrics["value"].median()
    elif metric_type == "average":
        base_value = df_control_metrics["value"].mean()
        exp_value = df_pilot_metrics["value"].mean()
    elif metric_type == "share":
        _, _, base_value = _share_success_obs(df_control_metrics)
        _, _, exp_value = _share_success_obs(df_pilot_metrics)
    else:
        raise ValueError(f"Unsupported metric type: {metric_type}")
    abs_delta_value = exp_value - base_value if _is_finite(base_value) and _is_finite(exp_value) else np.nan
    rel_delta_value = _safe_div(abs_delta_value, base_value)
    return base_value, exp_value, abs_delta_value, rel_delta_value

def calculate_avg_base_exp_values(df_control_metrics, df_pilot_metrics, metric_type):
    if metric_type in ["additive", "average"]:
        avg_base = df_control_metrics["value"].mean()
        avg_exp = df_pilot_metrics["value"].mean()
        avg_abs = avg_exp - avg_base
        avg_rel = _safe_div(avg_abs, avg_base) * 100 if _is_finite(avg_base) else np.nan
        return avg_base, avg_exp, avg_abs, avg_rel
    if metric_type == "ratio":
        den_mean = df_control_metrics["den"].mean()
        if not _is_finite(den_mean) or float(den_mean) == 0:
            return np.nan, np.nan, np.nan, np.nan
        avg_base = df_control_metrics["value"].mean() / den_mean
        avg_exp = df_pilot_metrics["value"].mean() / den_mean
        avg_abs = avg_exp - avg_base
        return avg_base, avg_exp, avg_abs, np.nan
    return np.nan, np.nan, np.nan, np.nan

def calculate_p_value(df_control_metrics, df_pilot_metrics, metric_type):
    if metric_type in ["additive", "average", "ratio"]:
        x1 = df_control_metrics["value"].dropna()
        x2 = df_pilot_metrics["value"].dropna()
        if len(x1) < 2 or len(x2) < 2:
            return np.nan
        p_value = CompareMeans(DescrStatsW(x1), DescrStatsW(x2)).ttest_ind(alternative="two-sided", usevar="unequal")[1]
    elif metric_type == "median":
        x1 = df_control_metrics["value"].dropna()
        x2 = df_pilot_metrics["value"].dropna()
        if len(x1) == 0 or len(x2) == 0:
            return np.nan
        p_value = stats.mannwhitneyu(x1, x2, alternative="two-sided")[1]
    elif metric_type == "share":
        success = np.array([
            int(((df_control_metrics["den"] > 0) & (df_control_metrics["num"] > 0)).sum()),
            int(((df_pilot_metrics["den"] > 0) & (df_pilot_metrics["num"] > 0)).sum()),
        ])
        obs = np.array([int((df_control_metrics["den"] > 0).sum()), int((df_pilot_metrics["den"] > 0).sum())])
        p_value = np.nan if obs.min() == 0 else proportions_ztest(success, obs)[1]
    else:
        raise ValueError(f"Unsupported metric type: {metric_type}")
    return float(p_value) if _is_finite(p_value) else np.nan

def calculate_stat_pair(df_control_metrics, df_pilot_metrics, metric_type, direction, days_running: int, alpha=ALPHA, power=POWER):
    base_value, exp_value, abs_delta_value, rel_delta_value = calculate_base_exp_values(df_control_metrics, df_pilot_metrics, metric_type)
    avg_base_value, avg_exp_value, avg_abs_delta, avg_rel_delta = calculate_avg_base_exp_values(df_control_metrics, df_pilot_metrics, metric_type)
    mde, effect = calculate_mde(df_control_metrics, df_pilot_metrics, metric_type, alpha, power)
    number_samples, n_base, n_exp = calculate_number_samples(df_control_metrics, df_pilot_metrics)
    p_value = calculate_p_value(df_control_metrics, df_pilot_metrics, metric_type)

    if metric_type in ["additive", "average"]:
        observed_delta = avg_abs_delta
    elif metric_type in ["ratio", "share"]:
        observed_delta = abs_delta_value
    else:
        observed_delta = np.nan

    n_required_per_group = _required_n_per_group_for_observed_delta(df_control_metrics, df_pilot_metrics, metric_type, observed_delta, alpha, power)
    power_now = _power_for_delta(df_control_metrics, df_pilot_metrics, metric_type, observed_delta=observed_delta, alpha=alpha)
    days_more_base = _days_to_reach_required_n(n_base, n_required_per_group, days_running)
    days_more_exp = _days_to_reach_required_n(n_exp, n_required_per_group, days_running)
    vals = [x for x in [days_more_base, days_more_exp] if _is_finite(x)]
    days_more_if_same_delta = int(max(vals)) if vals else np.nan

    result = "neutral"
    if _is_finite(p_value) and p_value < alpha and _is_finite(abs_delta_value):
        if direction == "positive":
            result = "positive" if abs_delta_value > 0 else "negative"
        elif direction == "negative":
            result = "negative" if abs_delta_value > 0 else "positive"

    return {
        "value_base": base_value,
        "value_exp": exp_value,
        "abs_delta": abs_delta_value,
        "rel_delta": rel_delta_value,
        "p_value": p_value,
        "result": result,
        "mde": mde,
        "number_samples": number_samples,
        "number_samples_base": n_base,
        "number_samples_exp": n_exp,
        "avg_value_base": avg_base_value,
        "avg_value_exp": avg_exp_value,
        "avg_abs_delta": avg_abs_delta,
        "avg_rel_delta": avg_rel_delta,
        "days_more_if_same_delta": days_more_if_same_delta,
        "n_required_per_group": n_required_per_group,
        "power_now": power_now,
        "effect_size_proxy": effect,
        "days_more_base": days_more_base,
        "days_more_exp": days_more_exp,
    }

def _apply_pairwise_pvalue_adjustment(df_result, alpha=ALPHA, method="holm", group_cols=("metric_name",)):
    df_result = df_result.copy()
    df_result["p_value_adj"] = df_result["p_value"]
    df_result["is_significant_adj"] = df_result["p_value"] < alpha
    if method is None:
        return df_result
    for _, idx in df_result.groupby(list(group_cols)).groups.items():
        idx = list(idx)
        pvals = df_result.loc[idx, "p_value"].values.astype(float)
        mask = np.isfinite(pvals)
        if mask.sum() == 0:
            continue
        corrected = multipletests(pvals[mask], alpha=alpha, method=method)
        pvals_adj = np.full_like(pvals, fill_value=np.nan, dtype=float)
        sig_adj = np.full_like(mask, fill_value=False, dtype=bool)
        pvals_adj[mask] = corrected[1]
        sig_adj[mask] = corrected[0]
        df_result.loc[idx, "p_value_adj"] = pvals_adj
        df_result.loc[idx, "is_significant_adj"] = sig_adj
    return df_result

def _recompute_result_by_adjusted_p(df_result, alpha=ALPHA):
    df_result = df_result.copy()
    results_adj = []
    for _, row in df_result.iterrows():
        p = row.get("p_value_adj", np.nan)
        delta = row.get("abs_delta", np.nan)
        direction = row.get("direction", None)
        res = "neutral"
        if _is_finite(p) and p < alpha and _is_finite(delta):
            if direction == "positive":
                res = "positive" if delta > 0 else "negative"
            elif direction == "negative":
                res = "negative" if delta > 0 else "positive"
        results_adj.append(res)
    df_result["result_adj"] = results_adj
    return df_result

def _check_required_columns(df: pd.DataFrame, metrics_df: pd.DataFrame):
    required = {"user_id", "exp_group"}
    for _, row in metrics_df.iterrows():
        required.add(row["num"])
        required.add(row["den"])
    missing = [c for c in sorted(required) if c not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")

def run_abc_test(df: pd.DataFrame, metrics_df: pd.DataFrame, exp_start_date, exp_end_date, include_bc: bool = True, alpha: float = ALPHA, power: float = POWER, pvalue_adjust_method: str | None = "holm") -> pd.DataFrame:
    metrics_df = validate_metrics_config(metrics_df)
    _check_required_columns(df, metrics_df)
    days_running = _days_elapsed(exp_start_date, exp_end_date)

    pairs = [("A", "B", "A_vs_B"), ("A", "C", "A_vs_C")]
    if include_bc:
        pairs.append(("B", "C", "B_vs_C"))

    rows = []
    for metric in metrics_df.index:
        metric_type = str(metrics_df.loc[metric, "type"]).lower()
        metric_name = metrics_df.loc[metric, "desc"]
        metric_direction = str(metrics_df.loc[metric, "direction"]).lower()
        num = metrics_df.loc[metric, "num"]
        den = metrics_df.loc[metric, "den"]

        if num != den:
            df_metric = df[["user_id", "exp_group", num, den]].rename(columns={num: "num", den: "den"}).copy()
        else:
            df_metric = df[["user_id", "exp_group", num]].rename(columns={num: "num"}).copy()
            df_metric["den"] = 1.0

        for group_base, group_exp, pair_name in pairs:
            df_control_metrics, df_pilot_metrics = _prep_metric_pair_frames(df_metric, group_base, group_exp, metric_type)
            stat_row = calculate_stat_pair(df_control_metrics, df_pilot_metrics, metric_type, metric_direction, days_running, alpha, power)
            rows.append({
                "metric_name": metric_name,
                "metric_type": metric_type,
                "pair": pair_name,
                "group_base": group_base,
                "group_exp": group_exp,
                "direction": metric_direction,
                **stat_row,
            })

    df_result = pd.DataFrame(rows)
    df_result = _apply_pairwise_pvalue_adjustment(df_result, alpha=alpha, method=pvalue_adjust_method, group_cols=("metric_name",))
    df_result = _recompute_result_by_adjusted_p(df_result, alpha=alpha)
    df_result = df_result[AGGREGATE_COLUMN_NAMES_ABC].copy()
    return df_result

def format_result(val: str) -> str:
    if pd.isna(val):
        return ""
    v = str(val).lower()
    if v == "positive":
        return "background-color:#0ea75a; color:#ffffff; font-weight:700;"
    if v == "negative":
        return "background-color:#dc2626; color:#ffffff; font-weight:700;"
    return "background-color:#374151; color:#e5e7eb; font-weight:600;"

def highlight_pairs(row):
    pair = row.get("pair")
    if pair == "A_vs_B":
        return ["background-color: rgba(40, 120, 240, 0.10); border-top: 2px solid #1e3a8a;"] * len(row)
    if pair == "A_vs_C":
        return ["background-color: rgba(40, 240, 120, 0.08); border-top: 2px solid #166534;"] * len(row)
    if pair == "B_vs_C":
        return ["background-color: rgba(240, 200, 40, 0.08); border-top: 2px solid #92400e;"] * len(row)
    return [""] * len(row)

def style_table_abc(df_result: pd.DataFrame, caption: str = ""):
    styler = (
        df_result.style.set_table_attributes("style='display:inline'")
        .set_caption(caption)
        .apply(highlight_pairs, axis=1)
        .map(format_result, subset=["result", "result_adj"])
        .background_gradient(subset=["rel_delta"], axis=0, cmap="RdYlGn")
        .format("{:,.4f}", subset=["value_base", "value_exp", "abs_delta", "mde", "avg_value_base", "avg_value_exp", "avg_abs_delta", "p_value", "p_value_adj", "power_now"])
        .format("{:.2%}", subset=["rel_delta"])
        .format("{:.2f}", subset=["avg_rel_delta"])
        .format("{:.0f}", subset=["number_samples", "number_samples_base", "number_samples_exp", "days_more_if_same_delta", "days_more_base", "days_more_exp", "n_required_per_group"])
    )
    return styler
