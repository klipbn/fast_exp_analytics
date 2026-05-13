from __future__ import annotations

import math
from typing import Mapping, Sequence
import numpy as np
import pandas as pd
from scipy.stats import norm


DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80


# Generic helpers
# =========================
def _to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    a = _to_numeric(a)
    b = _to_numeric(b)
    out = a / b.replace({0: np.nan})
    return out.replace([np.inf, -np.inf], np.nan)


def _normalize_rollout(rollout_pct: float) -> float:
    rollout = float(rollout_pct)
    if rollout > 1:
        rollout = rollout / 100.0

    if not (0 < rollout <= 1):
        raise ValueError("rollout_pct must be in (0, 1] or (0, 100].")

    return rollout


def _z_sum(alpha: float = DEFAULT_ALPHA, power: float = DEFAULT_POWER) -> float:
    return float(norm.ppf(1 - alpha / 2) + norm.ppf(power))


def _validate_days(days: Sequence[int] | int) -> list[int]:
    if isinstance(days, int):
        days = [days]

    days = [int(x) for x in days]
    if any(x <= 0 for x in days):
        raise ValueError("All exp_days must be positive integers.")

    return days


def _resolve_group_shares(
    experiment_type: str = "ab",
    group_shares: Sequence[float] | None = None,
) -> tuple[float, ...]:
    experiment_type = str(experiment_type).lower()

    if group_shares is None:
        if experiment_type == "ab":
            shares = (0.5, 0.5)
        elif experiment_type == "abc":
            shares = (1 / 3, 1 / 3, 1 / 3)
        else:
            raise ValueError("experiment_type must be 'ab' or 'abc'.")
    else:
        shares = tuple(float(x) for x in group_shares)

    if len(shares) < 2:
        raise ValueError("group_shares must contain at least 2 groups.")

    if any(x <= 0 for x in shares):
        raise ValueError("All group shares must be positive.")

    total = sum(shares)
    if not np.isclose(total, 1.0):
        raise ValueError(f"group_shares must sum to 1.0, got {total:.6f}")

    return shares


def _smallest_group_share(group_shares: Sequence[float]) -> float:
    return float(min(group_shares))


# =========================
# Metric config
def default_duration_metrics_config() -> pd.DataFrame:
    """
    source_col должен существовать в aggregated dataframe.
    transform:
        - none
        - log1p
    """
    return pd.DataFrame.from_dict(
        {
            "shows": {
                "label": "Показы MDE, %",
                "source_col": "total_shows",
                "transform": "none",
            },
            "clicks": {
                "label": "Клики MDE, %",
                "source_col": "total_clicks",
                "transform": "none",
            },
            "amount": {
                "label": "Списания MDE, %",
                "source_col": "total_amount",
                "transform": "none",
            },
            "goals": {
                "label": "Цели MDE, %",
                "source_col": "total_goals",
                "transform": "none",
            },
            "cpm": {
                "label": "CPM MDE, %",
                "source_col": "cpm",
                "transform": "none",
            },
            "ctr": {
                "label": "CTR MDE, %",
                "source_col": "ctr",
                "transform": "none",
            },
            "cpa": {
                "label": "CPA MDE, %",
                "source_col": "cpa",
                "transform": "none",
            },
            "life_days": {
                "label": "Life days MDE, %",
                "source_col": "life_days",
                "transform": "none",
            },
            "amount_per_day": {
                "label": "Amount per day MDE, %",
                "source_col": "amount_per_day",
                "transform": "none",
            },
            "goals_per_day": {
                "label": "Goals per day MDE, %",
                "source_col": "goals_per_day",
                "transform": "none",
            },
            "active_days": {
                "label": "Active days MDE, %",
                "source_col": "active_days",
                "transform": "none",
            },
            "entities_cnt": {
                "label": "Entities count MDE, %",
                "source_col": "entities_cnt",
                "transform": "none",
            },
        },
        orient="index",
    )


def _normalize_metrics_config(
    metrics_config: pd.DataFrame | Mapping[str, Mapping[str, str]] | None = None,
) -> pd.DataFrame:
    if metrics_config is None:
        metrics_df = default_duration_metrics_config().copy()
    elif isinstance(metrics_config, pd.DataFrame):
        metrics_df = metrics_config.copy()
    else:
        metrics_df = pd.DataFrame.from_dict(metrics_config, orient="index")

    required_cols = {"label", "source_col"}
    missing = required_cols - set(metrics_df.columns)
    if missing:
        raise ValueError(f"metrics_config is missing required columns: {sorted(missing)}")

    if "transform" not in metrics_df.columns:
        metrics_df["transform"] = "none"

    return metrics_df


# =========================
# aggregation
def build_experiment_level(
    df_raw: pd.DataFrame,
    *,
    unit_id_col: str,
    date_col: str = "date",
    amount_col: str = "amount",
    shows_col: str = "shows",
    clicks_col: str = "clicks",
    goals_col: str = "main_goals",
    entity_id_col: str | None = "campaign_id",
    extra_group_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    агрегат на уровне unit_id_col.

    Например:
    - unit_id_col="user_id"
    - unit_id_col="campaign_id"
    - unit_id_col="ad_plan_id"
    - unit_id_col="hid"
    """
    df = df_raw.copy()

    required = [unit_id_col, date_col, amount_col, shows_col, clicks_col, goals_col]
    if entity_id_col:
        required.append(entity_id_col)
    if extra_group_cols:
        required.extend(extra_group_cols)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in df_raw: {missing}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    for c in [amount_col, shows_col, clicks_col, goals_col]:
        df[c] = _to_numeric(df[c]).fillna(0.0)

    group_cols = [unit_id_col]
    if extra_group_cols:
        group_cols.extend(extra_group_cols)

    agg_dict = {
        "start_date": (date_col, "min"),
        "end_date": (date_col, "max"),
        "active_days": (date_col, lambda s: s.dt.date.nunique()),
        "total_amount": (amount_col, "sum"),
        "total_shows": (shows_col, "sum"),
        "total_clicks": (clicks_col, "sum"),
        "total_goals": (goals_col, "sum"),
    }

    if entity_id_col:
        agg_dict["entities_cnt"] = (entity_id_col, "nunique")

    agg = (
        df.groupby(group_cols, as_index=False)
        .agg(**agg_dict)
    )

    agg["life_days"] = (
        (pd.to_datetime(agg["end_date"]) - pd.to_datetime(agg["start_date"])).dt.days + 1
    ).clip(lower=1)

    if "entities_cnt" not in agg.columns:
        agg["entities_cnt"] = np.nan

    agg["cpm"] = _safe_div(agg["total_amount"] * 1000.0, agg["total_shows"])
    agg["ctr"] = _safe_div(agg["total_clicks"], agg["total_shows"])
    agg["cpa"] = _safe_div(agg["total_amount"], agg["total_goals"])
    agg["amount_per_day"] = _safe_div(agg["total_amount"], agg["active_days"].clip(lower=1))
    agg["goals_per_day"] = _safe_div(agg["total_goals"], agg["active_days"].clip(lower=1))

    agg["is_with_spend"] = (agg["total_amount"] > 0).astype(int)
    agg["is_with_goals"] = (agg["total_goals"] > 0).astype(int)

    return agg


def units_per_day_from_raw(
    df_raw: pd.DataFrame,
    *,
    date_col: str = "date",
    unit_id_col: str,
) -> float:
    """
    Среднее число уникальных unit_id_col в день по сырому датасету.
    """
    df = df_raw.copy()

    if date_col not in df.columns or unit_id_col not in df.columns:
        raise KeyError(f"df_raw must contain '{date_col}' and '{unit_id_col}'")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    n_days = df[date_col].dt.date.nunique()
    n_units = df[unit_id_col].nunique()

    return float(n_units / n_days) if n_days else float("nan")


# =========================
# MDE
def mde_rel_percent_from_series(
    x: pd.Series,
    n_per_group: int,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
) -> float:
    x = _to_numeric(x).replace([np.inf, -np.inf], np.nan).dropna()

    if n_per_group <= 1 or x.empty:
        return float("nan")

    mu = float(x.mean())
    sigma = float(x.std(ddof=1))

    if mu == 0:
        return float("nan")
    if sigma == 0:
        return 0.0

    z = _z_sum(alpha, power)
    mde_abs = z * sigma * math.sqrt(2.0 / n_per_group)
    return float(mde_abs / abs(mu) * 100.0)


def required_n_per_group_for_target_mde(
    x: pd.Series,
    target_mde_pct: float,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
) -> float:
    x = _to_numeric(x).replace([np.inf, -np.inf], np.nan).dropna()

    if x.empty:
        return float("nan")

    mu = float(x.mean())
    sigma = float(x.std(ddof=1))

    if mu == 0 or sigma == 0:
        return float("nan")

    target_rel = float(target_mde_pct) / 100.0
    if target_rel <= 0:
        raise ValueError("target_mde_pct must be positive.")

    z = _z_sum(alpha, power)
    n = 2.0 * ((z * sigma) / (abs(mu) * target_rel)) ** 2
    return float(math.ceil(n))


def _apply_transform(x: pd.Series, transform: str | None) -> pd.Series:
    transform = (transform or "none").lower()

    if transform == "none":
        return x
    if transform == "log1p":
        return np.log1p(x.clip(lower=0))

    raise ValueError(f"Unsupported transform: {transform}")


def _build_series_map(
    agg_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> dict[str, tuple[str, pd.Series]]:
    out: dict[str, tuple[str, pd.Series]] = {}

    for metric_key, row in metrics_df.iterrows():
        source_col = row["source_col"]
        label = row["label"]
        transform = row.get("transform", "none")

        if source_col not in agg_df.columns:
            raise KeyError(f"source_col '{source_col}' is not present in aggregated dataframe")

        series = _apply_transform(agg_df[source_col], transform)
        out[metric_key] = (label, series)

    return out


def mde_table_from_aggregated_df(
    agg_df: pd.DataFrame,
    *,
    units_per_day: float,
    rollout_pct: float,
    exp_days: Sequence[int] | int,
    experiment_type: str = "ab",
    group_shares: Sequence[float] | None = None,
    metrics_config: pd.DataFrame | Mapping[str, Mapping[str, str]] | None = None,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
    include_debug_columns: bool = True,
) -> pd.DataFrame:
    rollout = _normalize_rollout(rollout_pct)
    days_list = _validate_days(exp_days)
    shares = _resolve_group_shares(experiment_type=experiment_type, group_shares=group_shares)
    metrics_df = _normalize_metrics_config(metrics_config)
    series_map = _build_series_map(agg_df, metrics_df)

    eligible_per_day = units_per_day * rollout
    smallest_share = _smallest_group_share(shares)

    rows: list[dict[str, float | int | str]] = []

    for T in days_list:
        n_per_group = int(math.floor(eligible_per_day * T * smallest_share))

        row: dict[str, float | int | str] = {
            "% Выкатки": rollout * 100.0,
            "Число дней экспа": int(T),
            "Тип эксперимента": experiment_type.upper(),
            "Число групп": len(shares),
        }

        for _, (label, series) in series_map.items():
            row[label] = mde_rel_percent_from_series(
                x=series,
                n_per_group=n_per_group,
                alpha=alpha,
                power=power,
            )

        if include_debug_columns:
            row["_units_per_day"] = units_per_day
            row["_eligible_per_day"] = eligible_per_day
            row["_smallest_group_share"] = smallest_share
            row["_n_per_group"] = n_per_group
            row["_alpha"] = alpha
            row["_power"] = power
            row["_group_shares"] = ",".join(f"{x:.4f}" for x in shares)

        rows.append(row)

    return pd.DataFrame(rows)


def required_days_from_aggregated_df(
    agg_df: pd.DataFrame,
    *,
    units_per_day: float,
    rollout_pct: float,
    target_mde_pct: float | Sequence[float],
    experiment_type: str = "ab",
    group_shares: Sequence[float] | None = None,
    metrics_config: pd.DataFrame | Mapping[str, Mapping[str, str]] | None = None,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
    include_debug_columns: bool = True,
) -> pd.DataFrame:
    rollout = _normalize_rollout(rollout_pct)
    shares = _resolve_group_shares(experiment_type=experiment_type, group_shares=group_shares)
    metrics_df = _normalize_metrics_config(metrics_config)

    if isinstance(target_mde_pct, (int, float)):
        target_mdes = [float(target_mde_pct)]
    else:
        target_mdes = [float(x) for x in target_mde_pct]

    if any(x <= 0 for x in target_mdes):
        raise ValueError("All target_mde_pct values must be positive.")

    series_map = _build_series_map(agg_df, metrics_df)

    eligible_per_day = units_per_day * rollout
    smallest_share = _smallest_group_share(shares)

    if eligible_per_day <= 0 or smallest_share <= 0:
        raise ValueError("eligible_per_day and smallest_group_share must be positive.")

    rows: list[dict[str, float | int | str]] = []

    for target in target_mdes:
        row: dict[str, float | int | str] = {
            "% Выкатки": rollout * 100.0,
            "Target MDE, %": float(target),
            "Тип эксперимента": experiment_type.upper(),
            "Число групп": len(shares),
        }

        for _, (label, series) in series_map.items():
            n_req = required_n_per_group_for_target_mde(
                x=series,
                target_mde_pct=target,
                alpha=alpha,
                power=power,
            )

            days_req = np.nan if pd.isna(n_req) else math.ceil(n_req / (eligible_per_day * smallest_share))

            pretty_label = label.replace(" MDE, %", " — дней до target MDE")
            row[pretty_label] = days_req

            if include_debug_columns:
                debug_label = label.replace(" MDE, %", " — n_per_group")
                row[debug_label] = n_req

        if include_debug_columns:
            row["_units_per_day"] = units_per_day
            row["_eligible_per_day"] = eligible_per_day
            row["_smallest_group_share"] = smallest_share
            row["_alpha"] = alpha
            row["_power"] = power
            row["_group_shares"] = ",".join(f"{x:.4f}" for x in shares)

        rows.append(row)

    return pd.DataFrame(rows)


def mde_table_for_experiment_duration(
    df_raw: pd.DataFrame,
    *,
    unit_id_col: str,
    rollout_pct: float,
    exp_days: Sequence[int] | int,
    experiment_type: str = "ab",
    group_shares: Sequence[float] | None = None,
    metrics_config: pd.DataFrame | Mapping[str, Mapping[str, str]] | None = None,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
    date_col: str = "date",
    amount_col: str = "amount",
    shows_col: str = "shows",
    clicks_col: str = "clicks",
    goals_col: str = "main_goals",
    entity_id_col: str | None = "campaign_id",
    extra_group_cols: Sequence[str] | None = None,
    include_debug_columns: bool = True,
) -> pd.DataFrame:
    agg_df = build_experiment_level(
        df_raw=df_raw,
        unit_id_col=unit_id_col,
        date_col=date_col,
        amount_col=amount_col,
        shows_col=shows_col,
        clicks_col=clicks_col,
        goals_col=goals_col,
        entity_id_col=entity_id_col,
        extra_group_cols=extra_group_cols,
    )

    upd = units_per_day_from_raw(
        df_raw=df_raw,
        date_col=date_col,
        unit_id_col=unit_id_col,
    )

    return mde_table_from_aggregated_df(
        agg_df=agg_df,
        units_per_day=upd,
        rollout_pct=rollout_pct,
        exp_days=exp_days,
        experiment_type=experiment_type,
        group_shares=group_shares,
        metrics_config=metrics_config,
        alpha=alpha,
        power=power,
        include_debug_columns=include_debug_columns,
    )


def required_days_for_target_mde_table(
    df_raw: pd.DataFrame,
    *,
    unit_id_col: str,
    rollout_pct: float,
    target_mde_pct: float | Sequence[float],
    experiment_type: str = "ab",
    group_shares: Sequence[float] | None = None,
    metrics_config: pd.DataFrame | Mapping[str, Mapping[str, str]] | None = None,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
    date_col: str = "date",
    amount_col: str = "amount",
    shows_col: str = "shows",
    clicks_col: str = "clicks",
    goals_col: str = "main_goals",
    entity_id_col: str | None = "campaign_id",
    extra_group_cols: Sequence[str] | None = None,
    include_debug_columns: bool = True,
) -> pd.DataFrame:
    agg_df = build_experiment_level(
        df_raw=df_raw,
        unit_id_col=unit_id_col,
        date_col=date_col,
        amount_col=amount_col,
        shows_col=shows_col,
        clicks_col=clicks_col,
        goals_col=goals_col,
        entity_id_col=entity_id_col,
        extra_group_cols=extra_group_cols,
    )

    upd = units_per_day_from_raw(
        df_raw=df_raw,
        date_col=date_col,
        unit_id_col=unit_id_col,
    )

    return required_days_from_aggregated_df(
        agg_df=agg_df,
        units_per_day=upd,
        rollout_pct=rollout_pct,
        target_mde_pct=target_mde_pct,
        experiment_type=experiment_type,
        group_shares=group_shares,
        metrics_config=metrics_config,
        alpha=alpha,
        power=power,
        include_debug_columns=include_debug_columns,
    )


def duration_plan_summary(
    df_raw: pd.DataFrame,
    *,
    unit_id_col: str,
    rollout_pct: float,
    exp_days: Sequence[int] | int,
    target_mde_pct: float | Sequence[float],
    experiment_type: str = "ab",
    group_shares: Sequence[float] | None = None,
    metrics_config: pd.DataFrame | Mapping[str, Mapping[str, str]] | None = None,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
    date_col: str = "date",
    amount_col: str = "amount",
    shows_col: str = "shows",
    clicks_col: str = "clicks",
    goals_col: str = "main_goals",
    entity_id_col: str | None = "campaign_id",
    extra_group_cols: Sequence[str] | None = None,
) -> dict[str, pd.DataFrame]:
    agg_df = build_experiment_level(
        df_raw=df_raw,
        unit_id_col=unit_id_col,
        date_col=date_col,
        amount_col=amount_col,
        shows_col=shows_col,
        clicks_col=clicks_col,
        goals_col=goals_col,
        entity_id_col=entity_id_col,
        extra_group_cols=extra_group_cols,
    )

    upd = units_per_day_from_raw(
        df_raw=df_raw,
        date_col=date_col,
        unit_id_col=unit_id_col,
    )

    mde_by_days = mde_table_from_aggregated_df(
        agg_df=agg_df,
        units_per_day=upd,
        rollout_pct=rollout_pct,
        exp_days=exp_days,
        experiment_type=experiment_type,
        group_shares=group_shares,
        metrics_config=metrics_config,
        alpha=alpha,
        power=power,
        include_debug_columns=True,
    )

    days_for_target = required_days_from_aggregated_df(
        agg_df=agg_df,
        units_per_day=upd,
        rollout_pct=rollout_pct,
        target_mde_pct=target_mde_pct,
        experiment_type=experiment_type,
        group_shares=group_shares,
        metrics_config=metrics_config,
        alpha=alpha,
        power=power,
        include_debug_columns=True,
    )

    return {
        "aggregated_df": agg_df,
        "mde_by_days": mde_by_days,
        "days_for_target_mde": days_for_target,
    }