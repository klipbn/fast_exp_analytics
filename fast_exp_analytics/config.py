from __future__ import annotations

import pandas as pd

_REQUIRED_COLUMNS = ["desc", "type", "num", "den", "direction"]
_ALLOWED_TYPES = {"additive", "average", "ratio", "share", "median"}
_ALLOWED_DIRECTIONS = {"positive", "negative"}

def validate_metrics_config(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(metrics_df, pd.DataFrame):
        raise TypeError("metrics_df must be a pandas DataFrame")
    missing = [c for c in _REQUIRED_COLUMNS if c not in metrics_df.columns]
    if missing:
        raise ValueError(f"metrics_df is missing required columns: {missing}")
    if metrics_df.index.duplicated().any():
        dup = metrics_df.index[metrics_df.index.duplicated()].tolist()
        raise ValueError(f"metrics_df has duplicated metric names in index: {dup}")
    bad_types = sorted(set(metrics_df["type"].astype(str).str.lower()) - _ALLOWED_TYPES)
    if bad_types:
        raise ValueError(f"Unsupported metric types: {bad_types}")
    bad_dirs = sorted(set(metrics_df["direction"].astype(str).str.lower()) - _ALLOWED_DIRECTIONS)
    if bad_dirs:
        raise ValueError(f"Unsupported directions: {bad_dirs}")
    return metrics_df.copy()

def default_metrics_config() -> pd.DataFrame:
    return pd.DataFrame.from_dict(
        {
            "shows": ["Показы", "additive", "shows", "shows", "positive"],
            "clicks": ["Клики", "additive", "clicks", "clicks", "positive"],
            "amount": ["Списания", "additive", "amount", "amount", "positive"],
            "a_amount_payment": ["Живые пополнения", "average", "a_amount_payment", "a_amount_payment", "positive"],
            "cpm": ["CPM", "ratio", "amount_1000", "shows", "negative"],
            "ctr": ["CTR", "ratio", "clicks", "shows", "positive"],
            "cpc": ["CPC", "ratio", "amount", "clicks", "negative"],
            "goals": ["Цели", "additive", "goals", "goals", "positive"],
            "cpa": ["CPA", "ratio", "amount", "goals", "negative"],
            "cr_users_created_ad": ["Конверсия в создавших кампанию", "share", "is_create_ad", "is_in_exp", "positive"],
            "cr_users_with_spents": ["Конверсия в начавших тратить", "share", "is_with_spents", "is_in_exp", "positive"],
            "cr_users_created_to_spents": ["Конверсия из создавших в начавших тратить", "share", "is_with_spents", "is_create_ad", "positive"],
            "camp_days_total": ["Суммарные дни открутки", "average", "camp_days_total", "camp_days_total", "positive"],
            "camp_days_avg": ["Средние дни открутки", "average", "camp_days_avg", "is_in_exp", "positive"],
            "camp_days_max": ["Campaign life days", "average", "camp_days_max", "is_in_exp", "positive"],
            "amount_per_day": ["Amount per day", "average", "amount_per_day", "is_in_exp", "positive"],
            "goals_per_day": ["Goals per day", "average", "goals_per_day", "is_in_exp", "positive"],
        },
        orient="index",
        columns=["desc", "type", "num", "den", "direction"],
    )
