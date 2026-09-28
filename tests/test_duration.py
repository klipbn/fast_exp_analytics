import numpy as np
import pandas as pd
import pytest

from fast_exp_analytics.duration import (
    build_experiment_level,
    default_duration_metrics_config,
    mde_table_from_aggregated_df,
    required_days_from_aggregated_df,
    units_per_day_from_raw,
)


def _raw_daily(n_units: int = 60, n_days: int = 30, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_rows = n_units * n_days // 2
    return pd.DataFrame(
        {
            "user_id": rng.choice([f"u{i}" for i in range(n_units)], size=n_rows),
            "date": rng.choice(
                pd.date_range("2025-01-01", periods=n_days, freq="D"), size=n_rows
            ),
            "amount": rng.exponential(50, size=n_rows),
            "shows": rng.integers(1, 1000, size=n_rows).astype(float),
            "clicks": rng.integers(0, 50, size=n_rows).astype(float),
            "main_goals": rng.integers(0, 5, size=n_rows).astype(float),
            "campaign_id": rng.integers(0, 3, size=n_rows),
        }
    )


def test_build_experiment_level_aggregates_totals_and_derived_metrics():
    raw = _raw_daily()
    agg = build_experiment_level(raw, unit_id_col="user_id")

    unit = raw[raw["user_id"] == raw["user_id"].iloc[0]]
    row = agg[agg["user_id"] == unit["user_id"].iloc[0]].iloc[0]

    assert row["total_shows"] == pytest.approx(unit["shows"].sum())
    assert row["total_amount"] == pytest.approx(unit["amount"].sum())
    assert row["life_days"] >= row["active_days"]
    assert row["ctr"] == pytest.approx(row["total_clicks"] / row["total_shows"])
    assert row["is_with_spend"] == int(row["total_amount"] > 0)


def test_units_per_day_from_raw_uses_unique_units_and_days():
    raw = _raw_daily(n_units=50, n_days=10)
    expected = raw["user_id"].nunique() / raw["date"].nunique()

    assert units_per_day_from_raw(raw, unit_id_col="user_id") == pytest.approx(expected)


def test_mde_table_shrinks_with_longer_duration():
    agg = build_experiment_level(_raw_daily(), unit_id_col="user_id")
    table = mde_table_from_aggregated_df(
        agg,
        units_per_day=30.0,
        rollout_pct=0.5,
        exp_days=[7, 28],
        metrics_config=default_duration_metrics_config().loc[["amount"]],
    )

    short, long_ = table.sort_values("Число дней экспа").iloc
    assert long_["Списания MDE, %"] < short["Списания MDE, %"]
    assert table["_n_per_group"].iloc[1] > table["_n_per_group"].iloc[0]


def test_required_days_grows_for_smaller_target_mde():
    agg = build_experiment_level(_raw_daily(), unit_id_col="user_id")
    days = required_days_from_aggregated_df(
        agg,
        units_per_day=30.0,
        rollout_pct=0.5,
        target_mde_pct=[5, 10],
        metrics_config=default_duration_metrics_config().loc[["amount"]],
    ).sort_values("Target MDE, %")

    assert days.iloc[0]["Списания — дней до target MDE"] > days.iloc[1][
        "Списания — дней до target MDE"
    ]


def test_invalid_rollout_raises():
    agg = build_experiment_level(_raw_daily(), unit_id_col="user_id")
    with pytest.raises(ValueError):
        mde_table_from_aggregated_df(agg, units_per_day=30.0, rollout_pct=0.0, exp_days=[7])


def test_group_shares_must_sum_to_one():
    agg = build_experiment_level(_raw_daily(), unit_id_col="user_id")
    with pytest.raises(ValueError):
        mde_table_from_aggregated_df(
            agg, units_per_day=30.0, rollout_pct=0.5, exp_days=[7], group_shares=[0.5, 0.5, 0.5]
        )


def test_missing_source_col_raises_key_error():
    agg = build_experiment_level(_raw_daily(), unit_id_col="user_id")
    bad_config = pd.DataFrame(
        {"label": ["Broken"], "source_col": ["nope"]}, index=["broken"]
    )
    with pytest.raises(KeyError):
        mde_table_from_aggregated_df(
            agg, units_per_day=30.0, rollout_pct=0.5, exp_days=[7], metrics_config=bad_config
        )


def test_required_days_rejects_non_positive_targets():
    agg = build_experiment_level(_raw_daily(), unit_id_col="user_id")
    with pytest.raises(ValueError):
        required_days_from_aggregated_df(
            agg, units_per_day=30.0, rollout_pct=0.5, target_mde_pct=[0, -5]
        )


def test_nan_mde_when_metric_has_zero_mean():
    agg = build_experiment_level(_raw_daily(), unit_id_col="user_id")
    agg["zero_metric"] = 0.0
    config = pd.DataFrame(
        {"label": ["Zero MDE, %"], "source_col": ["zero_metric"]}, index=["zero"]
    )
    table = mde_table_from_aggregated_df(
        agg, units_per_day=30.0, rollout_pct=0.5, exp_days=[7], metrics_config=config
    )

    assert np.isnan(table["Zero MDE, %"].iloc[0])
