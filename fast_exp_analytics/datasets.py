from __future__ import annotations
import numpy as np
import pandas as pd

def make_synthetic_abc_dataset(n: int = 100_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {
        "exp_group": rng.choice(["A", "B", "C"], size=n, p=[0.33, 0.33, 0.34]),
        "user_id": rng.integers(1000, 1_500_000, size=n),
        "is_create_ad": rng.binomial(1, 0.18, size=n),
        "is_with_payments": rng.binomial(1, 0.09, size=n),
        "is_with_spents": rng.binomial(1, 0.13, size=n),
        "campaigns": rng.integers(0, 8, size=n),
        "campaigns_with_spents": rng.integers(0, 5, size=n),
        "amount": np.round(rng.exponential(scale=180, size=n) * rng.binomial(1, 0.27, size=n), 4),
        "a_amount_payment": np.where(
            rng.binomial(1, 0.08, size=n) == 1,
            np.round(rng.exponential(scale=95, size=n), 4),
            np.nan,
        ),
        "shows": (rng.integers(0, 3500, size=n) * rng.binomial(1, 0.19, size=n)).astype(int),
        "clicks": (rng.integers(0, 220, size=n) * rng.binomial(1, 0.15, size=n)).astype(int),
        "goals": (rng.integers(0, 30, size=n) * rng.binomial(1, 0.10, size=n)).astype(int),
        "camp_days_total": np.round(rng.exponential(scale=6.5, size=n) * rng.binomial(1, 0.22, size=n), 4),
        "camp_days_avg": np.round(rng.uniform(0.5, 15, size=n) * rng.binomial(1, 0.22, size=n), 4),
        "camp_days_max": np.round(rng.uniform(1, 30, size=n) * rng.binomial(1, 0.22, size=n), 4),
    }
    df = pd.DataFrame(data)
    df.loc[df["exp_group"] == "B", "amount"] *= 5
    df.loc[df["campaigns"] == 0, ["camp_days_total", "camp_days_avg", "camp_days_max"]] = 0.0
    df.loc[df["campaigns_with_spents"] > 0, "is_with_spents"] = 1
    df.loc[df["campaigns"] > 0, "is_create_ad"] = 1
    df["camp_life_days"] = df["camp_days_total"]
    df["amount_per_day"] = df["amount"] / df["camp_life_days"].clip(lower=1)
    df["goals_per_day"] = df["goals"] / df["camp_life_days"].clip(lower=1)
    df["amount_per_day"] = df["amount_per_day"].replace([np.inf, -np.inf], np.nan)
    df["goals_per_day"] = df["goals_per_day"].replace([np.inf, -np.inf], np.nan)
    df["is_in_exp"] = 1
    df["amount_1000"] = df["amount"] * 1000
    for col in ["camp_days_total", "camp_days_avg", "camp_days_max"]:
        df[col] = df[col].replace(0, np.nan)
    int_cols = ["user_id", "is_create_ad", "is_with_payments", "is_with_spents", "campaigns", "campaigns_with_spents", "shows", "clicks", "goals"]
    for col in int_cols:
        df[col] = df[col].astype(int)
    return df


