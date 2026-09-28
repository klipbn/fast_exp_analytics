import pandas as pd
import pytest

from fast_exp_analytics.config import default_metrics_config, validate_metrics_config


def _config(rows: dict) -> pd.DataFrame:
    return pd.DataFrame.from_dict(
        rows, orient="index", columns=["desc", "type", "num", "den", "direction"]
    )


def test_valid_config_is_returned_as_copy():
    cfg = _config({"ctr": ["CTR", "ratio", "clicks", "shows", "positive"]})

    validated = validate_metrics_config(cfg)

    assert validated.equals(cfg)
    assert validated is not cfg


def test_missing_required_column_raises():
    cfg = pd.DataFrame({"desc": ["CTR"], "type": ["ratio"]})

    with pytest.raises(ValueError, match="missing required columns"):
        validate_metrics_config(cfg)


def test_non_dataframe_input_raises():
    with pytest.raises(TypeError):
        validate_metrics_config([("CTR", "ratio", "clicks", "shows", "positive")])


def test_unsupported_metric_type_raises():
    cfg = _config({"x": ["X", "banana", "a", "b", "positive"]})

    with pytest.raises(ValueError, match="Unsupported metric types"):
        validate_metrics_config(cfg)


def test_unsupported_direction_raises():
    cfg = _config({"x": ["X", "ratio", "a", "b", "sideways"]})

    with pytest.raises(ValueError, match="Unsupported directions"):
        validate_metrics_config(cfg)


def test_duplicated_metric_names_raise():
    cfg = pd.DataFrame(
        [
            ["CTR", "ratio", "clicks", "shows", "positive"],
            ["CTR copy", "ratio", "clicks", "shows", "positive"],
        ],
        index=["ctr", "ctr"],
        columns=["desc", "type", "num", "den", "direction"],
    )

    with pytest.raises(ValueError, match="duplicated metric names"):
        validate_metrics_config(cfg)


def test_default_metrics_config_is_valid():
    validated = validate_metrics_config(default_metrics_config())

    assert {"additive", "average", "ratio", "share"}.issubset(
        set(validated["type"].str.lower())
    )
