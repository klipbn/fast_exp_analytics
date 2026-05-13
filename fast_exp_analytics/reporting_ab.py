from __future__ import annotations

import html
import math
from urllib.parse import quote, urlencode

import pandas as pd
import numpy as np

METRIC_TYPE_LABELS = {
    "additive": "абс.",
    "ratio": "отн.",
    "share": "%",
    "average": "ср.",
    "median": "med",
}


def _is_nan(x) -> bool:
    try:
        return x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))
    except Exception:
        return True


def _fmt_metric_type(metric_type: str) -> str:
    if not metric_type:
        return ""
    return METRIC_TYPE_LABELS.get(str(metric_type).lower(), str(metric_type).lower())


def _fmt_float(x, decimals: int = 2) -> str:
    if _is_nan(x):
        return "н/д"
    return f"{float(x):.{decimals}f}".replace(".", ",")


def _fmt_pct(x) -> str:
    if _is_nan(x):
        return "н/д"
    return f"{float(x):.1f}%".replace(".", ",")


def _fmt_share01_to_pct(x, decimals: int = 2) -> str:
    if _is_nan(x):
        return "н/д"
    return f"{float(x) * 100:.{decimals}f}%".replace(".", ",")


def _fmt_p(p) -> str:
    if _is_nan(p):
        return "н/д"
    p = float(p)
    if p < 0.0001:
        return "&lt;0,0001"
    return f"{p:.4f}".replace(".", ",")


def _fmt_compact(x) -> str:
    if _is_nan(x):
        return "н/д"

    x = float(x)
    sgn = "-" if x < 0 else ""
    x = abs(x)

    if x >= 1e9:
        return f"{sgn}{(x / 1e9):.2f}B".replace(".", ",")
    if x >= 1e6:
        return f"{sgn}{(x / 1e6):.2f}M".replace(".", ",")
    if x >= 1e3:
        return f"{sgn}{(x / 1e3):.2f}K".replace(".", ",")
    return f"{sgn}{x:.4f}".replace(".", ",")


def _fmt_value_by_type(x, metric_type: str, metric_name: str | None = None) -> str:
    metric_type = (metric_type or "").lower()
    metric_name = (metric_name or "").upper()

    if metric_name == "CTR":
        return _fmt_share01_to_pct(x, decimals=3)

    if metric_type == "share":
        return _fmt_share01_to_pct(x, decimals=2)

    if metric_type == "additive":
        return _fmt_compact(x)

    return _fmt_float(x, 2)


def _fmt_days_more_human(x) -> str:
    if _is_nan(x):
        return "н/д"

    try:
        v = float(x)
        if math.isinf(v):
            return "н/д"
        if v <= 0:
            return "0д"
        if v <= 60:
            return f"{int(round(v))}д"
        if v <= 365:
            months = int(round(v / 30))
            return f"~{months} мес"
        return "слишком долго"
    except Exception:
        return "н/д"


def build_dashboard_url_ab(
    *,
    base_url: str,
    date_from: str,
    date_to: str,
    exp_id: str | int,
    extra_params: dict[str, str | int] | None = None,
) -> str:
    params: dict[str, str | int] = {
        "p_date_start": date_from,
        "p_date_end": date_to,
        "p_exp_id": exp_id,
        "x-horizon-role-mode": "true",
    }
    if extra_params:
        params.update(extra_params)

    return f"{base_url}?{urlencode({k: str(v) for k, v in params.items()}, quote_via=quote, safe='')}"


def _result_icon(result: str) -> str:
    result = str(result).lower()
    if result == "positive":
        return "🟢"
    if result == "negative":
        return "🔴"
    return "⚪"

def build_ab_chat_message(
    df_result: pd.DataFrame,
    *,
    experiment_desc: str,
    exp_id: str | int,
    date_from: str,
    date_to: str,
    dashboard_url: str,
    max_metrics: int | None = None,
    show_days_more: bool = True,
    sort_by: str = "abs_rel_delta",
) -> str:
    df = df_result.copy()

    # считаем длительность периода включительно
    period_days_text = ""
    try:
        dt_from = pd.to_datetime(date_from).date()
        dt_to = pd.to_datetime(date_to).date()
        days_cnt = (dt_to - dt_from).days + 1
        if days_cnt > 0:
            period_days_text = f" ({days_cnt} д.)"
    except Exception:
        period_days_text = ""

    period_text = f"{html.escape(date_from)} - {html.escape(date_to)}{period_days_text}"

    if df.empty:
        return (
            f"<b>AB тест: {html.escape(str(experiment_desc))}</b>\n"
            f"<b>Эксперимент:</b> {html.escape(str(exp_id))}\n"
            f"<b>Период:</b> {period_text}\n\n"
            f"Нет данных для отображения.\n\n"
            f"🔎 Подробнее дашборд:\n{dashboard_url}"
        )

    for col in ["p_value", "rel_delta", "value_base", "value_exp", "days_more_if_same_delta"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["abs_rel_delta"] = df["rel_delta"].abs()
    if sort_by == "p_value":
        df = df.sort_values(["p_value", "abs_rel_delta"], ascending=[True, False])
    else:
        df = df.sort_values(["abs_rel_delta", "p_value"], ascending=[False, True])

    if max_metrics is not None:
        df = df.head(max_metrics)

    lines = [
        f"<b>AB тест: {html.escape(str(experiment_desc))}</b>",
        f"<b>Эксперимент:</b> {html.escape(str(exp_id))}",
        f"<b>Период:</b> {period_text}",
        "",
    ]

    for _, row in df.iterrows():
        icon = _result_icon(row.get("result"))
        metric_name = html.escape(str(row.get("metric_name", "metric")))
        metric_type = _fmt_metric_type(str(row.get("metric_type", "")))
        base = _fmt_value_by_type(
            row.get("value_base"),
            str(row.get("metric_type", "")),
            str(row.get("metric_name", "")),
        )
        exp = _fmt_value_by_type(
            row.get("value_exp"),
            str(row.get("metric_type", "")),
            str(row.get("metric_name", "")),
        )
        delta = _fmt_pct(
            row.get("rel_delta", np.nan) * 100
            if not _is_nan(row.get("rel_delta"))
            else np.nan
        )
        p_value = _fmt_p(row.get("p_value"))

        extras = [f"p={p_value}"]
        if show_days_more:
            dmore = _fmt_days_more_human(row.get("days_more_if_same_delta"))
            if dmore not in {"н/д", "слишком долго"}:
                extras.append(f"ещё~{dmore}")

        lines.append(
            f"{icon} {metric_name} ({metric_type}): {base} → {exp} | Δ {delta} | "
            + " | ".join(extras)
        )

    lines.extend(["", "", f"🔎 Подробнее дашборд:\n{dashboard_url}"])
    return "\n".join(lines)