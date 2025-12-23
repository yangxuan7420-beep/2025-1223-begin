from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

import numpy as np
import pandas as pd


BASE_METRIC_KEYS: tuple[str, ...] = (
    "营业收入",
    "归母净利润",
    "经营活动产生的现金流量净额",
    "货币资金",
    "有息负债",
)

DERIVED_REQUIRED_KEYS: tuple[str, ...] = ("资本性支出", "应收账款")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.number)) and not isinstance(value, bool)


def _series_from_mapping(data: Mapping[str, Any], key: str) -> pd.Series:
    value = data.get(key, np.nan)
    if isinstance(value, pd.Series):
        return value
    if isinstance(value, pd.DataFrame):
        flattened = value.stack(dropna=False)
        return flattened if isinstance(flattened, pd.Series) else pd.Series([np.nan])
    if isinstance(value, Mapping):
        return pd.Series(value)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        items = list(value)
        return pd.Series(items if items else [np.nan])
    return pd.Series([value])


def _series_from_dataframe(frame: pd.DataFrame, key: str) -> pd.Series:
    if key in frame.index:
        selection = frame.loc[key]
    elif key in frame.columns:
        selection = frame[key]
    else:
        return pd.Series([np.nan])

    if isinstance(selection, pd.DataFrame):
        flattened = selection.stack(dropna=False)
        return flattened if isinstance(flattened, pd.Series) else pd.Series([np.nan])
    if isinstance(selection, pd.Series):
        return selection
    return pd.Series(selection)


def _extract_series(parsed_data: Any, key: str) -> pd.Series:
    if isinstance(parsed_data, pd.DataFrame):
        return _series_from_dataframe(parsed_data, key)
    if isinstance(parsed_data, Mapping):
        return _series_from_mapping(parsed_data, key)
    return pd.Series([np.nan])


def _safe_first(series: pd.Series) -> Any:
    if series.empty:
        return np.nan
    return series.iloc[0]


def _compute_yoy(series: pd.Series) -> float:
    if series.size < 2:
        return np.nan
    current = series.iloc[0]
    previous = series.iloc[1]
    if pd.isna(previous) or previous == 0:
        return np.nan
    if pd.isna(current):
        return np.nan
    return (current - previous) / abs(previous)


def _negative_or_invalid(value: Any) -> bool:
    if pd.isna(value):
        return True
    if isinstance(value, (int, float, np.number)):
        return bool(value < 0 or np.isinf(value))
    return False


def _yoy_flag(yoy_value: Any) -> bool:
    if pd.isna(yoy_value):
        return True
    if not _is_number(yoy_value):
        return True
    return abs(float(yoy_value)) > 5.0


def _derived_flag(value: Any, denominator_zero: bool = False) -> bool:
    if denominator_zero:
        return True
    return _negative_or_invalid(value)


def _compute_free_cash_flow(
    cfo_series: pd.Series, capex_series: pd.Series
) -> tuple[float, bool]:
    cfo = _safe_first(cfo_series)
    capex = _safe_first(capex_series)
    if pd.isna(cfo) or pd.isna(capex):
        return np.nan, True
    return cfo - capex, _negative_or_invalid(cfo - capex)


def _compute_dsri(
    receivable_series: pd.Series, revenue_series: pd.Series
) -> tuple[float, bool, bool]:
    receivable = _safe_first(receivable_series)
    revenue = _safe_first(revenue_series)
    denominator_zero = False
    if pd.isna(receivable) or pd.isna(revenue):
        return np.nan, True, denominator_zero
    if revenue == 0:
        denominator_zero = True
        return np.nan, True, denominator_zero
    ratio = receivable / revenue
    return ratio, _negative_or_invalid(ratio), denominator_zero


def compute_financial_indicators(parsed_data: Any) -> Dict[str, Any]:
    base_series: Dict[str, pd.Series] = {
        key: _extract_series(parsed_data, key) for key in BASE_METRIC_KEYS
    }
    supplemental_series: Dict[str, pd.Series] = {
        key: _extract_series(parsed_data, key) for key in DERIVED_REQUIRED_KEYS
    }

    base_metrics: Dict[str, Any] = {k: _safe_first(v) for k, v in base_series.items()}

    yoy_changes: Dict[str, Any] = {
        key: _compute_yoy(series) for key, series in base_series.items()
    }

    free_cash_flow, fcf_flag = _compute_free_cash_flow(
        base_series["经营活动产生的现金流量净额"], supplemental_series["资本性支出"]
    )

    dsri_value, dsri_flag, dsri_zero = _compute_dsri(
        supplemental_series["应收账款"], base_series["营业收入"]
    )

    yoy_flags: Dict[str, bool] = {k: _yoy_flag(v) for k, v in yoy_changes.items()}

    derived_flags: Dict[str, bool] = {
        "自由现金流": _derived_flag(free_cash_flow, fcf_flag),
        "DSRI": _derived_flag(dsri_value, dsri_zero),
    }
    derived_flags["同比变动率"] = any(yoy_flags.values())

    derived_metrics: Dict[str, Any] = {
        "同比变动率": yoy_changes,
        "自由现金流": free_cash_flow,
        "DSRI": dsri_value,
    }

    flags: Dict[str, Any] = {
        "同比异常": yoy_flags,
        "派生异常": derived_flags,
    }

    return {
        "基础指标": base_metrics,
        "派生指标": derived_metrics,
        "异常标记": flags,
    }
