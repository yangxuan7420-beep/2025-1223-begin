from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


BASE_METRIC_KEYS: tuple[str, ...] = (
    "营业收入",
    "归母净利润",
    "净利润",
    "经营活动产生的现金流量净额",
    "货币资金",
    "有息负债",
)

DERIVED_REQUIRED_KEYS: tuple[str, ...] = ("资本性支出", "应收账款")

METRIC_ALIASES: Dict[str, tuple[str, ...]] = {
    "营业收入": ("营业收入", "营业总收入", "主营业务收入"),
    "归母净利润": ("归母净利润", "归属于母公司股东的净利润"),
    "净利润": ("净利润",),
    "资本性支出": (
        "资本性支出",
        "购建固定资产、无形资产和其他长期资产支付的现金",
        "购建固定资产无形资产和其他长期资产所支付的现金",
    ),
    "应收账款": ("应收账款", "应收账款净额", "应收账款及票据"),
}

PARENT_PROFIT_ALIASES: tuple[str, ...] = (
    "归属于母公司股东的净利润",
    "归母净利润",
)

NET_PROFIT_ALIASES: tuple[str, ...] = ("净利润",)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.number)) and not isinstance(value, bool)


def _order_index_by_time(index: pd.Index, *, ascending: bool = False) -> pd.Index:
    if index.empty:
        return index

    datetime_index = pd.to_datetime(index, errors="coerce")
    if not pd.isna(datetime_index).all():
        order = np.argsort(datetime_index)
        if not ascending:
            order = order[::-1]
        return index.take(order)

    numeric_index = pd.to_numeric(index, errors="coerce")
    if not pd.isna(numeric_index).all():
        order = np.argsort(numeric_index)
        if not ascending:
            order = order[::-1]
        return index.take(order)

    return index


def _order_series_by_time(series: pd.Series, *, ascending: bool = False) -> pd.Series:
    if series.empty:
        return series

    ordered_index = _order_index_by_time(series.index, ascending=ascending)
    return series.reindex(ordered_index)


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
    # 1. 精确 index 命中（保留原逻辑）
    if key in frame.index:
        selection = frame.loc[key]

    # 2. 模糊 index 命中（新增）
    elif isinstance(frame.index, pd.Index):
        mask = frame.index.astype(str).str.contains(key, regex=False, na=False)
        if mask.any():
            selection = frame.loc[mask]

        # 3. column 命中（保留）
        elif key in frame.columns:
            selection = frame[key]
        else:
            return pd.Series([np.nan])
    else:
        return pd.Series([np.nan])

    # 展平逻辑（保持你原有设计）
    if isinstance(selection, pd.DataFrame):
        ordered_columns = _order_index_by_time(selection.columns)
        flattened = selection.loc[:, ordered_columns].T.stack(dropna=False)
        return flattened if isinstance(flattened, pd.Series) else pd.Series([np.nan])
    if isinstance(selection, pd.Series):
        ordered_index = _order_index_by_time(selection.index)
        return selection.reindex(ordered_index)
    return pd.Series(selection)


def _extract_series(parsed_data: Any, key: str) -> pd.Series:
    # 1. 单 DataFrame（保留）
    if isinstance(parsed_data, pd.DataFrame):
        return _series_from_dataframe(parsed_data, key)

    # 2. 多表 dict：逐表尝试（核心修复）
    if isinstance(parsed_data, Mapping):
        for value in parsed_data.values():
            if isinstance(value, pd.DataFrame):
                series = _series_from_dataframe(value, key)
                # 只要不是全 NaN，就认为命中
                if not series.isna().all():
                    return series
        # 所有表都没命中
        return pd.Series([np.nan])

    return pd.Series([np.nan])


def _extract_series_with_aliases(parsed_data: Any, key: str) -> pd.Series:
    candidates = METRIC_ALIASES.get(key, (key,))
    for candidate in candidates:
        series = _extract_series(parsed_data, candidate)
        if not series.isna().all():
            return series
    return _extract_series(parsed_data, key)


def _safe_first(series: pd.Series) -> Any:
    if series.empty:
        return np.nan
    ordered = _order_series_by_time(series)
    non_na = ordered.dropna()
    if non_na.empty:
        return np.nan
    return non_na.iloc[0]


def _parse_year_like_values(index: pd.Index) -> np.ndarray:
    datetime_index = pd.to_datetime(index, errors="coerce")
    if not pd.isna(datetime_index).all():
        years = pd.Series(datetime_index).dt.year
        return years.to_numpy()

    numeric_index = pd.to_numeric(index, errors="coerce")
    if not pd.isna(numeric_index).all():
        return numeric_index.to_numpy()

    return np.full(len(index), np.nan)


def _years_are_consecutive(previous_year: Any, current_year: Any) -> bool:
    if pd.isna(previous_year) or pd.isna(current_year):
        return False
    try:
        return int(current_year) - int(previous_year) == 1
    except (TypeError, ValueError, OverflowError):
        return False


def _compute_yoy_series(series: pd.Series) -> pd.Series:
    if series.size < 2:
        return pd.Series(dtype=float)

    ordered = _order_series_by_time(series, ascending=True)
    years = _parse_year_like_values(ordered.index)

    yoy_values: list[float] = []
    yoy_index: list[Any] = []

    for idx in range(1, len(ordered)):
        current_value = ordered.iloc[idx]
        previous_value = ordered.iloc[idx - 1]
        current_year = years[idx]
        previous_year = years[idx - 1]

        yoy = np.nan
        if _years_are_consecutive(previous_year, current_year):
            if not pd.isna(previous_value) and previous_value != 0 and not pd.isna(current_value):
                yoy = (current_value - previous_value) / abs(previous_value)

        yoy_values.append(yoy)
        yoy_index.append(ordered.index[idx])

    return pd.Series(yoy_values, index=yoy_index)


def _compute_yoy(series: pd.Series) -> float:
    yoy_series = _compute_yoy_series(series)
    if yoy_series.empty:
        return np.nan
    return _safe_first(_order_series_by_time(yoy_series))


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
) -> tuple[float, bool, pd.Series]:
    if capex_series.isna().all():
        return np.nan, True, pd.Series(dtype=float)

    paired = pd.DataFrame({"cfo": cfo_series, "capex": capex_series})
    ordered_index = _order_index_by_time(paired.index, ascending=True)
    paired = paired.reindex(ordered_index)

    fcf_series = paired["cfo"] - paired["capex"]
    valid_mask = paired[["cfo", "capex"]].notna().all(axis=1)
    fcf_series.loc[~valid_mask] = np.nan

    latest_fcf = _safe_first(_order_series_by_time(fcf_series))
    return latest_fcf, _negative_or_invalid(latest_fcf), fcf_series


def _compute_dsri(
    receivable_series: pd.Series, revenue_series: pd.Series
) -> tuple[float, bool, bool]:
    paired = pd.DataFrame({"receivable": receivable_series, "revenue": revenue_series})
    ordered_index = _order_index_by_time(paired.index)
    paired = paired.reindex(ordered_index)

    selected = paired.iloc[:2]
    revenue_zero = (selected["revenue"] == 0).any()

    if selected.shape[0] < 2:
        return np.nan, True, revenue_zero

    if (
        selected["receivable"].isna().any()
        or selected["revenue"].isna().any()
        or revenue_zero
    ):
        return np.nan, True, revenue_zero

    ratios = selected["receivable"] / selected["revenue"]
    ratio_t = ratios.iloc[0]
    ratio_previous = ratios.iloc[1]

    if ratio_previous == 0:
        return np.nan, True, True

    dsri = ratio_t / ratio_previous
    return dsri, _negative_or_invalid(dsri), False


def compute_financial_indicators(parsed_data: Any) -> Dict[str, Any]:
    base_series: Dict[str, pd.Series] = {
        key: _extract_series_with_aliases(parsed_data, key) for key in BASE_METRIC_KEYS
    }
    supplemental_series: Dict[str, pd.Series] = {
        key: _extract_series_with_aliases(parsed_data, key) for key in DERIVED_REQUIRED_KEYS
    }

    parent_profit_series_candidates = [
        _extract_series(parsed_data, alias) for alias in PARENT_PROFIT_ALIASES
    ]
    net_profit_series_candidates = [
        _extract_series(parsed_data, alias) for alias in NET_PROFIT_ALIASES
    ]

    def _select_profit(
        candidates: Sequence[pd.Series], aliases: Sequence[str]
    ) -> tuple[pd.Series, str]:
        fallback_series = candidates[0] if candidates else pd.Series([np.nan])
        for series, alias in zip(candidates, aliases):
            if not series.isna().all():
                return series, alias
        return fallback_series, ""

    parent_profit_series, parent_profit_key = _select_profit(
        parent_profit_series_candidates, PARENT_PROFIT_ALIASES
    )
    net_profit_series, net_profit_key = _select_profit(
        net_profit_series_candidates, NET_PROFIT_ALIASES
    )

    if not parent_profit_series.isna().all():
        profit_series = parent_profit_series
        profit_mode = "归母净利润"
        profit_source_key = parent_profit_key or PARENT_PROFIT_ALIASES[0]
    elif not net_profit_series.isna().all():
        profit_series = net_profit_series
        profit_mode = "净利润"
        profit_source_key = net_profit_key or NET_PROFIT_ALIASES[0]
    else:
        profit_series = pd.Series([np.nan])
        profit_mode = "缺失"
        profit_source_key = ""

    base_metrics: Dict[str, Any] = {k: _safe_first(v) for k, v in base_series.items()}

    yoy_changes: Dict[str, Any] = {
        key: _compute_yoy(series) for key, series in base_series.items()
    }

    free_cash_flow, fcf_flag, free_cash_flow_series = _compute_free_cash_flow(
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

    revenue_yoy_series = _compute_yoy_series(base_series["营业收入"])
    profit_yoy_series = _compute_yoy_series(profit_series)

    derived_metrics: Dict[str, Any] = {
        "同比变动率": yoy_changes,
        "同比趋势": {
            "营业收入": revenue_yoy_series,
            "利润": profit_yoy_series,
        },
        "自由现金流": free_cash_flow,
        "自由现金流水平": free_cash_flow_series,
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
        "口径信息": {
            "profit_mode": profit_mode,
            "profit_source_key": profit_source_key,
        },
    }
