from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from rapidfuzz import fuzz


YOY_ANOMALY_THRESHOLD = 3.0
DSRI_THRESHOLD = 1.4


METRIC_CATALOG: Dict[str, Dict[str, Any]] = {
    "营业收入": {
        "aliases": ("营业收入", "营业总收入", "主营业务收入"),
        "preferred_tables": ("利润表", "损益", "收入"),
    },
    "归母净利润": {
        "aliases": ("归母净利润", "归属于母公司股东的净利润"),
        "preferred_tables": ("利润表", "损益"),
    },
    "净利润": {
        "aliases": ("净利润",),
        "preferred_tables": ("利润表", "损益"),
    },
    "经营活动产生的现金流量净额": {
        "aliases": ("经营活动产生的现金流量净额", "经营活动现金流净额", "经营现金流"),
        "preferred_tables": ("现金流量表", "现金流"),
    },
    "货币资金": {
        "aliases": ("货币资金", "现金及现金等价物余额"),
        "preferred_tables": ("资产负债表", "资产", "现金"),
    },
    "有息负债": {
        "aliases": ("有息负债", "带息负债", "带息债务"),
        "preferred_tables": ("资产负债表", "负债"),
    },
    "资本性支出": {
        "aliases": (
            "资本性支出",
            "购建固定资产、无形资产和其他长期资产支付的现金",
            "购建固定资产无形资产和其他长期资产所支付的现金",
        ),
        "preferred_tables": ("现金流量表", "投资活动", "现金流"),
    },
    "应收账款": {
        "aliases": ("应收账款", "应收账款净额", "应收账款及票据"),
        "preferred_tables": ("资产负债表", "资产"),
    },
}

METRIC_ALIASES: Dict[str, tuple[str, ...]] = {
    key: tuple(value.get("aliases", (key,))) for key, value in METRIC_CATALOG.items()
}

BASE_METRIC_KEYS: tuple[str, ...] = (
    "营业收入",
    "归母净利润",
    "净利润",
    "经营活动产生的现金流量净额",
    "货币资金",
    "有息负债",
)

DERIVED_REQUIRED_KEYS: tuple[str, ...] = ("资本性支出", "应收账款")

PARENT_PROFIT_ALIASES: tuple[str, ...] = METRIC_ALIASES["归母净利润"]

NET_PROFIT_ALIASES: tuple[str, ...] = METRIC_ALIASES["净利润"]


def _detect_table_type(table_name: str) -> str:
    lowercase = table_name.lower()
    if "现金流" in lowercase:
        return "现金流量表"
    if "利润" in lowercase or "损益" in lowercase:
        return "利润表"
    if "负债" in lowercase:
        return "资产负债表"
    if "资产" in lowercase:
        return "资产负债表"
    return "其他"


def _score_table_preference(
    table_name: str, preferred_tables: Sequence[str] | None, base_score: float
) -> float:
    if not preferred_tables:
        return base_score

    table_type = _detect_table_type(table_name)
    for preferred in preferred_tables:
        if preferred and preferred in table_type:
            return base_score + 8.0
        if preferred and preferred in table_name:
            return base_score + 4.0
    return base_score


def _normalize_data_sources(
    parsed_data: Any,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame | None]:
    if isinstance(parsed_data, Mapping):
        tables = {
            key: value for key, value in parsed_data.get("tables", {}).items() if isinstance(value, pd.DataFrame)
        }
        combined = parsed_data.get("combined")
        combined_df = combined if isinstance(combined, pd.DataFrame) else None
        return tables, combined_df

    combined_df = parsed_data if isinstance(parsed_data, pd.DataFrame) else None
    return {}, combined_df


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


def _select_series(parsed_data: Any, key: str, selected_mapping: Mapping[str, Any]) -> pd.Series:
    tables, combined = _normalize_data_sources(parsed_data)
    mapping_candidate = selected_mapping.get(key)

    if mapping_candidate:
        table_name = mapping_candidate.get("table_name")
        raw_name = mapping_candidate.get("raw_name")
        table = tables.get(table_name) if table_name else None
        if isinstance(table, pd.DataFrame) and raw_name:
            series = _series_from_dataframe(table, raw_name)
            if not series.isna().all():
                return series

        if isinstance(combined, pd.DataFrame) and raw_name:
            series = _series_from_dataframe(combined, raw_name)
            if not series.isna().all():
                return series

    data_source: Any = tables if tables else combined
    return _extract_series_with_aliases(data_source, key)


def _safe_first(series: pd.Series) -> Any:
    if series.empty:
        return np.nan
    ordered = _order_series_by_time(series)
    non_na = ordered.dropna()
    if non_na.empty:
        return np.nan
    return non_na.iloc[0]


def _latest_value_from_series(series: pd.Series) -> Any:
    if series.isna().all():
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


def _format_percent(value: float) -> str:
    return f"{value * 100:.0f}%"


def _evaluate_yoy_anomaly(series: pd.Series) -> dict[str, Any]:
    if series.size < 2:
        return {"value": None, "reasons": ["insufficient_periods"]}

    ordered = _order_series_by_time(series, ascending=True)
    years = _parse_year_like_values(ordered.index)

    valid_found = False
    anomaly_reasons: list[str] = []
    na_reasons: list[str] = []

    for idx in range(1, len(ordered)):
        current_value = ordered.iloc[idx]
        previous_value = ordered.iloc[idx - 1]
        current_year = years[idx]
        previous_year = years[idx - 1]

        if not _years_are_consecutive(previous_year, current_year):
            continue

        if pd.isna(previous_value) or pd.isna(current_value):
            na_reasons.append("missing_metric")
            continue

        if previous_value == 0:
            na_reasons.append("division_by_zero")
            continue

        yoy = (current_value - previous_value) / abs(previous_value)
        valid_found = True
        if abs(float(yoy)) > YOY_ANOMALY_THRESHOLD:
            label = ordered.index[idx]
            anomaly_reasons.append(f"{label} YoY={_format_percent(yoy)} > {_format_percent(YOY_ANOMALY_THRESHOLD)}")

    if valid_found:
        return {"value": bool(anomaly_reasons), "reasons": anomaly_reasons}

    fallback_reasons = na_reasons or ["missing_metric"]
    return {"value": None, "reasons": list(dict.fromkeys(fallback_reasons))}


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


def _compute_free_cash_flow(
    cfo_series: pd.Series, capex_series: pd.Series
) -> tuple[float, bool, pd.Series, str | None]:
    if capex_series.isna().all():
        return np.nan, True, pd.Series(dtype=float), "资本性支出数据缺失或未命中，无法计算自由现金流"

    paired = pd.DataFrame({"cfo": cfo_series, "capex": capex_series})
    ordered_index = _order_index_by_time(paired.index, ascending=True)
    paired = paired.reindex(ordered_index)

    fcf_series = paired["cfo"] - paired["capex"]
    valid_mask = paired[["cfo", "capex"]].notna().all(axis=1)
    fcf_series.loc[~valid_mask] = np.nan

    latest_fcf = _safe_first(_order_series_by_time(fcf_series))
    return latest_fcf, _negative_or_invalid(latest_fcf), fcf_series, None


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


def _evaluate_fcf_anomaly(
    fcf_value: Any, fcf_series: pd.Series, intrinsic_reason: str | None
) -> dict[str, Any]:
    reasons: list[str] = []

    if intrinsic_reason:
        reasons.append(f"missing_metric: {intrinsic_reason}")

    if fcf_series.empty or fcf_series.dropna().empty:
        reasons.append("insufficient_periods")

    if reasons:
        return {"value": None, "reasons": list(dict.fromkeys(reasons))}

    if pd.isna(fcf_value):
        return {"value": None, "reasons": ["missing_metric"]}

    if float(fcf_value) < 0:
        return {"value": True, "reasons": [f"FCF={fcf_value:,.2f} < 0"]}

    return {"value": False, "reasons": []}


def _evaluate_dsri_anomaly(dsri_value: Any, denominator_zero: bool) -> dict[str, Any]:
    if denominator_zero:
        return {"value": None, "reasons": ["division_by_zero"]}

    if pd.isna(dsri_value):
        return {"value": None, "reasons": ["missing_metric"]}

    if float(dsri_value) > DSRI_THRESHOLD:
        return {"value": True, "reasons": [f"DSRI={dsri_value:.2f} > {DSRI_THRESHOLD:.2f}"]}

    return {"value": False, "reasons": []}


def suggest_candidates(tables: Mapping[str, pd.DataFrame]) -> Dict[str, list[Dict[str, Any]]]:
    results: Dict[str, list[Dict[str, Any]]] = {}
    if not tables:
        return {metric: [] for metric in METRIC_CATALOG.keys()}

    for metric, config in METRIC_CATALOG.items():
        aliases = config.get("aliases", (metric,))
        preferred = config.get("preferred_tables", ())
        metric_candidates: list[Dict[str, Any]] = []

        for table_name, table in tables.items():
            if not isinstance(table, pd.DataFrame):
                continue
            table_label = Path(table_name).name
            for raw_name in table.index.astype(str):
                base_score = max(fuzz.partial_ratio(raw_name, alias) for alias in aliases)
                scored = _score_table_preference(table_name, preferred, float(base_score))
                series = _series_from_dataframe(table, raw_name)
                metric_candidates.append(
                    {
                        "raw_name": raw_name,
                        "table_name": table_name,
                        "table_label": table_label,
                        "score": round(scored, 2),
                        "latest_value": _latest_value_from_series(series),
                    }
                )

        sorted_candidates = sorted(
            metric_candidates, key=lambda item: item["score"], reverse=True
        )
        results[metric] = sorted_candidates[:5]

    return results


def compute_financial_indicators(
    parsed_data: Any, selected_mapping: Mapping[str, Any] | None = None
) -> Dict[str, Any]:
    mapping = dict(selected_mapping or {})

    base_series: Dict[str, pd.Series] = {
        key: _select_series(parsed_data, key, mapping) for key in BASE_METRIC_KEYS
    }
    supplemental_series: Dict[str, pd.Series] = {
        key: _select_series(parsed_data, key, mapping) for key in DERIVED_REQUIRED_KEYS
    }

    parent_profit_series = _select_series(parsed_data, "归母净利润", mapping)
    net_profit_series = _select_series(parsed_data, "净利润", mapping)

    def _select_profit(series: pd.Series, fallback_aliases: Sequence[str]) -> tuple[pd.Series, str]:
        if not series.isna().all():
            return series, fallback_aliases[0]
        candidates = [_select_series(parsed_data, alias, {}) for alias in fallback_aliases]
        for candidate_series, alias in zip(candidates, fallback_aliases):
            if not candidate_series.isna().all():
                return candidate_series, alias
        return pd.Series([np.nan]), ""

    parent_profit_series, parent_profit_key = _select_profit(
        parent_profit_series, PARENT_PROFIT_ALIASES
    )
    net_profit_series, net_profit_key = _select_profit(net_profit_series, NET_PROFIT_ALIASES)

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

    fcf_reason: str | None = None
    missing_parts: list[str] = []
    if mapping:
        if not mapping.get("经营活动产生的现金流量净额"):
            missing_parts.append("经营活动产生的现金流量净额未选择")
        if not mapping.get("资本性支出"):
            missing_parts.append("资本性支出未选择")

    if missing_parts:
        free_cash_flow = np.nan
        free_cash_flow_series = pd.Series(dtype=float)
        fcf_reason = "；".join(missing_parts)
    else:
        (
            free_cash_flow,
            _,
            free_cash_flow_series,
            intrinsic_reason,
        ) = _compute_free_cash_flow(
            base_series["经营活动产生的现金流量净额"], supplemental_series["资本性支出"]
        )
        fcf_reason = intrinsic_reason

    dsri_value, _, dsri_zero = _compute_dsri(supplemental_series["应收账款"], base_series["营业收入"])

    revenue_yoy_series = _compute_yoy_series(base_series["营业收入"])
    profit_yoy_series = _compute_yoy_series(profit_series)
    cfo_yoy_series = _compute_yoy_series(base_series["经营活动产生的现金流量净额"])
    cash_yoy_series = _compute_yoy_series(base_series["货币资金"])
    debt_yoy_series = _compute_yoy_series(base_series["有息负债"])

    def _not_applicable(reason: str) -> dict[str, Any]:
        return {"value": None, "reasons": [reason]}

    risk_matrix: Dict[str, Dict[str, Any]] = {
        "revenue": {
            "display_name": "营业收入",
            "yoy_anomaly": _evaluate_yoy_anomaly(base_series["营业收入"]),
            "derived_anomaly": _not_applicable("not_applicable"),
        },
        "profit": {
            "display_name": f"利润（{profit_mode}）",
            "yoy_anomaly": (
                _evaluate_yoy_anomaly(profit_series)
                if profit_mode != "缺失"
                else {"value": None, "reasons": ["not_in_selected_caliber"]}
            ),
            "derived_anomaly": _not_applicable("not_applicable"),
        },
        "operating_cf": {
            "display_name": "经营现金流",
            "yoy_anomaly": _evaluate_yoy_anomaly(base_series["经营活动产生的现金流量净额"]),
            "derived_anomaly": _not_applicable("not_applicable"),
        },
        "cash": {
            "display_name": "货币资金",
            "yoy_anomaly": _evaluate_yoy_anomaly(base_series["货币资金"]),
            "derived_anomaly": _not_applicable("not_applicable"),
        },
        "interest_bearing_debt": {
            "display_name": "有息负债",
            "yoy_anomaly": _evaluate_yoy_anomaly(base_series["有息负债"]),
            "derived_anomaly": _not_applicable("not_applicable"),
        },
        "free_cash_flow": {
            "display_name": "自由现金流",
            "yoy_anomaly": _not_applicable("not_applicable"),
            "derived_anomaly": _evaluate_fcf_anomaly(
                free_cash_flow, free_cash_flow_series, fcf_reason
            ),
        },
        "dsri": {
            "display_name": "DSRI",
            "yoy_anomaly": _not_applicable("not_applicable"),
        "derived_anomaly": _evaluate_dsri_anomaly(dsri_value, dsri_zero),
        },
    }

    def _normalize_flag(value: Any) -> bool:
        return bool(value) if value is not None else False

    yoy_flags: Dict[str, bool] = {
        item["display_name"]: _normalize_flag(item["yoy_anomaly"].get("value"))
        for item in risk_matrix.values()
        if isinstance(item, dict) and "yoy_anomaly" in item
    }

    derived_flags: Dict[str, bool] = {
        "自由现金流": _normalize_flag(risk_matrix["free_cash_flow"]["derived_anomaly"].get("value")),
        "DSRI": _normalize_flag(risk_matrix["dsri"]["derived_anomaly"].get("value")),
    }
    derived_flags["同比变动率"] = any(_normalize_flag(item["yoy_anomaly"].get("value")) for item in risk_matrix.values())

    derived_metrics: Dict[str, Any] = {
        "同比变动率": yoy_changes,
        "同比趋势": {
            "营业收入": revenue_yoy_series,
            "利润": profit_yoy_series,
            "经营现金流": cfo_yoy_series,
            "货币资金": cash_yoy_series,
            "有息负债": debt_yoy_series,
        },
        "自由现金流": free_cash_flow,
        "自由现金流缺失原因": fcf_reason,
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
        "风险矩阵": risk_matrix,
        "口径信息": {
            "profit_mode": profit_mode,
            "profit_source_key": profit_source_key,
            "selected_mapping": mapping,
        },
    }
