from __future__ import annotations

from typing import Any, Mapping

import pandas as pd


REQUIRED_KEYS = {"基础指标", "派生指标", "异常标记"}
REQUIRED_DERIVED = {"同比变动率", "自由现金流", "DSRI"}
REQUIRED_FLAGS = {"同比异常", "派生异常"}


def _ensure_keys(data: Mapping[str, Any], required: set[str], context: str) -> None:
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"{context}缺失字段: {', '.join(sorted(missing))}")


def _ensure_not_empty(value: Any, context: str) -> None:
    if value is None:
        raise ValueError(f"{context}结果缺失，无法展示。")
    if isinstance(value, (dict, list, tuple, set)) and len(value) == 0:
        raise ValueError(f"{context}结果为空，无法展示。")
    if isinstance(value, pd.DataFrame) and value.empty:
        raise ValueError(f"{context}结果为空，无法展示。")


def validate_logic_payload(payload: Mapping[str, Any]) -> None:
    _ensure_keys(payload, REQUIRED_KEYS, "逻辑输出")

    base = payload.get("基础指标")
    derived = payload.get("派生指标")
    flags = payload.get("异常标记")

    _ensure_not_empty(base, "基础指标")
    _ensure_not_empty(derived, "派生指标")
    _ensure_not_empty(flags, "异常标记")

    if not isinstance(derived, Mapping) or not isinstance(flags, Mapping):
        raise ValueError("指标或标记格式不符合约定。")

    _ensure_keys(derived, REQUIRED_DERIVED, "派生指标")
    _ensure_keys(flags, REQUIRED_FLAGS, "异常标记")
