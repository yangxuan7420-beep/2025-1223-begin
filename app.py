from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterable, List

import streamlit as st

from logic.metrics import compute_financial_indicators
from parser.wind_parser import WindParser
from view.display import render_charts, render_overview, render_risk_status
from view.validators import validate_logic_payload


st.set_page_config(page_title="VisionFinance", page_icon=":bar_chart:", layout="wide")

st.title("VisionFinance")
st.caption("基于 Wind 数据的二次分析展示")


def _save_uploaded_files(uploaded_files: Iterable, temp_dir: str) -> List[Path]:
    paths: List[Path] = []
    for file in uploaded_files:
        path = Path(temp_dir) / Path(file.name).name
        path.write_bytes(file.getbuffer())
        paths.append(path)
    return paths


def _run_pipeline(uploaded_files: List, merge_axis: str) -> dict:
    if not uploaded_files:
        raise ValueError("未检测到上传文件，无法展示。")

    parser = WindParser()
    with tempfile.TemporaryDirectory() as temp_dir:
        paths = _save_uploaded_files(uploaded_files, temp_dir)
        parsed = parser.parse(paths, merge_axis=merge_axis)

    combined = parsed.get("combined")
    if combined is None or getattr(combined, "empty", False):
        raise ValueError("解析结果缺失或为空，无法展示。")

    indicators = compute_financial_indicators(combined)
    validate_logic_payload(indicators)
    return indicators


def main() -> None:
    st.sidebar.header("文件上传区")
    merge_axis = st.sidebar.radio(
        "合并方式",
        options=["vertical", "horizontal"],
        index=0,
        format_func=lambda x: "纵向合并" if x == "vertical" else "横向合并",
    )
    uploaded_files = st.sidebar.file_uploader(
        "上传 Wind 导出文件（支持 CSV、Excel）",
        type=["csv", "xls", "xlsx"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("请上传文件以查看指标。")
        return

    try:
        indicators = _run_pipeline(uploaded_files, merge_axis)
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        st.stop()

    base_metrics = indicators["基础指标"]
    derived_metrics = indicators["派生指标"]
    flags = indicators["异常标记"]

    render_overview(base_metrics, derived_metrics)
    render_charts(derived_metrics.get("同比变动率", {}))
    render_risk_status(flags)


if __name__ == "__main__":
    main()
