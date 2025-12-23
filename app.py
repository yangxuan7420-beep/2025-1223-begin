from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, Iterable, List

import streamlit as st

from logic.metrics import METRIC_CATALOG, compute_financial_indicators, suggest_candidates
from parser.wind_parser import WindParser
from view.display import render_charts, render_overview, render_risk_status
from view.validators import validate_logic_payload
from views.research_signals import render_research_signals


st.set_page_config(page_title="VisionFinance", page_icon=":bar_chart:", layout="wide")

st.title("VisionFinance")
st.caption("基于 Wind 数据的二次分析展示")

if "current_view" not in st.session_state:
    st.session_state.current_view = "analysis"

st.session_state.setdefault("logic_output", None)


def _save_uploaded_files(uploaded_files: Iterable, temp_dir: str) -> List[Path]:
    paths: List[Path] = []
    for file in uploaded_files:
        path = Path(temp_dir) / Path(file.name).name
        path.write_bytes(file.getbuffer())
        paths.append(path)
    return paths


def _parse_uploaded_files(uploaded_files: List, merge_axis: str) -> dict:
    if not uploaded_files:
        raise ValueError("未检测到上传文件，无法展示。")

    parser = WindParser()
    with tempfile.TemporaryDirectory() as temp_dir:
        paths = _save_uploaded_files(uploaded_files, temp_dir)
        parsed = parser.parse(paths, merge_axis=merge_axis)

    combined = parsed.get("combined")
    if combined is None or getattr(combined, "empty", False):
        raise ValueError("解析结果缺失或为空，无法展示。")

    return parsed


def _format_candidate_option(candidate: Dict[str, object] | None) -> str:
    if candidate is None:
        return "无（不使用该指标）"

    table_label = candidate.get("table_label") or Path(candidate.get("table_name", "")).name
    raw_name = candidate.get("raw_name", "")
    score = candidate.get("score", 0)
    latest_value = candidate.get("latest_value", None)
    if latest_value is None:
        latest_display = "NaN"
    elif isinstance(latest_value, (int, float)):
        latest_display = f"{latest_value:,.2f}"
    else:
        latest_display = str(latest_value)

    return f"{raw_name} | 表：{table_label} | 相似度：{score} | 最新值：{latest_display}"


def _render_mapping_selector(
    candidates: Dict[str, List[Dict[str, object]]], prior_selection: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, object] | None]:
    st.header("口径选择区块")
    st.caption("请选择每个指标对应的科目，候选展示相似度、来源表与最新值。")
    selections: Dict[str, Dict[str, object] | None] = {}

    for metric in METRIC_CATALOG.keys():
        metric_candidates = candidates.get(metric, [])
        options: List[Dict[str, object] | None] = [None] + metric_candidates

        default_index = 0
        previous = prior_selection.get(metric)
        if previous:
            for idx, item in enumerate(metric_candidates, start=1):
                if (
                    item.get("raw_name") == previous.get("raw_name")
                    and item.get("table_name") == previous.get("table_name")
                ):
                    default_index = idx
                    break

        selection = st.selectbox(
            f"{metric} 候选", options=options, index=default_index, format_func=_format_candidate_option
        )
        selections[metric] = selection

    return selections


def main() -> None:
    if st.button("查看研究提示"):
        st.session_state.current_view = "research"

    if st.session_state.current_view == "research":
        if st.button("返回分析页面"):
            st.session_state.current_view = "analysis"

        logic_output = st.session_state.get("logic_output")
        if logic_output is None:
            st.info("请先完成财务分析，再查看研究提示。")
        else:
            render_research_signals(logic_output)
        return

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

    st.session_state.setdefault("selected_mapping", {})
    st.session_state.setdefault("mapping_confirmed", False)
    st.session_state.setdefault("upload_token", "")

    upload_token = "|".join(sorted(f"{file.name}:{getattr(file, 'size', 0)}" for file in uploaded_files))
    if upload_token and upload_token != st.session_state["upload_token"]:
        st.session_state["selected_mapping"] = {}
        st.session_state["mapping_confirmed"] = False
        st.session_state["upload_token"] = upload_token
        st.session_state["logic_output"] = None
        st.session_state.current_view = "analysis"

    try:
        parsed = _parse_uploaded_files(uploaded_files, merge_axis)
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        st.stop()

    tables = parsed.get("tables", {})
    candidates = suggest_candidates(tables)
    selection = _render_mapping_selector(candidates, st.session_state.get("selected_mapping", {}))

    if selection != st.session_state.get("selected_mapping", {}):
        st.session_state["selected_mapping"] = selection
        st.session_state["mapping_confirmed"] = False
        st.session_state["logic_output"] = None

    if st.button("确认口径并分析"):
        st.session_state["mapping_confirmed"] = True
        st.success("已确认口径，将基于选择进行分析。")

    if not st.session_state.get("mapping_confirmed", False):
        st.info("口径未确认，当前仅展示候选结果。")
        return

    try:
        indicators = compute_financial_indicators(parsed, st.session_state["selected_mapping"])
        validate_logic_payload(indicators)
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        st.stop()

    st.session_state["logic_output"] = indicators

    base_metrics = indicators["基础指标"]
    derived_metrics = indicators["派生指标"]
    risk_matrix = indicators["风险矩阵"]
    render_overview(base_metrics, derived_metrics)
    render_charts(derived_metrics.get("同比趋势", {}))
    render_risk_status(risk_matrix)


if __name__ == "__main__":
    main()
