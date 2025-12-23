from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Union

import pandas as pd


class WindParser:
    def __init__(self, header_scan_rows: int = 8) -> None:
        self.header_scan_rows = header_scan_rows

    def parse(
        self, paths: Iterable[Union[str, Path]], merge_axis: str = "vertical"
    ) -> Dict[str, pd.DataFrame]:
        if isinstance(paths, (str, Path)):
            normalized_paths = [Path(paths)]
        else:
            normalized_paths = [Path(item) for item in paths]

        axis = merge_axis.lower()
        frames: List[pd.DataFrame] = []
        table_map: Dict[str, pd.DataFrame] = {}

        for path in normalized_paths:
            cleaned = self._load_and_clean(path)
            table_map[str(path)] = cleaned
            frames.append(cleaned)

        combined = self._merge_frames(frames, axis)
        return {"combined": combined, "tables": table_map}

    def _load_and_clean(self, path: Path) -> pd.DataFrame:
        raw = self._read_file(path)
        header_row = self._find_header_row(raw)
        structured = self._apply_header(raw, header_row)
        structured = self._clean_index(structured)
        structured = self._clean_values(structured)
        return structured

    def _read_file(self, path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix in {".xls", ".xlsx"}:
            return pd.read_excel(path, header=None, dtype=str)
        return pd.read_csv(path, header=None, dtype=str)

    def _find_header_row(self, df: pd.DataFrame) -> int:
        max_rows = min(self.header_scan_rows, len(df))
        best_row = 0
        best_score = -1
        year_pattern = re.compile(r"(19|20)\d{2}")

        for idx in range(max_rows):
            row = df.iloc[idx]
            score = sum(
                1
                for value in row
                if isinstance(value, str) and bool(year_pattern.search(value))
            )
            if score > best_score:
                best_score = score
                best_row = idx
        return best_row

    def _apply_header(self, df: pd.DataFrame, header_row: int) -> pd.DataFrame:
        header_values = df.iloc[header_row].tolist()
        cleaned_headers = [self._clean_year_label(value) for value in header_values]
        body = df.iloc[header_row + 1 :].reset_index(drop=True)
        body.columns = cleaned_headers
        return body

    def _clean_year_label(self, value: object) -> str:
        text = "" if value is None else str(value).strip()
        match = re.search(r"(19|20)\d{2}", text)
        if match:
            return match.group()[:4]
        return text

    def _clean_index(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        index_col = df.columns[0]
        body = df.set_index(index_col)
        normalized_index = [self._normalize_subject(item) for item in body.index]
        body.index = normalized_index
        body = body.loc[body.index.astype(bool)]
        return body

    def _normalize_subject(self, value: object) -> str:
        text = "" if value is None else str(value)
        text = re.sub(r"\s+", "", text)
        text = re.sub(r"^[0-9]+[\.、)]*", "", text)
        text = re.sub(r"[（(].*?[）)]", "", text)
        text = text.upper()
        cleaned = []
        for ch in text:
            if re.match(r"[\u4e00-\u9fffA-Z0-9]", ch):
                cleaned.append(ch)
        return "".join(cleaned)

    def _clean_values(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.applymap(self._to_float)

    def _to_float(self, value: object) -> float:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return float("nan")

        if isinstance(value, (int, float)):
            return float(value)

        text = str(value).strip()
        if text in {"", "--", "—", "NaN", "nan"}:
            return float("nan")

        negative = False
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1]
            negative = True

        text = text.replace(",", "")
        text = text.replace("，", "")

        try:
            number = float(text)
            return -number if negative else number
        except ValueError:
            return float("nan")

    def _merge_frames(self, frames: List[pd.DataFrame], merge_axis: str) -> pd.DataFrame:
        if not frames:
            return pd.DataFrame()

        if merge_axis == "horizontal":
            return pd.concat(frames, axis=1, sort=False)
        return pd.concat(frames, axis=0, sort=False)
