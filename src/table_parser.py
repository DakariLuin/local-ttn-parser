import difflib
import math
import re
import statistics
from typing import Dict, List, Tuple, Optional

import pandas as pd


def _bbox_stats(box: List[List[float]]) -> Dict[str, float]:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return {
        "x_min": min(xs),
        "x_max": max(xs),
        "y_min": min(ys),
        "y_max": max(ys),
        "x_center": sum(xs) / 4,
        "y_center": sum(ys) / 4,
        "height": max(1.0, max(ys) - min(ys)),
    }


def _similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(None, left.lower(), right.lower()).ratio()


def _is_number_like(text: str) -> bool:
    normalized = text.strip().replace(" ", "").replace(",", ".")
    if not normalized:
        return False
    allowed = set("0123456789.-")
    return all(ch in allowed for ch in normalized)


def _normalize_for_match(text: str) -> str:
    # Helps fuzzy matching when OCR mixes Cyrillic/Latin letters.
    table = str.maketrans(
        {
            "a": "а",
            "b": "в",
            "c": "с",
            "e": "е",
            "k": "к",
            "m": "м",
            "h": "н",
            "o": "о",
            "p": "р",
            "t": "т",
            "x": "х",
            "y": "у",
        }
    )
    lowered = text.lower().translate(table)
    return "".join(ch for ch in lowered if ch.isalnum() or ch.isspace())


def _cleanup_text_artifacts(text: str) -> str:
    cleaned = (text or "").replace("\n", " ").replace("\t", " ")
    cleaned = re.sub(r"[`´¨^~]+", "", cleaned)
    cleaned = re.sub(r"[|¦]+", " ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def _apply_linguistic_corrections(text: str) -> str:
    if not text:
        return text

    translit_fix = str.maketrans(
        {
            "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М", "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "У",
            "a": "а", "b": "в", "c": "с", "e": "е", "h": "н", "k": "к", "m": "м", "o": "о", "p": "р", "t": "т", "x": "х", "y": "у",
        }
    )
    fixed = text.translate(translit_fix)

    # Common OCR mistakes for this domain.
    replacements = {
        "к ппате": "К оплате",
        "к плате": "К оплате",
        "расценка аз": "Расценка",
        "oтправителы": "Отправитель",
        "отпpавителы": "Отправитель",
        "получателы": "Получатель",
        "получате лы": "Получатель",
        "олуучатель": "Получатель",
    }

    low = fixed.lower()
    for bad, good in replacements.items():
        if bad in low:
            pattern = re.compile(re.escape(bad), flags=re.IGNORECASE)
            fixed = pattern.sub(good, fixed)
            low = fixed.lower()

    fixed = re.sub(r"\b([0-9])[оО]\b", r"\g<1>0", fixed)
    fixed = re.sub(r"\b([0-9])[зЗ]\b", r"\g<1>3", fixed)
    fixed = re.sub(r"\b([0-9])[тТ]\b", r"\g<1>7", fixed)
    fixed = re.sub(r"\s{2,}", " ", fixed).strip()
    return fixed


def _clean_ocr_text(text: str) -> str:
    return _apply_linguistic_corrections(_cleanup_text_artifacts(text))


class FieldExtractor:
    @staticmethod
    def extract_from_free_text(
        ocr_results: list,
        field_config: dict,
        free_text_y_min: float = 0.0,
        table_regions: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        results = {field_name: "" for field_name in field_config.keys()}
        used_indices = set()
        text_items = []
        flattened_keywords = [_normalize_for_match(kw) for kws in field_config.values() for kw in kws]

        def is_inside_table_region(y_center: float) -> bool:
            if not table_regions:
                return False
            return any(region["y_min"] <= y_center <= region["y_max"] for region in table_regions.values())

        for idx, res in enumerate(ocr_results):
            stats = _bbox_stats(res[0])
            text_lower = res[1][0].lower()
            cleaned_text = _clean_ocr_text(res[1][0])
            text_lower = cleaned_text.lower()
            text_norm = _normalize_for_match(text_lower)
            has_field_keyword = any(
                kw and (kw in text_norm or _similarity(kw, text_norm) >= 0.66)
                for kw in flattened_keywords
            )
            in_table_region = is_inside_table_region(stats["y_center"])
            if stats["y_center"] < free_text_y_min and not has_field_keyword:
                continue
            if in_table_region and not has_field_keyword:
                continue
            text_items.append(
                {
                    "idx": idx,
                    "text": cleaned_text,
                    "text_lower": text_lower,
                    "text_norm": text_norm,
                    **stats,
                }
            )

        for field_name, keywords in field_config.items():
            best_anchor = None
            best_score = 0.0
            for item in text_items:
                norm_keywords = [_normalize_for_match(kw) for kw in keywords]
                score = max((_similarity(kw, item["text_norm"]) for kw in norm_keywords), default=0.0)
                if any(kw and kw in item["text_norm"] for kw in norm_keywords):
                    score = max(score, 0.95)
                if score > best_score:
                    best_score = score
                    best_anchor = item

            if not best_anchor or best_score < 0.7:
                continue

            used_indices.add(best_anchor["idx"])
            if ":" in best_anchor["text"]:
                right_part = best_anchor["text"].split(":", 1)[1].strip()
                if right_part:
                    results[field_name] = _clean_ocr_text(right_part)
                    continue

            candidates = []
            for item in text_items:
                if item["idx"] == best_anchor["idx"]:
                    continue
                same_line = abs(item["y_center"] - best_anchor["y_center"]) <= max(14, best_anchor["height"] * 0.8)
                close_below = 0 < (item["y_center"] - best_anchor["y_center"]) <= best_anchor["height"] * 2.2
                right_side = item["x_min"] >= (best_anchor["x_max"] - 8)
                if (same_line or close_below) and right_side:
                    horizontal_gap = max(0, item["x_min"] - best_anchor["x_max"])
                    candidates.append((horizontal_gap, abs(item["y_center"] - best_anchor["y_center"]), item))

            if candidates:
                candidates.sort(key=lambda val: (val[0], val[1]))
                winner = candidates[0][2]
                results[field_name] = _clean_ocr_text(winner["text"])
                used_indices.add(winner["idx"])

        return results, used_indices


class TableParser:
    @staticmethod
    def _collect_header_candidates(indexed_items: List[dict], t_cfg: dict, min_header_score: float) -> List[dict]:
        candidates = []
        for item in indexed_items:
            text_norm = _normalize_for_match(item["text_lower"])
            for col_name, keywords in t_cfg.items():
                score = 0.0
                for kw in keywords:
                    kw_norm = _normalize_for_match(kw)
                    score = max(score, _similarity(kw_norm, text_norm))
                    if kw_norm and kw_norm in text_norm:
                        score = max(score, 0.95)
                if score >= min_header_score:
                    candidates.append({"column": col_name, "score": score, **item})
        return candidates

    @staticmethod
    def detect_table_regions(
        ocr_results: list,
        tables_cfg: dict,
        field_keywords: List[str],
        parser_settings: Optional[dict] = None,
    ) -> Dict[str, Dict[str, float]]:
        parser_settings = parser_settings or {}
        min_header_score = parser_settings.get("MIN_HEADER_SCORE", 0.7)

        indexed_items = []
        for idx, res in enumerate(ocr_results):
            stats = _bbox_stats(res[0])
            indexed_items.append(
                {
                    "idx": idx,
                    "text": res[1][0],
                    "text_lower": _clean_ocr_text(res[1][0]).lower(),
                    **stats,
                }
            )

        table_markers = {}
        band_size = parser_settings.get("HEADER_BAND_SIZE", 35)
        for table_name, t_cfg in tables_cfg.items():
            candidates = TableParser._collect_header_candidates(indexed_items, t_cfg, min_header_score)
            if not candidates:
                continue

            bands = {}
            for cand in candidates:
                band_id = int(cand["y_center"] // max(10, band_size))
                if band_id not in bands:
                    bands[band_id] = {"columns": set(), "score_sum": 0.0, "ys": []}
                bands[band_id]["columns"].add(cand["column"])
                bands[band_id]["score_sum"] += cand["score"]
                bands[band_id]["ys"].append(cand["y_center"])

            best_band_id = max(
                bands.keys(),
                key=lambda b: (len(bands[b]["columns"]), bands[b]["score_sum"]),
            )
            table_markers[table_name] = sum(bands[best_band_id]["ys"]) / len(bands[best_band_id]["ys"])

        sorted_tables = sorted(table_markers.items(), key=lambda pair: pair[1])
        if not sorted_tables:
            return {}

        field_marker_ys = []
        for item in indexed_items:
            if any(kw in item["text_lower"] for kw in field_keywords):
                field_marker_ys.append(item["y_center"])

        regions = {}
        for i, (table_name, y_anchor) in enumerate(sorted_tables):
            y_min = max(0.0, y_anchor - 40)
            next_table_y = sorted_tables[i + 1][1] if i + 1 < len(sorted_tables) else float("inf")
            next_field_y = min((fy for fy in field_marker_ys if fy > y_anchor), default=float("inf"))
            y_max = min(next_table_y - 15, next_field_y - 15)
            if y_max == float("inf"):
                y_max = max(item["y_max"] for item in indexed_items) + 25
            if y_max < y_min:
                y_max = y_min + 50
            regions[table_name] = {"y_min": y_min, "y_max": y_max}

        return regions

    @staticmethod
    def _get_headers(
        ocr_results: list,
        t_cfg: dict,
        region: Dict[str, float],
        min_header_score: float,
    ) -> Dict[str, dict]:
        header_info = {}
        candidates = []
        for idx, res in enumerate(ocr_results):
            stats = _bbox_stats(res[0])
            if not (region["y_min"] <= stats["y_center"] <= region["y_max"]):
                continue
            text = _normalize_for_match(_clean_ocr_text(res[1][0]).lower())
            for col_name, keywords in t_cfg.items():
                score = 0.0
                for kw in keywords:
                    kw_norm = _normalize_for_match(kw)
                    score = max(score, _similarity(kw_norm, text))
                    if kw_norm and kw_norm in text:
                        score = max(score, 0.95)
                if score >= min_header_score:
                    candidates.append({"col": col_name, "score": score, "idx": idx, **stats})

        # Select one best candidate per column, preferring the top header band.
        if not candidates:
            return header_info
        band_size = max(10, int(statistics.median([c["height"] for c in candidates]) * 1.8))
        bands = {}
        for cand in candidates:
            band = int(cand["y_center"] // band_size)
            bands.setdefault(band, []).append(cand)
        best_band = max(bands.keys(), key=lambda b: (len({c["col"] for c in bands[b]}), sum(c["score"] for c in bands[b])))
        for col in t_cfg.keys():
            col_candidates = [c for c in bands[best_band] if c["col"] == col]
            if col_candidates:
                best = max(col_candidates, key=lambda c: c["score"])
                header_info[col] = {"idx": best["idx"], **{k: best[k] for k in ("x_min", "x_max", "y_min", "y_max", "x_center", "y_center", "height")}}
        return header_info

    @staticmethod
    def _build_column_boundaries(
        header_info: Dict[str, dict],
        image_width: int,
        ordered_columns: List[str],
    ) -> Dict[str, Dict[str, float]]:
        discovered = [(col, header_info[col]) for col in ordered_columns if col in header_info]
        discovered.sort(key=lambda pair: pair[1]["x_center"])
        boundaries = {}

        for i, (col_name, info) in enumerate(discovered):
            left = 0 if i == 0 else (discovered[i - 1][1]["x_max"] + info["x_min"]) / 2
            right = image_width if i == len(discovered) - 1 else (info["x_max"] + discovered[i + 1][1]["x_min"]) / 2
            boundaries[col_name] = {"x_min": left, "x_max": right}

        if not boundaries:
            default_width = image_width / max(1, len(ordered_columns))
            for pos, col_name in enumerate(ordered_columns):
                boundaries[col_name] = {"x_min": pos * default_width, "x_max": (pos + 1) * default_width}
            return boundaries

        last_known = None
        known_positions = {name: idx for idx, name in enumerate(ordered_columns)}
        for col_name in ordered_columns:
            if col_name in boundaries:
                last_known = boundaries[col_name]
                continue
            pos = known_positions[col_name]
            left_neighbor = None
            right_neighbor = None
            for step in range(1, len(ordered_columns)):
                if pos - step >= 0 and ordered_columns[pos - step] in boundaries and left_neighbor is None:
                    left_neighbor = boundaries[ordered_columns[pos - step]]
                if pos + step < len(ordered_columns) and ordered_columns[pos + step] in boundaries and right_neighbor is None:
                    right_neighbor = boundaries[ordered_columns[pos + step]]
                if left_neighbor and right_neighbor:
                    break

            if left_neighbor and right_neighbor:
                span = right_neighbor["x_min"] - left_neighbor["x_max"]
                midpoint = left_neighbor["x_max"] + max(20, span / 2)
                boundaries[col_name] = {"x_min": left_neighbor["x_max"], "x_max": midpoint}
            elif left_neighbor:
                boundaries[col_name] = {"x_min": left_neighbor["x_max"], "x_max": min(image_width, left_neighbor["x_max"] + image_width * 0.12)}
            elif right_neighbor:
                boundaries[col_name] = {"x_min": max(0, right_neighbor["x_min"] - image_width * 0.12), "x_max": right_neighbor["x_min"]}
            elif last_known:
                boundaries[col_name] = {"x_min": last_known["x_max"], "x_max": min(image_width, last_known["x_max"] + image_width * 0.12)}
            else:
                boundaries[col_name] = {"x_min": 0, "x_max": image_width}

        return boundaries

    @staticmethod
    def _group_rows_adaptive(elements: List[dict], parser_settings: dict) -> List[List[dict]]:
        if not elements:
            return []

        row_factor = parser_settings.get("ROW_Y_TOL_FACTOR", 0.75)
        min_tol = parser_settings.get("MIN_ROW_Y_TOL", 10)
        median_height = statistics.median([el["height"] for el in elements]) if elements else min_tol
        y_tol = max(min_tol, median_height * row_factor)

        rows = []
        current = [elements[0]]
        current_y = elements[0]["y_center"]
        for el in elements[1:]:
            if abs(el["y_center"] - current_y) <= y_tol:
                current.append(el)
                current_y = sum(item["y_center"] for item in current) / len(current)
            else:
                rows.append(sorted(current, key=lambda item: item["x_center"]))
                current = [el]
                current_y = el["y_center"]
        rows.append(sorted(current, key=lambda item: item["x_center"]))

        # Post-merge for OCR fragments of the same visual row.
        # Keep the threshold conservative to avoid merging real adjacent rows.
        merged_rows = []
        merge_factor = parser_settings.get("ROW_POST_MERGE_FACTOR", 0.35)
        for row in rows:
            if not merged_rows:
                merged_rows.append(row)
                continue
            prev = merged_rows[-1]
            y_gap = abs((sum(r["y_center"] for r in row) / len(row)) - (sum(p["y_center"] for p in prev) / len(prev)))
            if y_gap <= y_tol * merge_factor:
                merged_rows[-1] = sorted(prev + row, key=lambda item: item["x_center"])
            else:
                merged_rows.append(row)
        return merged_rows

    @staticmethod
    def _find_best_column(el: dict, boundaries: Dict[str, Dict[str, float]], column_type_hints: dict) -> Optional[str]:
        best_col = None
        best_overlap = -1.0
        for col_name, bnd in boundaries.items():
            overlap = max(0.0, min(el["x_max"], bnd["x_max"]) - max(el["x_min"], bnd["x_min"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_col = col_name

        if best_col is None:
            return None

        hint = column_type_hints.get(best_col)
        if hint == "number" and not _is_number_like(el["text"]):
            center_candidates = sorted(
                boundaries.items(),
                key=lambda pair: abs((pair[1]["x_min"] + pair[1]["x_max"]) / 2 - el["x_center"]),
            )
            for candidate_name, _ in center_candidates:
                if column_type_hints.get(candidate_name) != "number":
                    return candidate_name
        return best_col

    @staticmethod
    def extract_table(
        ocr_results: list,
        t_cfg: dict,
        t_name: str,
        used_indices: set,
        image_width: int,
        all_fields_keywords: list,
        parser_settings: Optional[dict] = None,
        region_bounds: Optional[Dict[str, float]] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        parser_settings = parser_settings or {}
        min_header_score = parser_settings.get("MIN_HEADER_SCORE", 0.7)
        column_type_hints = parser_settings.get("COLUMN_TYPE_HINTS", {})

        if not region_bounds:
            regions = TableParser.detect_table_regions(
                ocr_results=ocr_results,
                tables_cfg={t_name: t_cfg},
                field_keywords=all_fields_keywords,
                parser_settings=parser_settings,
            )
            region_bounds = regions.get(t_name)
            if not region_bounds:
                return pd.DataFrame(), list(t_cfg.keys())

        header_info = TableParser._get_headers(ocr_results, t_cfg, region_bounds, min_header_score)
        if not header_info:
            return pd.DataFrame(), list(t_cfg.keys())
        missing = [col for col in t_cfg.keys() if col not in header_info]
        y_start = max(info["y_max"] for info in header_info.values()) + 5

        elements = []
        stop_keywords = [kw.lower() for kw in parser_settings.get("TABLE_STOP_KEYWORDS", [])] + [kw.lower() for kw in all_fields_keywords]
        stop_keywords_norm = [_normalize_for_match(kw) for kw in stop_keywords]
        for idx, res in enumerate(ocr_results):
            if idx in used_indices or idx in {header["idx"] for header in header_info.values()}:
                continue

            stats = _bbox_stats(res[0])
            if not (region_bounds["y_min"] <= stats["y_center"] <= region_bounds["y_max"]):
                continue
            if stats["y_center"] <= y_start:
                continue

            text = _clean_ocr_text(res[1][0])
            text_norm = _normalize_for_match(text)
            if any(_similarity(text_norm, kw) > 0.72 or (kw and kw in text_norm) for kw in stop_keywords_norm):
                continue

            if text:
                elements.append({"text": text, **stats})

        elements.sort(key=lambda item: item["y_center"])
        rows = TableParser._group_rows_adaptive(elements, parser_settings)
        boundaries = TableParser._build_column_boundaries(header_info, image_width, list(t_cfg.keys()))

        table_data = []
        min_non_empty_cols = parser_settings.get(
            "MIN_ROW_FILLED_COLS",
            max(2, int(math.ceil(len(t_cfg.keys()) * 0.34))),
        )
        for row in rows:
            row_dict = {col: "" for col in t_cfg.keys()}
            for el in row:
                target_col = TableParser._find_best_column(el, boundaries, column_type_hints)
                if not target_col:
                    continue
                row_dict[target_col] += (" " if row_dict[target_col] else "") + _clean_ocr_text(el["text"])

            non_empty_cols = sum(1 for value in row_dict.values() if value.strip())
            row_text_norm = _normalize_for_match(" ".join(v for v in row_dict.values() if v))
            has_stop_marker = any(kw and (_similarity(row_text_norm, kw) > 0.66 or kw in row_text_norm) for kw in stop_keywords_norm)
            numeric_cols = [c for c, hint in column_type_hints.items() if hint == "number" and c in row_dict]
            has_numeric_value = any(_is_number_like(row_dict.get(col, "")) for col in numeric_cols)
            non_numeric_cols = [c for c in row_dict.keys() if c not in numeric_cols]
            has_anchor_text = any(len((row_dict.get(col, "") or "").strip()) >= 3 for col in non_numeric_cols)

            # Demo-friendly acceptance: keep row when anchor text exists and at least 2 cells are filled,
            # even if OCR damaged all numeric cells.
            if (
                non_empty_cols >= min_non_empty_cols
                and not has_stop_marker
                and (not numeric_cols or has_numeric_value or has_anchor_text)
            ):
                table_data.append(row_dict)

        return pd.DataFrame(table_data), missing
