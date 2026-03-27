import json
import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import DOC_TEMPLATES  # noqa: E402
from table_parser import FieldExtractor, TableParser  # noqa: E402


def _normalize(value: str) -> str:
    return " ".join(str(value).split()).strip().lower()


def _cell_accuracy(actual_rows, expected_rows):
    total = 0
    correct = 0
    for actual_row, expected_row in zip(actual_rows, expected_rows):
        for key, expected_val in expected_row.items():
            total += 1
            if _normalize(actual_row.get(key, "")) == _normalize(expected_val):
                correct += 1
    return correct / total if total else 0.0


class TestTablePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture_path = PROJECT_ROOT / "tests" / "fixtures" / "two_tables_case.json"
        with fixture_path.open("r", encoding="utf-8") as f:
            cls.fixture = json.load(f)

        cls.template = DOC_TEMPLATES["ТТН (Тестовое изображение)"]
        cls.tables_cfg = cls.template["TABLES"]
        cls.fields_cfg = cls.template["FIELDS"]
        cls.parser_settings = cls.template["PARSER_SETTINGS"]

    def test_region_detection_and_table_parsing(self):
        ocr_results = self.fixture["ocr_results"]
        image_width = self.fixture["image_width"]
        expected_tables = self.fixture["expected"]["tables"]

        field_keywords = []
        for keywords in self.fields_cfg.values():
            field_keywords.extend(keywords)

        regions = TableParser.detect_table_regions(
            ocr_results=ocr_results,
            tables_cfg=self.tables_cfg,
            field_keywords=field_keywords,
            parser_settings=self.parser_settings,
        )

        self.assertIn("Расчет стоимости", regions)
        self.assertIn("Адрес и Количество", regions)
        self.assertLess(regions["Расчет стоимости"]["y_min"], regions["Адрес и Количество"]["y_min"])
        self.assertLess(regions["Расчет стоимости"]["y_max"], regions["Адрес и Количество"]["y_max"])

        all_accuracies = []
        for table_name, table_cfg in self.tables_cfg.items():
            expected_rows = expected_tables[table_name]
            parsed_df, _ = TableParser.extract_table(
                ocr_results=ocr_results,
                t_cfg=table_cfg,
                t_name=table_name,
                used_indices=set(),
                image_width=image_width,
                all_fields_keywords=field_keywords,
                parser_settings=self.parser_settings,
                region_bounds=regions.get(table_name),
            )
            actual_rows = parsed_df.to_dict(orient="records")
            self.assertEqual(len(actual_rows), len(expected_rows), f"Wrong row count for {table_name}")
            accuracy = _cell_accuracy(actual_rows, expected_rows)
            all_accuracies.append(accuracy)
            self.assertGreaterEqual(accuracy, 0.95, f"Low cell accuracy for {table_name}: {accuracy}")

        table_found_rate = sum(1 for name in self.tables_cfg if name in regions) / len(self.tables_cfg)
        self.assertGreaterEqual(table_found_rate, 0.98)
        self.assertGreaterEqual(sum(all_accuracies) / len(all_accuracies), 0.95)

    def test_free_text_participants_extraction(self):
        ocr_results = self.fixture["ocr_results"]
        expected_fields = self.fixture["expected"]["fields"]

        field_keywords = []
        for keywords in self.fields_cfg.values():
            field_keywords.extend(keywords)

        regions = TableParser.detect_table_regions(
            ocr_results=ocr_results,
            tables_cfg=self.tables_cfg,
            field_keywords=field_keywords,
            parser_settings=self.parser_settings,
        )
        free_text_y_min = max(region["y_max"] for region in regions.values()) + 5

        actual_fields, _ = FieldExtractor.extract_from_free_text(
            ocr_results=ocr_results,
            field_config=self.fields_cfg,
            free_text_y_min=free_text_y_min,
        )

        for key, expected in expected_fields.items():
            self.assertEqual(_normalize(actual_fields.get(key, "")), _normalize(expected))


if __name__ == "__main__":
    unittest.main()
