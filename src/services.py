import cv2
import numpy as np
import re
from paddleocr import PaddleOCR
from models import TTNDocument, DocumentMeta, Participants, VehicleInfo, Totals

class ImagePreprocessor:
    """Модуль улучшения качества изображения перед OCR"""
    @staticmethod
    def process(image: np.ndarray) -> np.ndarray:
        # Перевод в градации серого
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Увеличение контраста (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        # Конвертация обратно в RGB для PaddleOCR
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


class OCREngine:
    """Обертка над PaddleOCR"""
    def __init__(self):
        # use_angle_cls=True - автоматический поворот текста
        # lang='ru' - поддержка русского + английские цифры
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ru', show_log=False)

    def extract(self, image: np.ndarray) -> list:
        results = self.ocr.ocr(image, cls=True)
        # Извлекаем только текст (без координат для MVP парсера)
        extracted_text = []
        if results[0]:
            for line in results[0]:
                text = line[1][0]
                extracted_text.append(text)
        return extracted_text


class DataExtractor:
    """Парсер сырого текста в структурированный JSON (на базе эвристик/Regex)"""
    
    @staticmethod
    def parse(text_lines: list) -> TTNDocument:
        doc = TTNDocument()
        full_text = " ".join(text_lines)

        # 1. Парсинг меты (очень примитивный regex для MVP)
        doc_num_match = re.search(r'№\s*(\d+)', full_text)
        date_match = re.search(r'(\d{2}\.\d{2}\.\d{4})', full_text)
        if doc_num_match: doc.document_meta.doc_number = doc_num_match.group(1)
        if date_match: doc.document_meta.doc_date = date_match.group(1)

        # 2. Поиск участников (эвристика: ищем ключевое слово и берем следующий текст)
        def find_after_keyword(keyword, lines):
            for i, line in enumerate(lines):
                if keyword.lower() in line.lower() and i + 1 < len(lines):
                    return lines[i+1]
            return None

        doc.participants.shipper = find_after_keyword("Грузоотправитель", text_lines)
        doc.participants.consignee = find_after_keyword("Грузополучатель", text_lines)
        doc.participants.carrier = find_after_keyword("Перевозчик", text_lines)

        # 3. Итоги
        total_sum_match = re.search(r'Итого.*?(\d+[\.,]\d{2})', full_text, re.IGNORECASE)
        if total_sum_match: doc.totals.total_amount = total_sum_match.group(1)

        # Примечание: парсинг таблиц (goods_table) сложен для регулярных выражений. 
        # Для реального проекта тут внедряется LayoutLM или Paddle Structure.
        # В MVP оставляем список пустым или парсим примитивно.

        return doc


class TTNPipeline:
    """Legacy MVP конвейер (оставлен для обратной совместимости).

    Основной production-пайплайн с region-based парсингом таблиц находится в app.py/table_parser.py.
    """
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.ocr_engine = OCREngine()
        self.extractor = DataExtractor()

    def run(self, image: np.ndarray) -> TTNDocument:
        processed_img = self.preprocessor.process(image)
        raw_text = self.ocr_engine.extract(processed_img)
        structured_data = self.extractor.parse(raw_text)
        return structured_data, raw_text