import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import paddle
from paddleocr import PaddleOCR
import math

# ==========================================
# 1. МОДУЛЬ ПРЕДОБРАБОТКИ ИЗОБРАЖЕНИЙ
# ==========================================
class ImagePreprocessor:
    @staticmethod
    def enhance_image(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=30)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


# ==========================================
# 2. МОДУЛЬ OCR И ИЗВЛЕЧЕНИЯ
# ==========================================
class OCRProcessor:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ru')

    def extract_data(self, image: np.ndarray):
        results = self.ocr.ocr(image)
        return results[0] if results and results[0] else []

    @staticmethod
    def draw_boxes(image: Image.Image, ocr_results: list) -> Image.Image:
        draw_img = image.copy()
        draw = ImageDraw.Draw(draw_img)
        for line in ocr_results:
            if line and len(line) > 0:
                box = line[0]
                points = [(point[0], point[1]) for point in box]
                draw.polygon(points, outline="red", width=3)
        return draw_img

    @staticmethod
    def structure_text_spatially(ocr_results: list) -> dict:
        """
        Группирует текст по координате Y (собирает текст в логические строки).
        Это первый шаг к структурированию ТТН до применения LLM/RegEx.
        """
        lines = []
        for res in ocr_results:
            box = res[0]
            text = res[1][0]
            # Берем среднюю Y-координату блока
            center_y = sum([point[1] for point in box]) / 4
            lines.append({"y": center_y, "text": text, "box": box})
            
        # Сортируем все блоки сверху вниз
        lines.sort(key=lambda x: x['y'])
        
        structured_lines = {}
        line_threshold = 15 # Погрешность в пикселях для одной строки
        
        current_y = None
        for item in lines:
            if current_y is None:
                current_y = item['y']
                structured_lines[current_y] = [item['text']]
            elif abs(item['y'] - current_y) < line_threshold:
                # Текст находится на той же визуальной строке
                structured_lines[current_y].append(item['text'])
            else:
                # Новая строка
                current_y = item['y']
                structured_lines[current_y] = [item['text']]
                
        # Форматируем в читаемый вид
        formatted_output = {}
        for idx, (y, texts) in enumerate(structured_lines.items()):
            # Соединяем текст на одной строке через разделитель
            formatted_output[f"Строка {idx+1}"] = " | ".join(texts)
            
        return formatted_output


# ==========================================
# 3. GUI
# ==========================================
st.set_page_config(page_title="Демо-стенд OCR ТТН", layout="wide")
st.title("👁️ Демонстрационный стенд распознавания ТТН")
st.markdown("Загрузите фото или скан накладной. Система автоматически очистит фото, выровняет текст и извлечет данные.")

# Инициализация модулей (кэшируем OCR, чтобы не грузить модель в память каждый раз)
@st.cache_resource
def get_ocr_processor():
    return OCRProcessor()

preprocessor = ImagePreprocessor()
ocr_processor = get_ocr_processor()

# Модуль загрузки
uploaded_file = st.file_uploader("📂 Загрузите скан или фото (JPG, PNG)", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Чтение изображения
    original_image = Image.open(uploaded_file)
    original_array = np.array(original_image)
    
    st.divider()
    
    # Создаем 3 колонки для пошаговой демонстрации
    col1, col2, col3 = st.columns(3)
    
    with st.spinner("Магия происходит... (Предобработка и OCR)"):
        # 1. Предобработка
        enhanced_array = preprocessor.enhance_image(original_array)
        enhanced_image = Image.fromarray(enhanced_array)
        
        # 2. OCR Распознавание
        raw_ocr_data = ocr_processor.extract_data(enhanced_array)
        
        # 3. Визуализация рамок
        boxed_image = ocr_processor.draw_boxes(enhanced_image, raw_ocr_data)
        
        # 4. Базовое пространственное структурирование
        structured_data = ocr_processor.structure_text_spatially(raw_ocr_data)

    # --- ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ ---
    
    with col1:
        st.subheader("1. Оригинал")
        st.image(original_image, use_column_width=True)
        st.caption("Сырое фото с телефона/сканера")

    with col2:
        st.subheader("2. Предобработка + Поиск")
        st.image(boxed_image, use_column_width=True)
        st.caption("Удаление шумов, выравнивание света, поиск текстовых блоков (Bounding Boxes)")

    with col3:
        st.subheader("3. Базово-структурированный текст")
        st.markdown("Текст сгруппирован по визуальным строкам (сверху-вниз, слева-направо).")
        st.json(structured_data)
        
        # Дополнительный вывод полного склеенного текста
        with st.expander("Показать весь текст сплошняком"):
            full_text = " ".join([res[1][0] for res in raw_ocr_data])
            st.write(full_text)