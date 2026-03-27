import streamlit as st
import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
import pandas as pd

from config import DOC_TEMPLATES
from table_parser import FieldExtractor, TableParser

# ==========================================
# 1. OCR И ПРЕДОБРАБОТКА
# ==========================================
class ImagePreprocessor:
    @staticmethod
    def enhance_image(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        denoised = cv2.bilateralFilter(gray, 5, 50, 50) 
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return cv2.cvtColor(clahe.apply(denoised), cv2.COLOR_GRAY2RGB)

class OCRProcessor:
    def __init__(self):
        self.engine = PaddleOCR(use_angle_cls=True, lang='ru', show_log=False, 
                                det_db_unclip_ratio=1.5, det_db_thresh=0.2, 
                                det_db_box_thresh=0.5, det_limit_side_len=1500)

    def extract_data(self, image: np.ndarray, progress_bar=None):
        dt_boxes, _ = self.engine.text_detector(image)
        if dt_boxes is None or len(dt_boxes) == 0: return []
        results = []
        for i, box in enumerate(dt_boxes):
            pts = box.astype("float32")
            w, h = int(np.linalg.norm(pts[0]-pts[1])), int(np.linalg.norm(pts[0]-pts[3]))
            # Минимальный паддинг 2px
            M = cv2.getPerspectiveTransform(pts, np.array([[2,2],[w+2,2],[w+2,h+2],[2,h+2]], "float32"))
            crop = cv2.warpPerspective(image, M, (w+4, h+4))
            up = cv2.resize(crop, (w*3, h*3), interpolation=cv2.INTER_LANCZOS4)
            rec, _ = self.engine.text_recognizer([up])
            if rec and rec[0][1] > 0.4:
                results.append([box.tolist(), rec[0]])
            if progress_bar: progress_bar.progress((i+1)/len(dt_boxes))
        return results

# ==========================================
# 2. GUI
# ==========================================
st.set_page_config(page_title="TTN Parser Elite", layout="wide")

@st.cache_resource
def get_ocr(): return OCRProcessor()
ocr = get_ocr()

st.title("📄 ТТН в json")

with st.sidebar:
    selected_template = st.selectbox("Шаблон:", list(DOC_TEMPLATES.keys()))
    config = DOC_TEMPLATES[selected_template]

up_file = st.file_uploader("Загрузите файл", type=['png', 'jpg', 'jpeg'])

if up_file:
    img_arr = np.array(Image.open(up_file))
    p_bar = st.progress(0)
    enhanced = ImagePreprocessor.enhance_image(img_arr)
    raw_data = ocr.extract_data(enhanced, p_bar)
    p_bar.empty()
    
    col_img, col_res = st.columns([1, 1.8])
    with col_img: st.image(enhanced)

    with col_res:
        # 1. Поля (FIELDS)
        field_keywords = []
        for kws in config.get("FIELDS", {}).values(): field_keywords.extend(kws)
        
        parser_settings = config.get("PARSER_SETTINGS", {})
        table_regions = TableParser.detect_table_regions(
            ocr_results=raw_data,
            tables_cfg=config.get("TABLES", {}),
            field_keywords=field_keywords,
            parser_settings=parser_settings,
        )
        free_text_y_min = 0.0
        if table_regions:
            free_text_y_min = max(region["y_max"] for region in table_regions.values()) + 5

        fields, used_idx = FieldExtractor.extract_from_free_text(
            raw_data,
            config.get("FIELDS", {}),
            free_text_y_min=free_text_y_min,
            table_regions=table_regions,
        )
        
        st.subheader("📝 Поля документа")
        f_cols = st.columns(len(fields))
        for i, (k, v) in enumerate(fields.items()):
            with f_cols[i % len(f_cols)]:
                st.metric(k, v if v else "❌")
        
        st.divider()

        # 2. Таблицы
        if "TABLES" in config:
            for t_name, t_cfg in config["TABLES"].items():
                st.subheader(f"📊 {t_name}")
                df, missing = TableParser.extract_table(
                    raw_data,
                    t_cfg,
                    t_name,
                    used_idx,
                    img_arr.shape[1],
                    field_keywords,
                    parser_settings=parser_settings,
                    region_bounds=table_regions.get(t_name),
                )
                
                if not df.empty:
                    # Функция для закрашивания missing столбцов
                    def highlight(x):
                        c = 'background-color: #ffffcc'
                        df1 = pd.DataFrame('', index=x.index, columns=x.columns)
                        for col in missing:
                            if col in df1.columns: df1[col] = c
                        return df1

                    st.dataframe(df.style.apply(highlight, axis=None), use_container_width=True, hide_index=True)
                    st.download_button(f"Скачать JSON", df.to_json(orient='records', force_ascii=False), key=f"d_{t_name}")
                else: st.warning(f"Таблица '{t_name}' не найдена.")