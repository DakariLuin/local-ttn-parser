import streamlit as st
import json
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import pandas as pd
import re

from config import DOC_TEMPLATES
from llm_parser import LLMDataExtractor

# Настройка страницы
st.set_page_config(page_title="SOTA VLM Parser Pro", layout="wide")

class SotaVLMProcessor:
    def __init__(self):
        self.model_id = "Qwen/Qwen2-VL-2B-Instruct"
        try:
            with st.spinner("Загрузка SOTA VLM модели (4GB)..."):
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_id, 
                    torch_dtype="auto", 
                    device_map="auto"
                )
                self.processor = AutoProcessor.from_pretrained(self.model_id)
        except Exception as e:
            st.error(f"Критическая ошибка загрузки модели: {e}")

    def process_image(self, image_pil, config):
        fields = list(config.get("FIELDS", {}).keys())
        tables = list(config.get("TABLES", {}).keys())

        # МАКСИМАЛЬНО ЖЕСТКИЙ ПРОМПТ НА АНГЛИЙСКОМ (для лучшего понимания моделью 2B)
        # Мы требуем только JSON-подобную структуру в Markdown, чтобы исключить болтовню.
        prompt = f"""Task: OCR Extraction.
1. Identify these keys: {fields}.
2. Extract these tables: {tables}.

Instructions:
- Use Russian language for values.
- If a value is not found, write "not found".
- Format: Strictly Markdown.
- No conversational text. No "Sure", no "I can help". 
- Only data."""

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "image": image_pil,
                        "min_pixels": 256 * 256,
                        "max_pixels": 1024 * 1024, # Оптимально для 2B
                    },
                    {"type": "text", "text": prompt}
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=1024,
                temperature=0.0,      # АБСОЛЮТНЫЙ НОЛЬ (запрет на фантазию)
                do_sample=False,      # Отключаем случайный выбор токенов
                repetition_penalty=1.2,
                no_repeat_ngram_size=5 
            )
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            return self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# --- Инициализация ---

@st.cache_resource
def load_vlm_engine():
    return SotaVLMProcessor()

@st.cache_resource
def load_llm_extractor():
    return LLMDataExtractor()

vlm = load_vlm_engine()
llm = load_llm_extractor()

# --- Интерфейс ---

st.title("🚀 SOTA Document VLM (No-Hallucination Mode)")

with st.sidebar:
    st.header("Настройки")
    selected_tpl_name = st.selectbox("Шаблон документа:", list(DOC_TEMPLATES.keys()))
    current_config = DOC_TEMPLATES[selected_tpl_name]
    if st.button("Очистить"):
        st.session_state.clear()
        st.rerun()

file = st.file_uploader("Загрузить скан", type=['png', 'jpg', 'jpeg'])

if file:
    img = Image.open(file).convert("RGB")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(img, use_column_width=True)

    with col2:
        if st.button("🚀 Начать разбор"):
            with st.spinner("VLM сканирует (Strict Mode)..."):
                markdown_output = vlm.process_image(img, current_config)
                st.session_state.last_markdown = markdown_output

        if 'last_markdown' in st.session_state:
            with st.expander("Сырой Markdown от VLM"):
                st.text(st.session_state.last_markdown)

            if st.button("📦 Сформировать JSON"):
                with st.spinner("Ollama чистит галлюцинации..."):
                    final_json, raw_res = llm.extract_from_vlm(st.session_state.last_markdown, current_config)
                    st.session_state.last_json = final_json
            
            if 'last_json' in st.session_state and st.session_state.last_json:
                st.subheader("Итоговый JSON")
                st.json(st.session_state.last_json)
                st.download_button("📥 Скачать JSON", json.dumps(st.session_state.last_json, indent=4, ensure_ascii=False), "result.json")