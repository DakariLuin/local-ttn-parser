import json
import re
from openai import OpenAI

class LLMDataExtractor:
    def __init__(self):
        self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    def extract_from_vlm(self, markdown_text: str, config: dict) -> tuple:
        fields = list(config.get("FIELDS", {}).keys())
        tables = list(config.get("TABLES", {}).keys())

        prompt = f"""Ты — конвертер данных. Преврати текст из Markdown в JSON.
### ОЖИДАЕМЫЙ ШАБЛОН:
Ключи: {fields} и названия таблиц: {tables}.

### ДАННЫЕ (Markdown):
{markdown_text}

Верни только валидный JSON.
"""
        try:
            response = self.client.chat.completions.create(
                model="qwen2.5:14b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            raw_res = response.choices[0].message.content.strip()
            match = re.search(r'\{.*\}', raw_res, re.DOTALL)
            if match: raw_res = match.group(0)
            return json.loads(raw_res), raw_res
        except Exception as e:
            return None, str(e)