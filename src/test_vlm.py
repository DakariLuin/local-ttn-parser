import sys
from types import ModuleType

# ЭКСТРЕННАЯ ЗАПЛАТКА: Создаем фейковый модуль, если оригинальный сломан
try:
    import aistudio_sdk.hub
except (ImportError, AttributeError):
    print("Фикс: Исправляю структуру aistudio_sdk в памяти...")
    mock_hub = ModuleType("aistudio_sdk.hub")
    mock_hub.download = lambda *args, **kwargs: None
    sys.modules["aistudio_sdk.hub"] = mock_hub

import paddle
from paddlenlp.transformers import AutoModel, AutoImageProcessor, AutoTokenizer

model_name = "PaddlePaddle/PaddleOCR-VL-1.5-0.9B"

print("--- ТЕСТ ЗАГРУЗКИ SOTA VLM ---")
try:
    print("1. Загрузка токенайзера...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("2. Загрузка процессора изображений...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    
    print("3. Загрузка модели (около 2Гб, ждите)...")
    # Используем CPU для теста, чтобы не упасть по памяти GPU
    paddle.set_device("cpu")
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    print("\n✅ УСПЕХ! Все компоненты PaddleOCR-VL-1.5 загружены.")
except Exception as e:
    print(f"\n❌ ОШИБКА: {e}")