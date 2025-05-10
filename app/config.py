import os
import logging

# Backend
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:3000")
BACKEND_API_PREFIX = os.getenv("BACKEND_API_PREFIX", "/api/v1")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "./weights/yolov8-vn-food-classification.pt")
YOLO_GENERAL_CLS_MODEL_PATH = os.getenv("YOLO_GENERAL_CLS_MODEL_PATH", "./weights/yolo11s-cls.pt")
DEFAULT_AUTH_TOKEN = os.getenv("DEFAULT_AUTH_TOKEN", "")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", 10.0))
OLLAMA_BASE_URL = "http://localhost:11434"  
OLLAMA_MODEL = "llama3.1:8b"  
# Logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("rag_indexer")
