import os
import logging
from dotenv import dotenv_values

# Load environment variables from .env file
config = dotenv_values(".env")

# Backend
BACKEND_URL = config["BACKEND_URL"]
BACKEND_API_PREFIX = config["BACKEND_API_PREFIX"]
YOLO_MODEL_PATH = config["YOLO_MODEL_PATH"]
YOLO_GENERAL_CLS_MODEL_PATH = config["YOLO_GENERAL_CLS_MODEL_PATH"]
DEFAULT_AUTH_TOKEN = config["DEFAULT_AUTH_TOKEN"]
REQUEST_TIMEOUT = float(config["REQUEST_TIMEOUT"])
OLLAMA_BASE_URL = "http://localhost:11434"  
OLLAMA_MODEL = "llama3.1:8b"  
# Logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("rag_indexer")
