import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Environment configs
MODEL_PATH = "yolov8n.pt"
MAX_IMAGE_SIZE = 800
LOG_LEVEL = "INFO"

# App configs
EDGE_METHODS = ["Canny", "Sobel", "Laplacian"]
DEFAULT_CLASSES_TO_SHOW = 3

# Cache settings
CACHE_DIR = BASE_DIR / "cache"
CACHE_TIMEOUT = 3600  # 1 hour

# Image processing configs
SUPPORTED_FORMATS = ["jpg", "jpeg", "png"]
MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5MB