import logging
import time
from functools import wraps
from pathlib import Path
from config import CACHE_DIR, CACHE_TIMEOUT, MAX_UPLOAD_SIZE, SUPPORTED_FORMATS

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_logging():
    """Configure logging settings"""
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def log_execution_time(func):
    """Decorator to log function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
        return result
    return wrapper

def validate_image(image_file):
    """Validate uploaded image file"""
    if image_file is None:
        return False, "No file uploaded"
    
    file_size = len(image_file.getvalue())
    if file_size > MAX_UPLOAD_SIZE:
        return False, f"File size exceeds {MAX_UPLOAD_SIZE/1024/1024}MB limit"
    
    file_ext = Path(image_file.name).suffix.lower()[1:]
    if file_ext not in SUPPORTED_FORMATS:
        return False, f"Unsupported file format. Supported formats: {', '.join(SUPPORTED_FORMATS)}"
    
    return True, "Valid image file"

def cleanup_cache():
    """Clean up expired cache files"""
    try:
        current_time = time.time()
        cache_dir = Path(CACHE_DIR)
        
        if not cache_dir.exists():
            return
            
        for cache_file in cache_dir.glob("*"):
            if current_time - cache_file.stat().st_mtime > CACHE_TIMEOUT:
                cache_file.unlink()
                logger.info(f"Removed expired cache file: {cache_file}")
    except Exception as e:
        logger.error(f"Cache cleanup failed: {str(e)}")