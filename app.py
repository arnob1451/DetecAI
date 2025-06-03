import os
from dotenv import load_dotenv
import streamlit as st
import cv2
import numpy as np
import json
import io
import warnings
import time
from PIL import Image
from ultralytics import YOLO
import pandas as pd
from pathlib import Path
from config import *
from utils import logger, log_execution_time, validate_image, cleanup_cache, setup_logging

# Load environment variables
load_dotenv()

# Setup logging
setup_logging()

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

[... Rest of your existing app.py code ...]