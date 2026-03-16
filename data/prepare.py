import os
import json
import re
import urllib.request
import zipfile
from pathlib import Path
import urllib.parse
import unicodedata
import time
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Fix for Windows console encoding to support Unicode characters like emojis
if os.name == 'nt':
    import subprocess
    try:
        subprocess.run('chcp 65001', shell=True, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        pass  # Ignore if it fails

# ----------------------------------------------------------------------
# Configuración de Logging
# ----------------------------------------------------------------------

def configurar_logger():
    """Configura un logger mejorado con timestamps y colores."""
    logger = logging.getLogger("dataset-builder")
    logger.setLevel(logging.DEBUG)
    
    # Formatos con timestamps
    fmt_console = logging.Formatter(
        '[%(asctime)s] %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    fmt_file = logging.Formatter(
        '[%(asctime)s] %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler para consola
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt_console)
    logger.addHandler(ch)
    
    # Handler para archivo
    fh = logging.FileHandler('dataset-builder.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt_file)
    logger.addHandler(fh)
    
    return logger

logger = configurar_logger()

# ----------------------------------------------------------------------
# Configuración General
# ----------------------------------------------------------------------

OUTPUT_FILE = "data/dataset.jsonl"
os.makedirs("data", exist_ok=True)

from data.sources import IDS_LIBROS_GUTENBERG, ARTICULOS_WIKIPEDIA
