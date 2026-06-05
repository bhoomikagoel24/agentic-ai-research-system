import os
import sys
import logging
from datetime import datetime

# LOG DIRECTORY

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# LOG FILE
LOG_FILE = os.path.join(
LOG_DIR,
f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
)

# LOGGER FACTORY

def get_logger(name: str) -> logging.Logger:

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # FORMATTER
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    # FILE HANDLER
    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8"
    )

    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # CONSOLE HANDLER
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Windows UTF-8 fix
    if hasattr(console_handler.stream, "reconfigure"):
        console_handler.stream.reconfigure(encoding="utf-8")

    # ADD HANDLERS
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
