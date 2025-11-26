import logging
import os

def get_logger(name: str = __name__) -> logging.Logger:

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if imported multiple times
    if not logger.handlers:
        handler = logging.StreamHandler()  # prints to terminal
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        os.makedirs("logs", exist_ok=True)
        file_handler = logging.FileHandler("logs/app.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
