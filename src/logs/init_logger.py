import logging
import os


def init_logger(name: str, filepath: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
