import logging
import time
from functools import wraps


def setup_logger(name: str = "hr_interview") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def log_duration(logger: logging.Logger, step_name: str, threshold: float = 120.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start
                if elapsed >= threshold:
                    logger.warning(
                        "SLOW | %s took %.2f seconds (exceeded %.0f s threshold)",
                        step_name, elapsed, threshold,
                    )
                else:
                    logger.info(
                        "%s completed in %.2f seconds", step_name, elapsed,
                    )
        return wrapper
    return decorator
