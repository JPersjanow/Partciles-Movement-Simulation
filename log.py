import logging
import os

logger = None


def setup_custom_logger(name):
    global logger

    if not logger:
        formater = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(module)s:%(lineno)s - %(message)s"
        )

        # logging in console
        handler = logging.StreamHandler()
        handler.setFormatter(formater)

        try:
            os.remove(f"{name}.log")
        except OSError:
            pass

        file_handler = logging.FileHandler(f"{name}.log")
        file_handler.setFormatter(formater)
        file_handler.setLevel(logging.DEBUG)

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        # logger.addHandler(handler)
        logger.addHandler(file_handler)

    return logger
