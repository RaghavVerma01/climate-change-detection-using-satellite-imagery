import logging
import os

def get_logger(name="satellite_pipeline"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%H:%M:%S"
    )

    #Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    #File handler
    os.makedirs("logs",exist_ok=True)
    fh = logging.FileHandler("logs/pipeline.log")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger