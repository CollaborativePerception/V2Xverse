import sys
import logging


def initialize_root_logger(path : str) -> logging.Logger:
    """Define a root logger

        NOTE: debug information is set to be output via stream (stdout)
        and is not going to be written to log file (for log simplicity).
        Thus, set INFO level to message that you want to output to log file.

    Args:
        path: path to log file

    Return:
        Initilaized root logger
    """
    FORMAT = '%(asctime)s :: %(name)s :: %(levelname)-8s :: %(message)s'
    FORMATTER= logging.Formatter(fmt=FORMAT)

    logger = logging.getLogger()  # root logger
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(FORMATTER)
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(filename=path)
    file_handler.setFormatter(FORMATTER)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger
