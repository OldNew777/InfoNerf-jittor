import logging
import os
import sys

from logging import handlers

log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'log.log')
if not os.path.exists(os.path.dirname(log_file)):
    os.makedirs(os.path.dirname(log_file))


def clear_log_file() -> bool:
    try:
        with open(log_file, 'w') as f:
            return True
    except:
        return False


log_format = '[%(asctime)s] [%(levelname)s] %(message)s'
date_format = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

logging.basicConfig(level=logging.DEBUG, format=log_format, datefmt=date_format)

logger_instance = logging.getLogger(__name__)
logger_instance.setLevel(logging.DEBUG)

file_handler = handlers.RotatingFileHandler(log_file, mode='a', maxBytes=1024 * 1024 * 8)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)
logger_instance.addHandler(file_handler)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.DEBUG)
logger_instance.addHandler(stream_handler)


class MyLogger:
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    FATAL = logging.FATAL

    @classmethod
    def debug(cls, *msg):
        logger_instance.debug(' '.join([str(m) for m in msg]))

    @classmethod
    def info(cls, *msg):
        logger_instance.info(' '.join([str(m) for m in msg]))

    @classmethod
    def warning(cls, *msg):
        logger_instance.warning(' '.join([str(m) for m in msg]))

    warn = warning

    @classmethod
    def error(cls, *msg):
        logger_instance.error(' '.join([str(m) for m in msg]))

    @classmethod
    def fatal(cls, *msg):
        logger_instance.critical(' '.join([str(m) for m in msg]))

    @classmethod
    def assert_true(cls, cond: bool, *msg):
        if not cond:
            cls.fatal(*msg)
            sys.exit(1)

    @classmethod
    def exception(cls, *msg):
        logger_instance.exception(' '.join([str(m) for m in msg]))

    @classmethod
    def log(cls, level, *msg):
        logger_instance.log(level, ' '.join([str(m) for m in msg]))

    @classmethod
    def set_level(cls, level):
        stream_handler.setLevel(level)


logger = MyLogger
