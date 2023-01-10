import time

from mylogger import logger


def time_it(func):
    def wrapper(*args, **kwargs):
        logger.info(f'Running {func.__name__}...')
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"Time taken by {func.__name__} is {end - start:.04f} seconds")
        return result
    return wrapper
