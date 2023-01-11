import time
import jittor as jt

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


def log_exception(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(e)
            raise e
    return wrapper


def std(x: jt.Var):
    r"""Compute the standard deviation along the specified dimension.

    Args:
        x (jt.Var): input tensor
        dim (int): the dimension to reduce
        unbiased (bool): whether to use the unbiased estimation or not
        keepdim (bool): whether the output tensor has dim retained or not

    Returns:
        jt.Var: the standard deviation
    """
    mean = x.mean(dim=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdims=True)
    return var.sqrt()
