import os
import sys
import time
import jittor as jt
import numpy as np
from tqdm import tqdm


def train():
    pass


if __name__ == '__main__':
    # disable multi-GPUs before running because of the bug of Jittor
    jt.flags.use_cuda = 1
    train()
