import os
import random
from numpy.random import seed as numpy_seed
import torch


def make_deterministic(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
