from functools import partial 

import tinygrad 
import numpy as np 
from tinygrad.tensor import Tensor 
from tinygrad import nn
from einops import rearrange, repeat, reduce, pack, unpack

from typing import Callable

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d 

def noop(*args, **kwargs):
    pass 

def identity(t):
    return t 


