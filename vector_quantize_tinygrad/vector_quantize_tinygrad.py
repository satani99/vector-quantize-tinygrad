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

def normalize(t, p=2, dim=1, eps=1e-12):
    norm = np.linalg.norm(t.numpy(), ord=p, axis=dim, keepdim=True)
    norm += eps 
    normalized_t = t / norm 
    return Tensor(normalized_t)

def l2norm(t):
    return normalize(t, p=2, dim=-1)

def cdlist(x, y):
    x2 = reduce(x.numpy() ** 2, 'b n d -> b n', 'sum')
    y2 = reduce(y.numpy() ** 2, 'b n d -> b n', 'sum')
    xy = np.einsum('b i d, b j d -> b i j', x.numpy(), y.numpy()) * -2 
    return (Tensor(rearrange(x2, 'b i -> b i 1') + rearrange(y2, 'b j -> b 1 j') + xy)).sqrt()

def log(t, eps=1e-20):
    return t.clip(min_=eps, max_=float("inf")).log()




