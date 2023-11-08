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

def normalize(t, p=2, axis=1, eps=1e-12):
    norm = np.linalg.norm(t.numpy(), ord=p, axis=axis, keepdim=True)
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

def lerp(input, end, weight):
    out = input + weight * (end - input)
    return out 

def ema_inplace(old, new, decay):
    is_mps = str(old.device).startswith('mps:')

    if not is_mps:
        old = lerp(old, new, 1-decay)
    else: 
        old = (old * decay) + (new * (1 - decay))

def pack_one(t, pattern):
    return Tensor(pack([t.numpy()], pattern))

def unpack_one(t, ps, pattern):
    return Tensor(unpack(t.numpy(), ps, pattern)[0])

def uniform_init(*shape):
    t = Tensor.kaiming_uniform(shape)
    return t 

def gumbel_noise(t):
    noise = Tensor.uniform(*t.shape)
    return -log(-log(noise))

def one_hot(arr, num_classes, dtype):
    one_hot = np.zeros((len(arr), num_classes), dtype=dtype)
    one_hot[np.arange(len(arr)), arr] = 1
    return one_hot

def gumbel_sample(
    logits,
    temperature = 1.,
    stochastic = False,
    straight_through = False,
    reinmax = False,
    axis = -1,
    training = True 
):
    dtype, size = logits.dtype, logits.shape[dim]

    if training and stochastic and temperature > 0:
        sampling_logits = (logits / temperature) + gumbel_noise(logits)
    else:
        sampling_logits = logits 

    ind = sampling_logits.argmax(axis=axis)
    one_hot = Tensor(one_hot(list(ind), size, dtype))

    assert not (reinmax and not straight_through), 'reinmax can only be turned on if using straight through gumbel softmax'

    if not straight_through or temperature <= 0. or not training:
        return ind, one_hot 

    # use reinmax for better second-order accuracy - https://arxiv.org/abs/2304.08612
    # algorithm 2

    if reinmax:
        pi0 = logits.softmax(axis=axis)
        pi1 = (one_hot + (logits / temperature).softamx(axis=axis)) / 2
        pi1 = ((log(pi1) - logits).detach() + logits).softmax(axis=axis)
        pi2 = 2 * pi1 - 0.5 * pi0 
        one_hot = pi2 - pi2.detach() + one_hot 
    else:
        pi1 = (logits / temperature).softmax(axis=axis)
        one_hot = one_hot + pi1 - pi1.detach()

    return ind, one_hot

def laplace_smoothing(x, n_categories, eps = 1e-5, dim = -1):
    denom = x.sum(axis=axis, keepdim=True)
    return (x + eps) / (denom + n_categories * eps)







