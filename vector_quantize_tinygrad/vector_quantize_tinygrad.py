from functools import partial 

import tinygrad 
import numpy as np 
from tinygrad.tensor import Tensor 
from tinygrad import nn
from tinygrad.helpers import dtypes
from einops import rearrange, repeat, reduce, pack, unpack
from mpi4py import MPI 
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

def cdist(x, y):
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

def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device 
    if num_samples >= num:
        indices = Tensor(np.random.permutation(num_samples), device=device)[:num]
    else:
        indices = Tensor(np.random.randint(0, num_samples, size=(num,)), device=device) 

    return samples[indices] 

def unbind(t, axis=0):
    sub_tensor = np.split(t, t.shape[axis], axis)
    return Tensor(sub_tensor)

def batched_sample_vectors(samples, num):
    return Tensor.stack([sample_vectors(sample, num) for sample in unbind(samples, axis=0)], dim=0)

def pad_shape(shape, size, dim=0):
    return [size if i == dim else s for i, s in enumerate(shape)]

def sample_multinomial(total_count, probs):
    device = probs.device 
    probs = probs.cpu() 

    total_count = probs.full((), total_count)
    remainder = probs.ones(())
    sample = Tensor.empty(*probs.shape, dtype=dtypes.int8)

    for i, p in enumerate(probs):
        s = Tensor(np.random.binomial(total_count.numpy(), (p / remainder).numpy()))
        sample[i] = s 
        total_count -= s 
        remainder -= p 

    return sample.to(device)

def get_world_size(distributed_tensor):
    return distributed_tensor.shape[0]

def all_gather(local_data):
    global_data = np.ndarray((world_size, *local_data.shape))
    for i in range(world_size):
        global_data[i, :] = local_data
    return Tensor(global_data)

def all_gather_sizes(x, dim):
    size = Tensor(x.shape[dim], dtype=dtypes.int8, device=x.device)
    world_size = get_world_size(x)
    all_sizes = all_gather(size)
    return Tensor.stack(all_sizes)

def get_rank(tensor):
    ranks = tensor.shape[0]
    return ranks

def new_empty(l):
    return Tensor(np.empty_like(np.ndarray((*l))))

def broadcast(tensor, root_rank, current_rank=0):
    comm = MPI.COMM_WORLD
    if current_rank == root_rank:
        for i in range(world_size):
            for i != root_rank:
                comm.send(tensor, dest=i)
    else:
        tensor = comm.recv(source=root_rank)
def barrier():
    comm = MPI.COMM_WORLD 
    comm.Barrier()

def all_gather_variably_sized(x, sizes, dim=0):
    rank = get_rank(x)
    all_x = []

    for i, size in enumerate(sizes):
        t = x if i == rank else new_empty(pad_shape(x.shape, size, dim))
        broadcast(t, root_rank=i, current_rank=i)
        all_x.append(t)
    
    barrier() 
    return all_x
    
def sample_vectors_distributed(local_samples, num):
    local_samples = rearrange(local_samples, '1 ... -> ...')

    rank = get_rank(local_samples)
    all_num_samples = all_gather_sizes(local_samples, dim=0)

    if rank == 0:
        samples_per_rank = sample_multinomial(num, all_num_samples / all_num_samples.sum())
    else:
        samples_per_rank = Tensor.empty(*all_num_samples.shape)

    broadcast(samples_per_rank, root_rank=0)
    samples_per_rank = samples_per_rank.numpy().tolist()

    local_samples = sample_vectors(local_samples, samples_per_rank[rank])
    all_samples = all_gather_variably_sized(local_samples, samples_per_rank, dim = 0)
    out = Tensor.cat(*all_samples, dim=0)

    return rearrange(out, '... -> 1 ...')

def scatter_add(target, indices, updates):
    Tensor(np.add.at(target.numpy(), indices.numpy(), updates.numpy()))

def batched_bincount(x, *, minlength):
    batch, dtype, device = x.shape[0], x.dtype, x.device 
    target = Tensor.zeros(batch, minlength, dtype=dtype, device=device)
    values = Tensor.ones(*x.shape)
    scatter_add(target, x, values)
    return target 

def kmeans(
    samples,
    num_clusters,
    num_iters=10,
    use_cosine_sim = False, 
    sample_fn = batched_sample_vectors,
    all_reduce_fn = noop,
):
    num_codebooks, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device 

    means = sample_fn(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples.matmul(rearrange(means, 'h n d -> h d n'))
        else:
            dists = -cdist(samples, means)

        buckets = dists.argmax(axis=-1)
        bins = batched_bincount(buckets, minlength = num_clusters)
        all_reduce_fn(new_means)

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = Tensor(np.where(
            rearrange(zero_mask, '... -> ... 1').numpy(),
            means.numpy(),
            new_means.numpy(),
        ))

    return means, bins 








