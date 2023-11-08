from tinygrad.tensor import Tensor
from einops import rearrange 
import numpy as np
a = Tensor([1., 2., 3.])
# print(rearrange(a.numpy(), 'b -> b 1'))
b = np.einsum('b , b -> b', a.numpy(), a.numpy())
# print(type(Tensor(b)))
c = a.clip(min_=0, max_=float("inf")).log().numpy()
print(c)
print(a.numpy())