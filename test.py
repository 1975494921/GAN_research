import torch
import numpy as np

a = torch.randn(2, 3, 4)

# get the shape of the tensor

a_shape = a.shape

# change the shape tensor to numpy array

a_np = np.array(a_shape)

print(a_np)