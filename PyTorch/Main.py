import random

import torch
import numpy as np


dd = np.random.randint(10, size=(3, 4))
t = torch.tensor(dd)
print(t.dtype)
print(t.shape)
print(t.device)
print(t.layout)
