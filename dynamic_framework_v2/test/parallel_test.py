import torch
import multiprocessing
import tensorly as tl
import time
import numpy as np
tl.set_backend('pytorch')

def decomposition_worker(x):
    ft = tl.decomposition.parafac(x, rank=5)
    return ft


tensor = torch.rand([128,26,26],device="cpu")

begin_time = time.time_ns()
for i in range(tensor.shape[0]):
    ft = tl.decomposition.parafac(tensor[i], rank=5)

print((time.time_ns()-begin_time)/1e6)


begin = time.time_ns()
with multiprocessing.Pool(processes=6) as pool:
    results = pool.map(decomposition_worker, tensor)  # Parallel execution

print((time.time_ns()-begin)/1e6)