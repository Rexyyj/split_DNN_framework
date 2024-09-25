import tensorly as tl
import torch


def compressor_decomposition(tensor, rank):
    factors = tl.decomposition.parafac(tensor.numpy(),rank=rank)
    compressed_size = 0
    compressed_size+=(factors.factors[0].shape[0]*factors.factors[0].shape[1])*4
    compressed_size+=(factors.factors[1].shape[0]*factors.factors[1].shape[1])*4
    compressed_size+=(factors.factors[2].shape[0]*factors.factors[2].shape[1])*4
    compressed_size = compressed_size*4
    return factors, compressed_size # compressed size in bytes

def decompressor_decomposition(factors):
    reconstructed_tensor = tl.cp_to_tensor(factors)
    return reconstructed_tensor

def compressor_decomposition_slice(tensor):
    tl.set_backend('pytorch')
    factors= []
    compressed_size = 0
    for i in range(tensor.shape[0]):
        if torch.linalg.matrix_rank(tensor[i]).item() == 0:
            factors.append(0)
        else:
            rank = int(torch.linalg.matrix_rank(tensor[i]))
            ft = tl.decomposition.parafac(tensor[i], rank=rank)
            factors.append(ft)
            compressed_size += (ft.factors[0].shape[0]*ft.factors[0].shape[1])*4
            compressed_size += (ft.factors[1].shape[0]*ft.factors[1].shape[1])*4
    return factors, compressed_size

def decompressor_decomposition_slice(tensor_shape,factors):
    # reconstruct
    tl.set_backend('pytorch')
    reconstructed_tensor = torch.zeros(tensor_shape)
    for i in range(tensor_shape[0]):
        reconstructed_tensor[i] = tl.cp_to_tensor(factors[i])
    return reconstructed_tensor