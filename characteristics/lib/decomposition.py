import tensorly as tl
import torch


def compressor_decomposition(tensor, rank):
    factors = tl.decomposition.parafac(tensor.numpy(),rank=rank)
    compressed_size = 0
    compressed_size+=factors.factors[0].shape[0]*factors.factors[0].shape[1]
    compressed_size+=factors.factors[1].shape[0]*factors.factors[1].shape[1]
    compressed_size+=factors.factors[2].shape[0]*factors.factors[2].shape[1]
    compressed_size = compressed_size*4
    return factors, compressed_size # compressed size in bytes

def decompressor_decomposition(factors):
    reconstructed_tensor = tl.cp_to_tensor(factors)
    return reconstructed_tensor