import numpy as np
import simplejpeg
import torch


def compressor_jpeg(tensor, quality):
        # Normalize & Quantize Config
        normalize_base =torch.abs(tensor).max()
        tensor_normal = tensor / normalize_base
        scale, zero_point = (tensor_normal.max()-tensor_normal.min())/255,127
        dtype = torch.quint8

        tensor_normal = torch.quantize_per_tensor(tensor_normal, scale, zero_point, dtype)
        tensor_normal = tensor_normal.int_repr()
        tensor_normal = tensor_normal.to(torch.uint8)

       # JPEG encoding/decoding
        encoded_data = simplejpeg.encode_jpeg(tensor_normal.cpu().numpy().astype(np.uint8),quality,'RGB')
        compressed_size = len(encoded_data) + 12

        return normalize_base, scale,zero_point, encoded_data, compressed_size # compressed size in bytes


def compressor_jpeg_direct(tensor, quality):

       # JPEG encoding/decoding
        encoded_data = simplejpeg.encode_jpeg(tensor.cpu().numpy().astype(np.uint8),quality,'RGB')
        compressed_size = len(encoded_data)

        return encoded_data, compressed_size # compressed size in bytes

def decompressor_jpeg(tensor_shape,normalize_base, scale,zero_point, encoded_data):
    decoded_data =decoded_data = torch.from_numpy(simplejpeg.decode_jpeg(encoded_data,"RGB"))
    reconstructed_tensor = decoded_data.reshape(tensor_shape)
    reconstructed_tensor = (reconstructed_tensor.to(torch.float)-zero_point) * scale * normalize_base
    return reconstructed_tensor

def decompressor_jpeg_direct(tensor_shape, encoded_data):
    decoded_data =decoded_data = torch.from_numpy(simplejpeg.decode_jpeg(encoded_data,"RGB"))
    reconstructed_tensor = decoded_data.reshape(tensor_shape)
    return reconstructed_tensor