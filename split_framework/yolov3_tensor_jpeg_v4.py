################################### setting path ###################################
import sys
sys.path.append('../')
################################### import libs ###################################
import numpy as np
import torch
from nvjpeg import NvJpeg
import pickle
import torch.nn.functional as F
################################### class definition ###################################

class SplitFramework():

    def __init__(self,device) -> None:
        self.device = device
        self.reference_tensor=None
        self.tensor_size= None
        self.tensor_shape = None
        self.pruning_threshold= None
        self.pedding = (0, 0, 0, 3328)

        # Measurements
        self.diff_tensor_sparsity = []
        self.pruned_tensor_sparsity = []
        self.pruned_tensor_rank =[]
        self.data_size = []
        self.reconstruct_error = []
        self.head_time =[]
        self.framework_time = []
        self.tail_time = []
        self.time_start = torch.cuda.Event(enable_timing=True)
        self.time_end = torch.cuda.Event(enable_timing=True)
        self.nj =  NvJpeg()

    def set_reference_tensor(self, head_tensor):
        self.tensor_shape = head_tensor.shape
        self.reference_tensor = torch.zeros( self.tensor_shape, dtype=torch.float32).to(self.device)
        self.reference_tensor_edge = torch.zeros( self.tensor_shape, dtype=torch.float32).to(self.device)
        self.tensor_size = self.tensor_shape[0]*self.tensor_shape[1]*self.tensor_shape[2]*self.tensor_shape[3]

    def set_pruning_threshold(self, threshold):
        self.pruning_threshold = threshold
    
    def set_jpeg_quality(self, quality):
        self.jpeg_quality =  quality

    def calculate_snr(self,original_signal, reconstructed_signal):
        # Calculate the noise signal
        noise_signal = original_signal - reconstructed_signal
        
        # Calculate the power of the original signal
        P_signal = np.mean(np.square(original_signal))
        
        # Calculate the power of the noise signal
        P_noise = np.mean(np.square(noise_signal))
        
        # Calculate SNR in dB
        SNR_dB = 10 * np.log10(P_signal / P_noise)
        
        return SNR_dB
    
    def jpeg_encode(self,tensor):
        # Normalize & Quantize Config
        normalize_base =torch.abs(tensor).max()
        tensor_normal = tensor / normalize_base
        scale, zero_point = (tensor_normal.max()-tensor_normal.min())/255,127
        dtype = torch.quint8

        tensor_normal = torch.quantize_per_tensor(tensor_normal, scale, zero_point, dtype)
        tensor_normal = tensor_normal.int_repr()
        tensor_normal = tensor_normal.to(torch.uint8).reshape((self.tensor_shape[1]*self.tensor_shape[2]*self.tensor_shape[3],1))
        tensor_padded = F.pad(tensor_normal, self.pedding)
        tensor_padded = tensor_padded.reshape((128,234,3))
        # JPEG encoding/decoding
        encoded_data = self.nj.encode(tensor_padded.cpu().numpy().astype(np.uint8),self.jpeg_quality)
        # self.data_size.append(transfer_data.shape[0])
        decoded_data = torch.from_numpy(self.nj.decode(encoded_data)).to(self.device)
        decoded_data = decoded_data.reshape((decoded_data.shape[0]*decoded_data.shape[1]*decoded_data.shape[2],1))
        decoded_data = decoded_data[0:128*26*26]
        # # Reconstruct diff tensor
        reconstructed_tensor = decoded_data.reshape(self.tensor_shape)
        reconstructed_tensor = (reconstructed_tensor.to(torch.float)-zero_point) * scale * normalize_base
        # snr = self.calculate_snr(reconstructed_tensor.reshape(self.tensor_size).cpu().numpy() , tensor.reshape(self.tensor_size).cpu().numpy())
        # self.reconstruct_error.append(snr)
        return normalize_base, scale,zero_point, encoded_data, reconstructed_tensor

    def jpeg_decode(self, tensor_dict):
        decoded_data  = torch.from_numpy(self.nj.decode(tensor_dict["encoded"])).to(self.device)
        decoded_data = decoded_data.reshape((decoded_data.shape[0]*decoded_data.shape[1]*decoded_data.shape[2],1))
        decoded_data = decoded_data[0:128*26*26]
        reconstructed_tensor = decoded_data.reshape(self.tensor_shape)
        reconstructed_tensor = (reconstructed_tensor.to(torch.float)-tensor_dict["zero"]) * tensor_dict["scale"] * tensor_dict["normal"]
        return reconstructed_tensor

    def split_framework_encode(self,id, head_tensor):
        with torch.no_grad():
            # self.time_start.record()
            # diff operator
            diff_tensor = head_tensor-self.reference_tensor
            self.diff_tensor_sparsity.append(torch.sum(diff_tensor==0).cpu().item()/self.tensor_size)
            # pruner
            diff_tensor_normal = torch.nn.functional.normalize(diff_tensor)
            pruned_tensor = diff_tensor*(abs(diff_tensor_normal) > self.pruning_threshold)
            # torch.cuda.synchronize()
            # self.time_end.record()
            operation_time = 0# self.time_start.elapsed_time(self.time_end)

            ## Jpeg encoding
            # self.time_start.record()
            normalize_base, scale,zero_point, encoded_data, reconstructed_tensor = self.jpeg_encode(pruned_tensor)
            # torch.cuda.synchronize()
            # self.time_end.record()
            jpeg_encoding_time = 0#self.time_start.elapsed_time(self.time_end)

            payload = {
                "id" : id,
                "normal": normalize_base,
                "scale":scale,
                "zero": zero_point,
                "encoded": encoded_data
            }
            # updte the reference tensor
            self.reference_tensor = self.reference_tensor + reconstructed_tensor

        return operation_time,jpeg_encoding_time,pickle.dumps(payload)

    def split_framework_decode(self,tensor_dict):
        reconstructed_tensor = self.jpeg_decode(tensor_dict)
        reconstructed_head_tensor = self.reference_tensor + reconstructed_tensor
        self.reference_tensor = reconstructed_head_tensor
        return reconstructed_head_tensor
