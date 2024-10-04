################################### setting path ###################################
import sys
sys.path.append('../')
################################### import libs ###################################
import numpy as np
import torch
import simplejpeg
import pickle

import tensorly as tl
import torch_dct as dct
################################### class definition ###################################

class SplitFramework():

    def __init__(self,device) -> None:
        self.device = device
        self.reference_tensor=None
        self.tensor_size= None
        self.tensor_shape = None
        self.pruning_threshold= None

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
        self._sparsity = None
        self._decomposability = None
        self._pictoriality =None
        self._regularity = None

    def set_reference_tensor(self, head_tensor):
        self.tensor_shape = head_tensor.shape
        self.reference_tensor = torch.zeros( self.tensor_shape, dtype=torch.float32).to(self.device)
        self.reference_tensor_edge = torch.zeros( self.tensor_shape, dtype=torch.float32).to(self.device)
        self.tensor_size = self.tensor_shape[0]*self.tensor_shape[1]*self.tensor_shape[2]*self.tensor_shape[3]

    def set_pruning_threshold(self, threshold):
        self.pruning_threshold = threshold
    
    def set_quality(self, quality):
        self.quality =  quality

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
        tensor_normal = tensor_normal.to(torch.uint8).reshape((self.tensor_shape[1],self.tensor_shape[2]*self.tensor_shape[3],1))

        # JPEG encoding/decoding
        encoded_data = simplejpeg.encode_jpeg(tensor_normal.cpu().numpy().astype(np.uint8),self.quality,'GRAY')
        # self.data_size.append(transfer_data.shape[0])
        decoded_data = torch.from_numpy(simplejpeg.decode_jpeg(encoded_data,"GRAY")).to(self.device)
        # # Reconstruct diff tensor
        reconstructed_tensor = decoded_data.reshape(self.tensor_shape)
        reconstructed_tensor = (reconstructed_tensor.to(torch.float)-zero_point) * scale * normalize_base
        # snr = self.calculate_snr(reconstructed_tensor.reshape(self.tensor_size).cpu().numpy() , tensor.reshape(self.tensor_size).cpu().numpy())
        # self.reconstruct_error.append(snr)
        return normalize_base, scale,zero_point, encoded_data, reconstructed_tensor


    def jpeg_decode(self, tensor_dict):
        decoded_data = torch.from_numpy(simplejpeg.decode_jpeg(tensor_dict["encoded"],"GRAY")).to(self.device)
        reconstructed_tensor = decoded_data.reshape(self.tensor_shape)
        reconstructed_tensor = (reconstructed_tensor.to(torch.float)-tensor_dict["zero"]) * tensor_dict["scale"] * tensor_dict["normal"]
        return reconstructed_tensor

    def split_framework_encode(self,id, head_tensor, characteristic=False):
        with torch.no_grad():
            self.time_start.record()
            # diff operator
            diff_tensor = head_tensor-self.reference_tensor
            self.diff_tensor_sparsity.append(torch.sum(diff_tensor==0).cpu().item()/self.tensor_size)
            # pruner
            diff_tensor_normal = torch.nn.functional.normalize(diff_tensor)
            pruned_tensor = diff_tensor*(abs(diff_tensor_normal) > self.pruning_threshold)

            self.time_end.record()
            torch.cuda.synchronize()
            operation_time = self.time_start.elapsed_time(self.time_end)

            if characteristic == True:
                self._sparsity = calculate_sparsity(pruned_tensor[0].cpu().numpy())
                self._decomposability = get_tensor_decomposability(pruned_tensor[0])
                self._pictoriality=get_tensor_pictoriality(pruned_tensor[0])
                self._regularity = get_tensor_regularity(pruned_tensor[0])

            ## Jpeg encoding
            self.time_start.record()
            normalize_base, scale,zero_point, encoded_data, reconstructed_tensor = self.jpeg_encode(pruned_tensor)
            self.time_end.record()
            torch.cuda.synchronize()
            jpeg_encoding_time = self.time_start.elapsed_time(self.time_end)

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

    def get_tensor_characteristics(self):
        return self._sparsity, self._decomposability, self._regularity, self._pictoriality
    

####################### tool functions ##############################
def raise_error_when_not_numpy(value):
    if isinstance(value,np.ndarray) == False:
        raise Exception("Input not numpy array!")

def calculate_sparsity(tensor):
    raise_error_when_not_numpy(tensor)
    tensor_size = tensor.shape[0]* tensor.shape[1] * tensor.shape[2]
    zero_mask = tensor==0
    zero_num = np.sum(zero_mask)
    sparsity = zero_num / tensor_size
    return sparsity

# def get_tensor_pictoriality(tensor):
#     ents =[]
#     for i in range(tensor.shape[0]):
#         try:
#             entropy=calculate_entropy_float_tensor(dct.dct_2d(tensor[i]))
#             ents.append(1-entropy/8)
#         except:
#             pass
#     ents = np.array(ents)

#     return ents.mean() 

def get_tensor_pictoriality(tensor):
    dct_tensor = dct.dct_2d(tensor.reshape(tensor.shape[0], tensor.shape[1]*tensor.shape[2]))
    # entropy = torch.sum(torch.special.entr(get_probability_tensor(dct_tensor)))
    entropy = calculate_entropy_float_tensor(dct_tensor)
    return 1-entropy/8

def get_tensor_regularity(tensor):
    ents =[]
    for i in range(tensor.shape[0]):
        t = tensor[i][tensor[i]!=0]
        try:
            entropy=calculate_entropy_float_tensor(dct.dct(t))
            ents.append(1-entropy/8)
        except:
            pass
    ents = np.array(ents)

    return ents.mean() 

def get_tensor_decomposability(tensor):
    tensor_rank = calculate_slice_avg_rank(tensor)
    d = 1-(tensor_rank*tensor.shape[1]+tensor_rank*tensor.shape[2])*tensor.shape[0]/(tensor.numel())
    return d


def calculate_entropy_float_tensor(tensor):
    # Step 1: Flatten the tensor to 1D
    flattened_tensor = tensor.flatten().cpu().numpy()

    # Step 2: Discretize the values by creating a histogram with `num_bins` bins
    hist, bin_edges = np.histogram(flattened_tensor, bins=256, density=True)

    # Step 3: Calculate probabilities from the histogram
    probabilities = hist / np.sum(hist)  # Normalize to sum to 1

    # Step 4: Compute entropy
    probabilities = probabilities[probabilities > 0]  # Remove zero probabilities

    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy

def calculate_slice_avg_rank(tensor):
    rank = 0
    for i in range(tensor.shape[0]):
        rank +=torch.linalg.matrix_rank(tensor[i]).item()
    avg_rank = rank / tensor.shape[0]
    return avg_rank
