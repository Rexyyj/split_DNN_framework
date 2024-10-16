
import numpy as np
import torch
import simplejpeg
import pickle

import tensorly as tl
import torch_dct as dct

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

def calculate_snr(original_signal, reconstructed_signal): # Require 1-D signal in numpy
    # Calculate the noise signal
    noise_signal = original_signal - reconstructed_signal
    
    # Calculate the power of the original signal
    P_signal = np.mean(np.square(original_signal))
    
    # Calculate the power of the noise signal
    P_noise = np.mean(np.square(noise_signal))
    
    # Calculate SNR in dB
    SNR_dB = 10 * np.log10(P_signal / P_noise)
    
    return SNR_dB