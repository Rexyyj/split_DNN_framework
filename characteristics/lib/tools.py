import numpy as np
import torch
import tensorly as tl
import torch_dct as dct

def raise_error_when_not_numpy(value):
    if isinstance(value,np.ndarray) == False:
        raise Exception("Input not numpy array!")

def calculate_snr(tensor_size, original_tensor, reconstructed_tensor):
    raise_error_when_not_numpy(original_tensor)
    raise_error_when_not_numpy(reconstructed_tensor)
    
    original_signal = original_tensor.reshape(tensor_size)
    reconstructed_signal = reconstructed_tensor.reshape(tensor_size)

    # Calculate the noise signal
    noise_signal = original_signal - reconstructed_signal
    
    # Calculate the power of the original signal
    P_signal = np.mean(np.square(original_signal))
    
    # Calculate the power of the noise signal
    P_noise = np.mean(np.square(noise_signal))
    
    # Calculate SNR in dB
    SNR_dB = 10 * np.log10(P_signal / P_noise)
    
    return SNR_dB

def calculate_mse(original_tensor, reconstructed_tensor):
    raise_error_when_not_numpy(original_tensor)
    raise_error_when_not_numpy(reconstructed_tensor)
    tl.set_backend('numpy')
    error = tl.metrics.MSE(original_tensor, reconstructed_tensor)
    return error

def calculate_sparsity(tensor):
    raise_error_when_not_numpy(tensor)
    tensor_size = tensor.shape[0]* tensor.shape[1] * tensor.shape[2]
    zero_mask = tensor==0
    zero_num = np.sum(zero_mask)
    sparsity = zero_num / tensor_size
    return sparsity

def calculate_slice_avg_rank(tensor):
    rank = 0
    for i in range(tensor.shape[0]):
        rank +=torch.linalg.matrix_rank(tensor[i]).item()
    avg_rank = rank / tensor.shape[0]
    return avg_rank


def calculate_cp_rank(tensor, tol=1e-3, max_rank=50):
    raise_error_when_not_numpy(tensor)
    errors = []
    for rank in range(1, max_rank + 1):
        try:
            # print("Testing rank: "+str(rank))
            # Perform CP decomposition
            weights, factors = tl.decomposition.parafac(tensor, rank=rank,verbose=False)
            # Reconstruct the tensor from the CP decomposition
            reconstructed_tensor = tl.cp_to_tensor((weights, factors))
            # Calculate the reconstruction error
            error = tl.metrics.MSE(tensor, reconstructed_tensor)
            errors.append((rank, error))
            # print("Error:"+str(error))
            # Check if the error falls below the specified tolerance
            if error < tol:
                return rank
        except Exception:
            continue
    return max_rank  # Return the maximum rank if no suitable rank is found

# def calculate_cp_rank_fast(tensor, tol=1e-3, max_rank=30):
#     raise_error_when_not_numpy(tensor)
#     rank_previous=0
#     rank = max_rank
#     while True:
#         try:
#             print("Testing rank: "+str(rank))
#             # Perform CP decomposition
#             weights, factors = tl.decomposition.parafac(tensor, rank=rank,verbose=False)
#             # Reconstruct the tensor from the CP decomposition
#             reconstructed_tensor = tl.cp_to_tensor((weights, factors))
#             # Calculate the reconstruction error
#             error = tl.metrics.MSE(tensor, reconstructed_tensor)
#             # print("Error:"+str(error))
#             # Check if the error falls below the specified tolerance
#             if rank_previous == rank:
#                 break
#             else:
#                 if error > tol:
#                     temp = rank
#                     rank = round((rank+rank_previous)/2)
#                     rank_previous = temp
#                 else:

#                     rank_previous = rank
#                     rank = round(rank/2)
#                     if rank == 0:
#                         rank =1
                
#         except Exception:
#             continue
#     return rank  # Return the maximum rank if no suitable rank is found

def get_probability_tensor(tensor):
    # normalize_base =torch.abs(tensor).max()
    # quantized_tensor = torch.floor(((tensor / normalize_base +1)/2)*256 )
    tensor_min = tensor.min().item()
    tensor_max = tensor.max().item()
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    quantized_tensor = torch.floor(normalized_tensor *255)
    unique_values, counts = torch.unique(quantized_tensor, return_counts=True)
    probabilities = counts.float() / quantized_tensor.numel()  # Probability of each unique value

    # Step 5: Create a mapping of quantized values to probabilities
    value_to_prob = {val.item(): prob.item() for val, prob in zip(unique_values, probabilities)}

    # Step 6: Map the probabilities back to the quantized tensor
    probability_tensor = quantized_tensor.clone()
    for val in unique_values:
        probability_tensor[quantized_tensor == val] = value_to_prob[val.item()]
    return probability_tensor

def quantize_tensor(tensor):
    normalize_base =torch.abs(tensor).max()
    tensor_q = torch.floor(((tensor / normalize_base +1)/2)*256 )
    return tensor_q

def get_tensor_pictoriality(tensor):
    dct_tensor = dct.dct_2d(tensor)
    # entropy = torch.sum(torch.special.entr(get_probability_tensor(dct_tensor)))
    entropy = calculate_entropy_float_tensor(dct_tensor)
    return entropy/8

def get_tensor_regularity(tensor):
    # fft_tensor = torch.fft.fft(tensor[tensor!=0])
    # fft_tensor = dct.dct(tensor[tensor!=0])
    # fft_tensor = torch.fft.fft(tensor)
    # entropy = torch.sum(torch.special.entr(get_probability_tensor(tensor[tensor!=0])))
    entropy = calculate_entropy_float_tensor(tensor)
    return entropy/8

# def get_tensor_entropy(tensor):
#     entropy = torch.sum(torch.special.entr(get_probability_tensor(tensor)))
#     return entropy/5.544

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
