import numpy as np
import torch
import tensorly as tl

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
    error = tl.metrics.MSE(original_tensor, reconstructed_tensor)
    return error



def find_cp_rank(tensor, tol=1e-3, max_rank=50):
    raise_error_when_not_numpy(tensor)
    errors = []
    for rank in range(1, max_rank + 1):
        try:
            print("Testing rank: "+str(rank))
            # Perform CP decomposition
            weights, factors = tl.decomposition.parafac(tensor, rank=rank,verbose=False)
            # Reconstruct the tensor from the CP decomposition
            reconstructed_tensor = tl.cp_to_tensor((weights, factors))
            # Calculate the reconstruction error
            error = tl.metrics.MSE(tensor, reconstructed_tensor)
            errors.append((rank, error))
            print("Error:"+str(error))
            # Check if the error falls below the specified tolerance
            if error < tol:
                return rank, errors
        except Exception:
            continue
    return max_rank, errors  # Return the maximum rank if no suitable rank is found