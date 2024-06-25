################################### setting path ###################################
import sys
sys.path.append('../')
################################### import libs ###################################
import numpy as np
import torch
import simplejpeg
import pickle
import torch.nn as nn
################################### class definition ###################################

class RegressionNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(RegressionNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # First layer
        self.fcs = nn.ModuleList([nn.Linear(128, 128) for _ in range(5)])  # Correct way to add layers in a loop


        self.fc_out = nn.Linear(128, output_size)  # Output layer
        self.relu = nn.ReLU()  # Activation function

    def forward(self, x):
        x = self.relu(self.fc1(x))
        for layer in self.fcs:  # Iterating over registered modules
            x = self.relu(layer(x))  # Applying non-linearity after each hidden layer
        x = self.fc_out(x)
        return x

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

        # Regression params
        self.input_size = 676
        self.output_size = 3
        self.encoding_model_path = '../split_framework/ckpts/l1_loss_norm.pth'
        # Load model
        self.encode_model = RegressionNN(self.input_size, self.output_size)
        self.encode_model.load_state_dict(torch.load(self.encoding_model_path))
        self.encode_model.eval().to('cuda')
        index_base = np.arange(0,676)/676
        index = torch.zeros(128,676, device="cuda")
        for i in range(128):
            index[i] = torch.from_numpy(index_base)
        powers = torch.arange(3 - 1, -1, -1, device="cuda")
        self.index_powers = index.unsqueeze(2) ** powers  # Shape: [128, 676, 3]


    def set_reference_tensor(self, head_tensor):
        self.tensor_shape = head_tensor.shape
        self.reference_tensor = torch.zeros( self.tensor_shape, dtype=torch.float32).to(self.device)
        self.reference_tensor_edge = torch.zeros( self.tensor_shape, dtype=torch.float32).to(self.device)
        self.tensor_size = self.tensor_shape[0]*self.tensor_shape[1]*self.tensor_shape[2]*self.tensor_shape[3]

    def set_pruning_threshold(self, threshold):
        self.pruning_threshold = threshold
    

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
    
    def encode_bool_tensor_to_byte(self, tensor):
        tensor = tensor.reshape(128*26*26)
        numpy_array = tensor.cpu().numpy().astype(np.bool_)
        # Pack the NumPy array into bytes
        encoded_bytes = np.packbits(numpy_array).tobytes()
        return encoded_bytes
    
    def decode_byte_to_bool_tensor(self,encoded_bytes: bytes, length: int):
        # Decode the bytes back to a NumPy array
        unpacked_array = np.unpackbits(np.frombuffer(encoded_bytes, dtype=np.uint8))
        # Truncate the array to the original length (if necessary)
        unpacked_array = unpacked_array[:length]
        # Convert the NumPy array back to a PyTorch boolean tensor
        decoded_tensor = torch.tensor(unpacked_array, dtype=torch.bool)
        
        return decoded_tensor

    
    def regression_encode(self,tensor):
        # Normalize & Quantize Config
        tensor_reshaped = tensor.reshape([128,676])
        coffe = self.encode_model(tensor_reshaped)
        zero_mask = tensor_reshaped==0
        negative_mask = tensor_reshaped<0

        reconstructed_tensor = torch.matmul(self.index_powers, coffe.unsqueeze(2)).squeeze(2)
        reconstructed_tensor[zero_mask] =0
        reconstructed_tensor[negative_mask] = -reconstructed_tensor[negative_mask]
        reconstructed_tensor = reconstructed_tensor.reshape([1,128,26,26])

        zero_mask_bytes = self.encode_bool_tensor_to_byte(zero_mask.reshape(128*26*26))
        negative_mask_bytes = self.encode_bool_tensor_to_byte(negative_mask.reshape(128*26*26))

        return zero_mask_bytes, negative_mask_bytes, coffe, reconstructed_tensor

    def regression_decode(self, tensor_dict):
        zero_mask = self.decode_byte_to_bool_tensor(tensor_dict["zero_mask"], 128*26*26)
        zero_mask = zero_mask.reshape([128,676])
        negative_mask = self.decode_byte_to_bool_tensor(tensor_dict["negative_mask"], 128*26*26)
        negative_mask = negative_mask.reshape([128,676])
        reconstructed_tensor = torch.matmul(self.index_powers, tensor_dict["coffe"].unsqueeze(2)).squeeze(2)
        reconstructed_tensor[zero_mask] =0
        reconstructed_tensor[negative_mask] = -reconstructed_tensor[negative_mask]
        reconstructed_tensor = reconstructed_tensor.reshape([1,128,26,26])
        return reconstructed_tensor

    def split_framework_encode(self,id, head_tensor):
        with torch.no_grad():
            # diff operator
            diff_tensor = head_tensor-self.reference_tensor
            self.diff_tensor_sparsity.append(torch.sum(diff_tensor==0).cpu().item()/self.tensor_size)
            # pruner
            diff_tensor_normal = torch.nn.functional.normalize(diff_tensor)
            pruned_tensor = diff_tensor*(abs(diff_tensor_normal) > self.pruning_threshold)

            zero_mask, negative_mask,coffe, reconstructed_tensor = self.regression_encode(pruned_tensor)

            payload = {
                "id" : id,
                "zero_mask": zero_mask,
                "negative_mask":negative_mask,
                "coffe": coffe,
            }
            # updte the reference tensor
            self.reference_tensor = self.reference_tensor + reconstructed_tensor

        return pickle.dumps(payload)

    def split_framework_decode(self,tensor_dict):
        reconstructed_tensor = self.regression_decode(tensor_dict)
        reconstructed_head_tensor = self.reference_tensor + reconstructed_tensor
        self.reference_tensor = reconstructed_head_tensor
        return reconstructed_head_tensor
