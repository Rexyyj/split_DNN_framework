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
        self.fc1 = nn.Linear(input_size, 256)  # First layer
        self.fcs = nn.ModuleList([nn.Linear(256, 256) for _ in range(5)])  # Correct way to add layers in a loop


        self.fc_out = nn.Linear(256, output_size)  # Output layer
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
        self.input_size = 43264
        self.output_size = 3
        self.encoding_model_path = '../split_framework/ckpts/l1_loss_norm_large.pth'
        # Load model
        self.encode_model = RegressionNN(self.input_size, self.output_size)
        self.encode_model.load_state_dict(torch.load(self.encoding_model_path))
        self.encode_model.eval().to('cuda')
        index_base = np.arange(0,43264)/43264
        index = torch.zeros(64,43264, device="cuda")
        for i in range(64):
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
    
    def encode_consecutive_bool_tensor(self,bool_tensor):
    
        # Convert the boolean tensor to a tensor of integers (0 and 1)
        int_tensor = bool_tensor.to(torch.int)
        
        # Compute the differences between consecutive elements
        diff_tensor = torch.diff(int_tensor, prepend=int_tensor[:1])
        
        # Find positions where the value changes
        change_positions = torch.cat((torch.tensor([0],device="cuda"), torch.nonzero(diff_tensor, as_tuple=True)[0], torch.tensor([int_tensor.size(0)],device="cuda")))
        
        # Calculate the lengths of consecutive runs
        run_lengths = torch.diff(change_positions)
        
        # Identify run lengths that need to be split
        over_limit_mask = run_lengths > 255
        num_full_chunks = torch.div(run_lengths, 255, rounding_mode='floor')
        remainder_chunks = run_lengths % 255

        # Build the final tensor with correct sizes
        full_chunks = torch.repeat_interleave(torch.tensor([255], dtype=torch.uint8,device="cuda"), num_full_chunks.sum().item())
        remainders = remainder_chunks[remainder_chunks > 0].to(torch.uint8)

        # Concatenate all parts
        final_lengths = torch.cat((full_chunks, remainders))

        return bool_tensor[0].cpu().item(),final_lengths

    
    def merge_255_values(self,tensor):
        # Find positions of 255 values
        is_255 = tensor == 255
        changes = torch.diff(is_255.int(), prepend=torch.tensor([0],device="cuda"), append=torch.tensor([0],device="cuda"))

        # Start and end positions of consecutive 255 blocks
        start_positions = torch.nonzero(changes == 1, as_tuple=True)[0]
        end_positions = torch.nonzero(changes == -1, as_tuple=True)[0] - 1

        # Compute the cumulative sum for segments with 255 values
        mask = is_255.clone()
        mask[end_positions + 1] = True  # Include the element after the last 255 in the block
        cumsum_tensor = torch.cumsum(tensor * mask, dim=0)
        
        # Create a tensor to store the merged values
        merged_tensor = tensor.clone()

        # Add the cumulative sum of 255 blocks to the element after the last 255
        merged_tensor[end_positions + 1] += cumsum_tensor[end_positions] - cumsum_tensor[start_positions] + tensor[start_positions]

        # Remove the 255 values
        mask[start_positions] = False  # Keep the start of each block
        merged_tensor = merged_tensor[~is_255 | (is_255 & mask)]

        return merged_tensor
    
    def reconstruct_bool_tensor(self,encoded_list, first_value):
       # Convert the encoded list to a PyTorch tensor of uint8

        encoded_tensor = torch.tensor(encoded_list, dtype=torch.uint8,device="cuda")
        
        # Convert encoded_tensor to int32 for repeat_interleave
        encoded_tensor = encoded_tensor.to(torch.int32)
        tensor_len = 0

        while len(encoded_tensor)!=tensor_len:
            encoded_tensor = self.merge_255_values(encoded_tensor)
            tensor_len = len(encoded_tensor)
        
        
        # Create an alternating values tensor based on the first value
        num_segments = len(encoded_tensor)
        values = torch.tensor([first_value], dtype=torch.bool,device="cuda").repeat(num_segments)
        values[1::2] = ~values[1::2]  # Flip every other value
        
        # Create the decoded tensor by repeating each value according to the run lengths
        decoded_tensor = torch.repeat_interleave(values, encoded_tensor)
        
        return decoded_tensor
        

    
    def regression_encode(self,tensor):
        # Normalize & Quantize Config
        tensor_reshaped = tensor.reshape([64,43264])
        coffe = self.encode_model(tensor_reshaped)
        zero_mask = tensor_reshaped==0
        negative_mask = tensor_reshaped<0

        reconstructed_tensor = torch.matmul(self.index_powers, coffe.unsqueeze(2)).squeeze(2)
        reconstructed_tensor[zero_mask] =0
        reconstructed_tensor[negative_mask] = -reconstructed_tensor[negative_mask]
        reconstructed_tensor = reconstructed_tensor.reshape([1,64,208,208])

        zero_init_value,zero_mask_bytes = self.encode_consecutive_bool_tensor(zero_mask.reshape(64*208*208))
        nega_init_value,negative_mask_bytes = self.encode_consecutive_bool_tensor(negative_mask.reshape(64*208*208))

        return zero_init_value,zero_mask_bytes,nega_init_value, negative_mask_bytes, coffe, reconstructed_tensor

    def regression_decode(self, tensor_dict):
        zero_mask = self.reconstruct_bool_tensor(tensor_dict["zero_mask"], tensor_dict["zero_init"])
        zero_mask = zero_mask.reshape([64,43264])
        negative_mask = self.reconstruct_bool_tensor(tensor_dict["negative_mask"], tensor_dict["nega_init"])
        negative_mask = negative_mask.reshape([64,43264])
        reconstructed_tensor = torch.matmul(self.index_powers, tensor_dict["coffe"].unsqueeze(2)).squeeze(2)
        reconstructed_tensor[zero_mask] =0
        reconstructed_tensor[negative_mask] = -reconstructed_tensor[negative_mask]
        reconstructed_tensor = reconstructed_tensor.reshape([1,64,208,208])
        return reconstructed_tensor

    def split_framework_encode(self,id, head_tensor):
        with torch.no_grad():
            # diff operator
            diff_tensor = head_tensor-self.reference_tensor
            self.diff_tensor_sparsity.append(torch.sum(diff_tensor==0).cpu().item()/self.tensor_size)
            # pruner
            diff_tensor_normal = torch.nn.functional.normalize(diff_tensor)
            pruned_tensor = diff_tensor*(abs(diff_tensor_normal) > self.pruning_threshold)

            zero_init_value,zero_mask,nega_init_value, negative_mask,coffe, reconstructed_tensor = self.regression_encode(pruned_tensor)

            payload = {
                "id" : id,
                "zero_init":zero_init_value,
                "zero_mask": zero_mask,
                "nega_init":nega_init_value,
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
