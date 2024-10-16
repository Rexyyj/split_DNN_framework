################################### setting path ###################################
# import sys
# sys.path.append('./')
################################### import libs ###################################
import numpy as np
import torch
import simplejpeg
import pickle
import requests
from split_framework.stable.tools import *
from pytorchyolo.utils.utils import non_max_suppression
################################### Define version ###################################
__COLLECT_TENSOR_CHARACTERISTIC__ = True
__COLLECT_TENSOR_RECONSTRUCT__ = True
__COLLECT_FRAMEWORK_TIME__ = True
__COLLECT_OVERALL_TIME__ = True
################################### class definition ###################################
class SplitFramework():

    def __init__(self,model, device) -> None:
        self.device = device
        self.reference_tensor=None
        self.tensor_size= None
        self.tensor_shape = None
        self.pruning_threshold= None
        self.model = model

        # Measurements
        self._datasize_est=None
        self._datasize_real=None
        self._overall_time = -1
        self.time_start = torch.cuda.Event(enable_timing=True)
        self.time_end = torch.cuda.Event(enable_timing=True)
        self._model_head_time = -1
        self._model_tail_time = -1
        self._compression_time = -1
        self._decompression_time = -1
        self._framework_head_time = -1
        self._framework_tail_time = -1
        self._framework_response_time = -1
        self._sparsity = -1
        self._decomposability = -1
        self._pictoriality =-1
        self._regularity = -1
        self._reconstruct_snr = -1
        
    def set_reference_tensor(self, head_tensor):
        self.tensor_shape = head_tensor.shape
        self.reference_tensor = torch.zeros( self.tensor_shape, dtype=torch.float32).to(self.device)
        self.reference_tensor_edge = torch.zeros( self.tensor_shape, dtype=torch.float32).to(self.device)
        self.tensor_size = self.tensor_shape[0]*self.tensor_shape[1]*self.tensor_shape[2]*self.tensor_shape[3]

    def set_pruning_threshold(self, threshold):
        self.pruning_threshold = threshold
    
    def set_quality(self, quality):
        self.quality =  quality
    
    def compressor(self,tensor):
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


    def decompressor(self, tensor_dict):
        decoded_data = torch.from_numpy(simplejpeg.decode_jpeg(tensor_dict["encoded"],"GRAY")).to(self.device)
        reconstructed_tensor = decoded_data.reshape(self.tensor_shape)
        reconstructed_tensor = (reconstructed_tensor.to(torch.float)-tensor_dict["zero"]) * tensor_dict["scale"] * tensor_dict["normal"]
        return reconstructed_tensor

    def split_framework_encode(self, head_tensor):
        if __COLLECT_FRAMEWORK_TIME__:
            self.time_start.record()
            # Framework Head #
            diff_tensor = head_tensor-self.reference_tensor
            diff_tensor_normal = torch.nn.functional.normalize(diff_tensor)
            pruned_tensor = diff_tensor*(abs(diff_tensor_normal) > self.pruning_threshold)
            # Framework Head #
            self.time_end.record()
            torch.cuda.synchronize()
            self._framework_head_time = self.time_start.elapsed_time(self.time_end)
        else:
            # Framework Head #
            diff_tensor = head_tensor-self.reference_tensor
            diff_tensor_normal = torch.nn.functional.normalize(diff_tensor)
            pruned_tensor = diff_tensor*(abs(diff_tensor_normal) > self.pruning_threshold)
            # Framework Head #

        if __COLLECT_TENSOR_CHARACTERISTIC__:
            self._sparsity = calculate_sparsity(pruned_tensor[0].cpu().numpy())
            self._decomposability = get_tensor_decomposability(pruned_tensor[0])
            self._pictoriality=get_tensor_pictoriality(pruned_tensor[0])
            self._regularity = get_tensor_regularity(pruned_tensor[0])

        if __COLLECT_FRAMEWORK_TIME__:
            self.time_start.record()
            normalize_base, scale,zero_point, encoded_data, reconstructed_tensor = self.compressor(pruned_tensor)
            self.time_end.record()
            torch.cuda.synchronize()
            self._compression_time = self.time_start.elapsed_time(self.time_end)
            self.time_start.record()
            payload = {
                "normal": normalize_base,
                "scale":scale,
                "zero": zero_point,
                "encoded": encoded_data
            }
            # updte the reference tensor
            self.reference_tensor = self.reference_tensor + reconstructed_tensor
            self.time_end.record()
            torch.cuda.synchronize()
            self._framework_head_time+=self.time_start.elapsed_time(self.time_end)
        else:
            normalize_base, scale,zero_point, encoded_data, reconstructed_tensor = self.compressor(pruned_tensor)
            payload = {
                "normal": normalize_base,
                "scale":scale,
                "zero": zero_point,
                "encoded": encoded_data
            }
            # updte the reference tensor
            self.reference_tensor = self.reference_tensor + reconstructed_tensor

        if __COLLECT_TENSOR_RECONSTRUCT__:
            self._reconstruct_snr = calculate_snr(reconstructed_tensor.reshape(self.tensor_size).cpu().numpy() , head_tensor.reshape(self.tensor_size).cpu().numpy())

        request_payload = pickle.dumps(payload)
        self._datasize_est = len(encoded_data)
        self._datasize_real = len(request_payload)

        return request_payload
    
    def split_framework_decode(self,tensor_dict):
        if __COLLECT_FRAMEWORK_TIME__:
            self.time_start.record()
            reconstructed_tensor = self.decompressor(tensor_dict)
            self.time_end.record()
            torch.cuda.synchronize()
            self._decompression_time = self.time_start.elapsed_time(self.time_end)

            self.time_start.record()
            reconstructed_head_tensor = self.reference_tensor + reconstructed_tensor
            self.reference_tensor = reconstructed_head_tensor
            self.time_end.record()
            torch.cuda.synchronize()
            self._framework_tail_time = self.time_start.elapsed_time(self.time_end)
        else:
            reconstructed_tensor = self.decompressor(tensor_dict)
            reconstructed_head_tensor = self.reference_tensor + reconstructed_tensor
            self.reference_tensor = reconstructed_head_tensor
        return reconstructed_head_tensor
    
    def split_framework_client(self, frame_tensor, service_uri):
        with torch.no_grad():
            if __COLLECT_OVERALL_TIME__:
                overall_start = torch.cuda.Event(enable_timing=True)
                overall_end = torch.cuda.Event(enable_timing=True)
                overall_start.record()

            if __COLLECT_FRAMEWORK_TIME__:
                self.time_start.record()
                head_tensor = self.model(frame_tensor, 1)
                self.time_end.record()
                torch.cuda.synchronize()
                self._model_head_time = self.time_start.elapsed_time(self.time_end)
                data_to_trans = self.split_framework_encode(head_tensor)
                self.time_start.record()
                r = requests.post(url=service_uri, data=data_to_trans)
                response = pickle.loads(r.content)
                self.time_end.record()
                torch.cuda.synchronize()
                self._framework_response_time = self.time_start.elapsed_time(self.time_end)
            else:
                head_tensor = self.model(frame_tensor, 1)
                data_to_trans = self.split_framework_encode(head_tensor)
                r = requests.post(url=service_uri, data=data_to_trans)
                response = pickle.loads(r.content)

            if __COLLECT_OVERALL_TIME__:
                overall_end.record()
                torch.cuda.synchronize()
                self._overall_time = overall_start.elapsed_time(overall_end)
            
            self._framework_tail_time = response["fw_tail_time"]
            self._model_tail_time = response["model_tail_time"]
            self._decompression_time = response["decmp_time"]

            return response["detection"]

    def split_framework_service(self, compressed_data):
        with torch.no_grad():
            if __COLLECT_FRAMEWORK_TIME__:
                self.time_start.record()
                reconstructed_head_tensor = self.split_framework_decode(compressed_data)
                self.time_end.record()
                torch.cuda.synchronize()
                self._decompression_time = self.time_start.elapsed_time(self.time_end)
                
                ######## Framework decode ##########
                self.time_start.record()
                inference_result = self.model(reconstructed_head_tensor,2)
                detection = non_max_suppression(inference_result, 0.01, 0.5)
                self.time_end.record()
                torch.cuda.synchronize()
                self._model_tail_time = self.time_start.elapsed_time(self.time_end)
                
                self.time_start.record()
                inference = { 
                            "model_tail_time":self._model_tail_time,
                            "fw_tail_time": self._framework_tail_time,
                            "decmp_time":self._decompression_time,
                            "detection":detection }
                service_return = pickle.dumps(inference)
                self.time_end.record()
                torch.cuda.synchronize()
                self._framework_tail_time = self.time_start.elapsed_time(self.time_end)
            else:
                reconstructed_head_tensor = self.split_framework_decode(compressed_data)
                inference_result = self.model(reconstructed_head_tensor,2)
                detection = non_max_suppression(inference_result, 0.01, 0.5)
                inference = { 
                            "model_tail_time":-1,
                            "fw_tail_time": -1,
                            "decmp_time":-1,
                            "detection":detection }
                service_return = pickle.dumps(inference)
            return service_return
 

    def get_tensor_characteristics(self):
        return self._sparsity, self._decomposability, self._regularity, self._pictoriality
    
    def get_model_time_measurement(self):
        return self._model_head_time, self._model_tail_time

    def get_framework_time_measurement(self):
        return self._framework_head_time, self._framework_tail_time, self._framework_response_time
    
    def get_compression_time_measurement(self):
        return self._compression_time, self._decompression_time
    
    def get_overall_time_measurement(self):
        return self._overall_time

    def get_reconstruct_snr(self):
        return self._reconstruct_snr
    
    def get_data_size(self):
        return self._datasize_est, self._datasize_real


