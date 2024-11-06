################################### setting path ###################################
import sys
sys.path.append('./')
################################### import libs ###################################
import numpy as np
import torch
import pickle
import requests
from split_framework.stable.tools import *
from pytorchyolo.utils.utils import non_max_suppression
################################### Define version ###################################
__COLLECT_TENSOR_CHARACTERISTIC__ = False
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
        tensor = tensor.cpu()
        factors= []
        x_pos = []
        x_neg = []
        x_raw = np.arange(0,tensor.shape[1]*tensor.shape[2])
        compressed_size =0
        for i in range(tensor.shape[0]):
            if torch.linalg.matrix_rank(tensor[i]).item() == 0:
                factors.append([])
                x_pos.append(0)
                x_neg.append(0)
            else:
                t = tensor[i].reshape(tensor.shape[1]*tensor.shape[2])
                mask = t!=0
                y = abs(t[mask])
                x=x_raw[mask]
                m = np.polyfit(x,y,self.quality)
                factors.append(m)
                x_pos.append(mask)
                x_neg.append(t<0)
                # compressed_size +=  tensor.shape[1]*tensor.shape[2]/4 + self.quality*4
                compressed_size += len(t[mask])
        reconstructed_tensor = self.decompressor(tensor.shape,factors, x_pos, x_neg)
        return factors, x_pos, x_neg, compressed_size,reconstructed_tensor


    def decompressor(self,tensor_shape,factors, x_pos, x_neg):
        x_raw = np.arange(0,tensor_shape[1]*tensor_shape[2])
        reconstructed_tensor = torch.zeros(tensor_shape, device="cpu")
        for i in range(tensor_shape[0]):
            if len(factors[i]) != 0:
                recon = np.zeros((tensor_shape[1]*tensor_shape[2]))
                # print(factors[i])
                # y_pred = factors[i].predict(PolynomialFeatures(degree=self.regression_factor, include_bias=False).fit_transform( x_raw[x_pos[i]].reshape(-1,1)))
                y_pred = np.polyval(factors[i],x_raw[x_pos[i]])
                recon[x_pos[i]] = y_pred
                recon[x_raw[x_neg[i]]] = -recon[x_raw[x_neg[i]]]
                reconstructed_tensor[i] = torch.from_numpy(recon.reshape((tensor_shape[1],tensor_shape[2])))
        return reconstructed_tensor.reshape(self.tensor_shape).cuda()

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
            try:
                self._sparsity = calculate_sparsity(pruned_tensor[0].cpu().numpy())
                self._decomposability = get_tensor_decomposability(pruned_tensor[0])
                self._pictoriality=get_tensor_pictoriality(pruned_tensor[0])
                self._regularity = get_tensor_regularity(pruned_tensor[0])
            except:
                self._sparsity =-1
                self._decomposability=-1
                self._pictoriality=-1
                self._regularity=-1

        if __COLLECT_FRAMEWORK_TIME__:
            self.time_start.record()
            factors, x_pos, x_neg,compressed_size,reconstructed_tensor = self.compressor(pruned_tensor[0])
            self.time_end.record()
            torch.cuda.synchronize()
            self._compression_time = self.time_start.elapsed_time(self.time_end)
            self.time_start.record()
            payload = {
                "factors": factors,
                "x_pos":x_pos,
                "x_neg": x_neg,
                "tensor_shape":head_tensor[0].shape
            }
            # updte the reference tensor
            self.reference_tensor = self.reference_tensor + reconstructed_tensor
            self.time_end.record()
            torch.cuda.synchronize()
            self._framework_head_time+=self.time_start.elapsed_time(self.time_end)
        else:
            factors, x_pos, x_neg,compressed_size,reconstructed_tensor = self.compressor(pruned_tensor[0])
            payload = {
                "factors": factors,
                "x_pos":x_pos,
                "x_neg": x_neg,
                "tensor_shape":head_tensor[0].shape
            }
            # updte the reference tensor
            self.reference_tensor = self.reference_tensor + reconstructed_tensor

        if __COLLECT_TENSOR_RECONSTRUCT__:
            self._reconstruct_snr = calculate_snr(head_tensor.reshape(self.tensor_size).cpu().numpy(),self.reference_tensor.reshape(self.tensor_size).cpu().numpy())

        request_payload = pickle.dumps(payload)
        self._datasize_est = compressed_size
        self._datasize_real = len(request_payload)
        return request_payload
    
    def split_framework_decode(self,tensor_dict):
        if __COLLECT_FRAMEWORK_TIME__:
            self.time_start.record()
            reconstructed_tensor = self.decompressor(tensor_dict["tensor_shape"],tensor_dict["factors"],tensor_dict["x_pos"],tensor_dict["x_neg"])
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
            reconstructed_tensor = self.decompressor(tensor_dict["tensor_shape"],tensor_dict["factors"],tensor_dict["x_pos"],tensor_dict["x_neg"])
            reconstructed_head_tensor = self.reference_tensor + reconstructed_tensor
            self.reference_tensor = reconstructed_head_tensor
        return reconstructed_head_tensor
    
    def split_framework_client(self, frame_tensor, service_uri):
        try:
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
        except:
            self._datasize_est=-1
            self._datasize_real=-1
            self._overall_time = -1
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
            return []

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


