################################### setting path ###################################
import sys
sys.path.append('./')
################################### import libs ###################################
import numpy as np
import torch
# -*- coding: utf-8 -*-
import hashlib
# import array
import numpy as np
from dahuffman import HuffmanCodec
import datasketches as dsk
from numba import jit
import math
import torch
from split_framework.stable.tools import *
import pickle
import requests
from pytorchyolo.utils.utils import  non_max_suppression
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
        self.sketch_q =  quality[0]
        self.sketch_m =  quality[1]
        self.sketch_d =  quality[2]
    
    def get_bucket_index(self,tensor, n_buckets):
        tensor = tensor.reshape(tensor.shape[0]* tensor.shape[1]* tensor.shape[2])
        values = tensor[tensor !=0].cpu().numpy()
        keys = tensor.nonzero(as_tuple=True)[0].cpu().numpy()

        sketch = dsk.kll_floats_sketch()
        sketch.update(values)

        n_splits = n_buckets
        n_quantiles = []
        for i in range(1, n_splits):
            n_quantiles.append(i/n_splits)
        quantiles = sketch.get_quantiles(n_quantiles)

        bucket_index = np.zeros(len(values),dtype=np.uint8)
        bucket_means = np.zeros(n_splits)
        for k in range(n_splits):
            if k == 0:
                bucket_mask = values < quantiles[k] 
            elif k< (n_splits-1):
                bucket_mask = (values<quantiles[k]) ^ (values<quantiles[k-1])
            else:
                bucket_mask = values > quantiles[k-1]
            if len(values[bucket_mask]) == 0:
                bucket_means[k] = 0
            else:
                bucket_means[k] = values[bucket_mask].mean()
            bucket_index[bucket_mask] = k

        return bucket_means, bucket_index, keys

    def compressor(self,tensor):
         # Quantile-Bucket Quantification
        bucket_means, bucket_index,keys = self.get_bucket_index(tensor,self.sketch_q)
        
        m = int(len(bucket_index)*self.sketch_m)

        # MinMaxSketch insert
        minmaxsketch =  MinMaxSketchV2(m,self.sketch_d)
        minmaxsketch.add_array(keys, bucket_index)
        
        # Encoding
        encode_key = []
        for i in range(self.sketch_q):
            encode_key.append(i)
        encode_key.append(255)
        encoder = HuffmanCodec.from_data(encode_key)

        compressed_size = 0
        for i in range(len(minmaxsketch.tables)):
            compressed_size+=len(encoder.encode(minmaxsketch.tables[i].tolist()))
        compressed_size = compressed_size+len(bucket_means)*4+len(keys)
        reconstructed_tensor = self.decompressor(tensor.shape,minmaxsketch,bucket_means, keys)
        return minmaxsketch, bucket_means,keys, compressed_size, reconstructed_tensor

    def decompressor(self, tensor_shape,minmaxsketch,bucket_means, keys):
        reconstructed_bucket_index =  minmaxsketch.query_array(keys)
        # Tensor reconstruction
        reconstructed_tensor = np.zeros(tensor_shape[0]*tensor_shape[1]*tensor_shape[2], dtype=np.float32)
        reconstructed_tensor[keys] = bucket_means[reconstructed_bucket_index]
        reconstructed_tensor = torch.from_numpy(reconstructed_tensor).reshape(tensor_shape)
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
            self._sparsity = calculate_sparsity(pruned_tensor[0].cpu().numpy())
            self._decomposability = get_tensor_decomposability(pruned_tensor[0])
            self._pictoriality=get_tensor_pictoriality(pruned_tensor[0])
            self._regularity = get_tensor_regularity(pruned_tensor[0])

        if __COLLECT_FRAMEWORK_TIME__:
            self.time_start.record()
            minmaxsketch, bucket_means,keys, compressed_size, reconstructed_tensor = self.compressor(pruned_tensor[0])
            self.time_end.record()
            torch.cuda.synchronize()
            self._compression_time = self.time_start.elapsed_time(self.time_end)
            self.time_start.record()
            payload = {
                "minmaxsketch": minmaxsketch,
                "bucket_means":bucket_means,
                "keys":keys,
                "tensor_shape":pruned_tensor[0].shape
            }
            # updte the reference tensor
            self.reference_tensor = self.reference_tensor + reconstructed_tensor
            self.time_end.record()
            torch.cuda.synchronize()
            self._framework_head_time+=self.time_start.elapsed_time(self.time_end)
        else:
            minmaxsketch, bucket_means,keys, compressed_size, reconstructed_tensor = self.compressor(pruned_tensor[0])
            payload = {
                "minmaxsketch": minmaxsketch,
                "bucket_means":bucket_means,
                "keys":keys,
                "tensor_shape":pruned_tensor[0].shape
            }
            # updte the reference tensor
            self.reference_tensor = self.reference_tensor + reconstructed_tensor

        if __COLLECT_TENSOR_RECONSTRUCT__:
            self._reconstruct_snr = calculate_snr(reconstructed_tensor.reshape(self.tensor_size).cpu().numpy() , head_tensor.reshape(self.tensor_size).cpu().numpy())
        request_payload = pickle.dumps(payload)
        self._datasize_est = compressed_size
        self._datasize_real = len(request_payload)
        return request_payload
    
    def split_framework_decode(self,tensor_dict):
        if __COLLECT_FRAMEWORK_TIME__:
            self.time_start.record()
            reconstructed_tensor = self.decompressor(tensor_dict["tensor_shape"],tensor_dict["minmaxsketch"],tensor_dict["bucket_means"],tensor_dict["keys"])
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
            reconstructed_tensor =  self.decompressor(tensor_dict["tensor_shape"],tensor_dict["minmaxsketch"],tensor_dict["bucket_means"],tensor_dict["keys"])
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
        except Exception as error:
            print(error)
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


######################### SketchML IMPL###########################

"""
Original CountMinSketch Implementaiton: https://github.com/rafacarrascosa/countminsketch/tree/master
"""
class MinMaxSketchV2(object):
    """
    A class for counting hashable items using the Count-min Sketch strategy.
    It fulfills a similar purpose than `itertools.Counter`.
    The Count-min Sketch is a randomized data structure that uses a constant
    amount of memory and has constant insertion and lookup times at the cost
    of an arbitrarily small overestimation of the counts.
    It has two parameters:
     - `m` the size of the hash tables, larger implies smaller overestimation
     - `d` the amount of hash tables, larger implies lower probability of
           overestimation.
    An example usage:
        from countminsketch import CountMinSketch
        sketch = CountMinSketch(1000, 10)  # m=1000, d=10
        sketch.add("oh yeah")
        sketch.add(tuple())
        sketch.add(1, value=123)
        print sketch["oh yeah"]       # prints 1
        print sketch[tuple()]         # prints 1
        print sketch[1]               # prints 123
        print sketch["non-existent"]  # prints 0
    Note that this class can be used to count *any* hashable type, so it's
    possible to "count apples" and then "ask for oranges". Validation is up to
    the user.
    """

    def __init__(self, m, d):
        """ `m` is the size of the hash tables, larger implies smaller
        overestimation. `d` the amount of hash tables, larger implies lower
        probability of overestimation.
        """
        if not m or not d:
            raise ValueError("Table size (m) and amount of hash functions (d)"
                             " must be non-zero")
        self.m = m
        self.d = d
        self.n = 0
        self.tables = []
        self.tables = np.full([self.d,self.m],128,dtype=np.int8)
        # for _ in range(d):
        #     # table = array.array("l", (999 for _ in range(m)))
        #     # self.tables.append(table)
        #     table = np.full(m,255,dtype=np.uint8)
        #     self.tables.append(table)

    def _hash(self, x):
        md5 = hashlib.md5(str(hash(x)).encode())
        for i in range(self.d):
            md5.update(str(i).encode())
            yield int(md5.hexdigest(), 16) % self.m

    # def add(self, x, value=1):
    #     """
    #     Count element `x` as if had appeared `value` times.
    #     By default `value=1` so:
    #         sketch.add(x)
    #     Effectively counts `x` as occurring once.
    #     """
    #     self.n += value
    #     for table, i in zip(self.tables, self._hash(x)):
    #         table[i] += value
    def update_table (self, table):
        self.tables = table

    def reset_table(self):
        self.tables = np.full([self.d,self.m],255,dtype=np.int32)

    def add(self,key,value):
        results = np.zeros(self.d, dtype=np.int32)
        self._hash2(key,self.d, self.m,results)
        for table, code in zip(self.tables, results):
            if table[code] > value:
                table[code] = value

    @staticmethod
    @jit(nopython=True)
    def _hash2(x, d, m,result):
        var = str(x)
        for i in range(d):
            var_hash = hash(var)
            var = str(var_hash)
            result[i] =  int(var_hash) % m

    # @jit(nopython=True)
    @staticmethod
    @jit(nopython=True)
    def _add_array(keys, values, tables, d, m):
        for key, value in zip(keys,values):
            var = str(key)
            for i in range(d):
                var_hash = hash(var)
                var = str(var_hash)
                code =  int(var_hash) % m
                if tables[i][code] > value:
                    tables[i][code] = value



            
    def add_array(self,keys,values):
        self._add_array(keys,values,self.tables,self.d,self.m)
        

    @staticmethod
    @jit(nopython=True)
    def _query_array(keys, tables, d, m,results):
        for k in range(len(keys)):
            key = keys[k]
            max_val = 0
            var = str(key)
            for i in range(d):
                var_hash = hash(var)
                var = str(var_hash)
                code =  int(var_hash) % m
                if tables[i][code] > max_val:
                    max_val = tables[i][code]
            results[k] =  max_val
           


    def query_array(self, keys):
        result = np.zeros(len(keys),dtype = np.int32)
        self._query_array(keys,self.tables,self.d,self.m,result)
        return result
    # def query(self, key):
    #     return max(table[i] for table, i in zip(self.tables, self._hash(key)))
    
    def query(self, key):
        result = np.zeros(self.d,dtype = np.int32)
        self._hash2(key,self.d,self.m,result)
        return max(table[i] for table, i in zip(self.tables, result))

    # def query(self, x):
    #     """
    #     Return an estimation of the amount of times `x` has ocurred.
    #     The returned value always overestimates the real value.
    #     """
    #     return min(table[i] for table, i in zip(self.tables, self._hash(x)))

    def __getitem__(self, x):
        """
        A convenience method to call `query`.
        """
        return self.query(x)

    def __len__(self):
        """
        The amount of things counted. Takes into account that the `value`
        argument of `add` might be different from 1.
        """
        return self.n



