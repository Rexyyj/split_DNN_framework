# -*- coding: utf-8 -*-
import hashlib
# import array
import numpy as np
from dahuffman import HuffmanCodec
import datasketches as dsk
from numba import jit
import math
import torch

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
        self.tables = np.full([self.d,self.m],255,dtype=np.int32)
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


def get_bucket_index(tensor, n_buckets):
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
            bucket_means[k] = values[bucket_mask].mean()
            bucket_index[bucket_mask] = k

        return bucket_means, bucket_index, keys


def compressor_sketchml(tensor, sketch_q, sketch_m, sketch_d):
    # Quantile-Bucket Quantification
    bucket_means, bucket_index,keys = get_bucket_index(tensor,sketch_q)
    
    m = int(len(bucket_index)*sketch_m)

    # MinMaxSketch insert
    minmaxsketch =  MinMaxSketchV2(m,sketch_d)
    minmaxsketch.add_array(keys, bucket_index)
    
    # Encoding
    encode_key = []
    for i in range(128):
        encode_key.append(i)
    encode_key.append(255)
    encoder = HuffmanCodec.from_data(encode_key)

    compressed_size = 0
    for i in range(len(minmaxsketch.tables)):
        compressed_size+=len(encoder.encode(minmaxsketch.tables[i].tolist()))+len(bucket_means)*4
    
    return minmaxsketch, bucket_means,keys, compressed_size
    

def decompressor_sketchml(tensor_shape,minmaxsketch,bucket_means, keys):
    # MinMaxSketch query
    reconstructed_bucket_index =  minmaxsketch.query_array(keys)

    # Tensor reconstruction
    reconstructed_tensor = np.zeros(tensor_shape[0]*tensor_shape[1]*tensor_shape[2], dtype=np.float32)
    reconstructed_tensor[keys] = bucket_means[reconstructed_bucket_index]
    reconstructed_tensor = torch.from_numpy(reconstructed_tensor).reshape(tensor_shape)
    return reconstructed_tensor