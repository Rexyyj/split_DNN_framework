import torch
import numpy as np

def compressor_regression(tensor, polynominal):
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
            m = np.polyfit(x,y,polynominal)
            factors.append(m)
            x_pos.append(mask)
            x_neg.append(t<0)
            compressed_size = compressed_size+ tensor.shape[1]*tensor.shape[2]/4 + polynominal*4
        
    return factors, x_pos, x_neg, compressed_size


def decompressor_regression(tensor_shape,factors, x_pos, x_neg):
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
    return reconstructed_tensor