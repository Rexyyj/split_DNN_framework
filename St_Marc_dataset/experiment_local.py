################################### setting path ###################################
import sys
sys.path.append('../')
################################### import libs ###################################
import cv2
from  pytorchyolo import  models_split_tiny
import numpy as np
import pandas as pd
import time
import torch
import os
from torch import tensor
from split_framework.yolov3_tensor_jpeg_v2 import SplitFramework

import requests
import pickle
import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from terminaltables import AsciiTable
from pytorchyolo.utils.utils import load_classes, ap_per_class, get_batch_statistics, non_max_suppression, to_cpu, xywh2xyxy, print_environment_info
from pytorchyolo.utils.datasets import ListDataset
from pytorchyolo.utils.transforms import DEFAULT_TRANSFORMS

################################### Varialbe init ###################################
testdata_path = "./data/test_0.txt"
class_name_path = "./data/coco.names"
log_dir = "./measurements/30_fps/"
N_warmup = 0

test_case = "tensor"
service_uri = "http://10.0.1.34:8090/tensor"
reset_uri = "http://10.0.1.34:8090/reset"


measurement_path = log_dir+"/"
map_output_path = measurement_path+ "map.csv"
time_output_path = measurement_path+ "time.csv"

# Note: model split layer should -1 for the actual split point

################################### Utility functions ###################################
def create_data_loader(data_path):
    dataset = ListDataset(data_path, img_size=416, multiscale=False, transform=DEFAULT_TRANSFORMS)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=dataset.collate_fn)
    return dataloader


def print_eval_stats(metrics_output, class_names, verbose):
    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output
        if verbose:
            # Prints class AP and mean AP
            ap_table = [["Index", "Class", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
        print(f"---- mAP {AP.mean():.5f} ----")
    else:
        print("---- mAP not measured (no detections found by model) ----")
################################### Main function ###################################

if __name__ == "__main__":
    # Load Model
    model = models_split_tiny.load_model("../pytorchyolo/config/yolov3-tiny.cfg","./ckpt/yolov3_ckpt_300.pth")
    model.set_split_layer(7) # layer <7
    model = model.eval()
    
    dataloader = create_data_loader(testdata_path)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    class_names = load_classes(class_name_path)  # List of class names
    for j in range(1):
        for i in range(1):
            
            frame_predicts = []
            # thresh = 0.02*(j+1)
            # quality =60+10*i

            time_start = torch.cuda.Event(enable_timing=True)
            time_end = torch.cuda.Event(enable_timing=True)
            ################## Init measurement lists ##########################
            transfer_data_size =[]

            head_time =[]
            tail_time =[]
            encode_time=[]
            decode_time=[]
            request_time=[]
            framework_time=[]
            jpeg_time = []


            labels = []
            sample_metrics = []  # List of tuples (TP, confs, pred)
            frame_index = 0
            for _, imgs, targets in tqdm.tqdm(dataloader, desc="testing"):
                frame_index+=1
                
                # Real measurements
                # Extract labels
                labels += targets[:, 1].tolist()
                # Rescale target
                targets[:, 2:] = xywh2xyxy(targets[:, 2:])
                targets[:, 2:] *= 416

                with torch.no_grad():
                    ##### Head Model #####

                    head_tensor = model(imgs.cuda(), 1)
                    inference_result = model(head_tensor,2)
                    detection = non_max_suppression(inference_result, 0.2, 0.5)
                  
                # print(detection)
                sample_metrics += get_batch_statistics(detection, targets, iou_threshold=0.1)
        
        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [
            np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        metrics_output = ap_per_class(
            true_positives, pred_scores, pred_labels, labels)

        print_eval_stats(metrics_output, class_names, True)




            # with open(map_output_path,'a') as f:
            #     f.write(str(thresh)+","
            #             +str(quality)+","
            #             +str(np.array(transfer_data_size).mean())+","
            #             +str(np.array(transfer_data_size).std())+","
            #             +str(maps["map"].item())+","
            #             +str(maps["map_50"].item())+","
            #             +str(maps["map_75"].item())+","
            #             +str(maps["map_small"].item())+","
            #             +str(maps["map_medium"].item())+","
            #             +str(maps["map_large"].item())+","
            #             +str(maps["mar_1"].item())+","
            #             +str(maps["mar_100"].item())+","
            #             +str(maps["mar_small"].item())+","
            #             +str(maps["mar_medium"].item())+","
            #             +str(maps["mar_large"].item())+"\n"
            #             )
                
            # with open(time_output_path,'a') as f:
            #     f.write(str(thresh)+","
            #             +str(quality)+","
            #             +str(np.array(head_time).mean())+","
            #             +str(np.array(head_time).std())+","
            #             +str(np.array(tail_time).mean())+","
            #             +str(np.array(tail_time).std())+","
            #             +str(np.array(framework_time).mean())+","
            #             +str(np.array(framework_time).std())+","
            #             +str(np.array(jpeg_time).mean())+","
            #             +str(np.array(jpeg_time).std())+","
            #             +str(np.array(encode_time).mean())+","
            #             +str(np.array(encode_time).std())+","
            #             +str(np.array(decode_time).mean())+","
            #             +str(np.array(decode_time).std())+","
            #             +str(np.array(request_time).mean())+","
            #             +str(np.array(request_time).std())+"\n"
            #     )
                