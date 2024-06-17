################################### setting path ###################################
import sys
sys.path.append('../')
################################### import libs ###################################
import cv2
from  pytorchyolo import detect, models_split_tiny
from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS
import torchvision.transforms as transforms
import numpy as np
from pytorchyolo.utils.utils import load_classes, rescale_boxes, non_max_suppression, print_environment_info
import pandas as pd
import time
import torch
import torchvision.ops.boxes as bops
import matplotlib.pyplot as plt
from torchvision import io as tvio
import os

################################### Varialbe init ###################################
video_path = "../dataset/test/"
video_files = os.listdir(video_path)
video_names = [name.replace('.mov','') for name in video_files]
N_frame = 100

test_case = "tensor_jpeg"

log_dir = "../measurements/"
measurement_path = log_dir+test_case+"/"
map_output_path = measurement_path+ "map.csv"
perf_output_path = measurement_path+ "time.csv"
resource_output_path = measurement_path+"resource.csv"

################################### Clean Old Logs ###################################
try:
    path = os.path.join(log_dir,test_case)
    os.mkdir(path)
except:
    os.system("rm -rf "+measurement_path)
    time.sleep(3)
    path = os.path.join(log_dir,test_case)
    os.mkdir(path)
        

with open(map_output_path,'a') as f:
    f.write("video_name,pruning_thresh,jepg_quality,map,map_50,map_75,map_small,map_medium, ap_large,mar_1,mar_100,mar_small,mar_medium,mar_large\n")

with open(perf_output_path,'a') as f:
    f.write("video_name,pruning_thresh,jepg_quality,snr_mean,snr_std,head_time_mean,head_time_std,framework_time_mean,framework_time_std,tail_time_mean,tail_time_std,data_size_mean,data_size_std\n")
with open(resource_output_path,'a') as f:
    f.write("video_name,pruning_thresh,jepg_quality,cpu_total_mean,cpu_total_std,cuda_total_mean,cuda_total_std,cpu_mem_mean,cpu_mem_std,cuda_mem_mean,cuda_mem_std\n")
################################### Utility functions ###################################
def convert_rgb_frame_to_tensor(image):
    img_size = 416
    # Configure input
    input_img = transforms.Compose([
    DEFAULT_TRANSFORMS,
    Resize(img_size)])(
        (image, np.zeros((1, 5))))[0].unsqueeze(0)
    input_img = input_img.cuda()

    return input_img

################################### Main function ###################################