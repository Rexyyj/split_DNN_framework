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
import os
from torch import tensor
from split_framework.yolov3_tensor_jpeg_v2 import SplitFramework

import requests
import pickle
from torchmetrics.detection import MeanAveragePrecision
from torch.profiler import profile, record_function, ProfilerActivity
################################### Varialbe init ###################################
frame_path = "./frames/30_fps/"
label_path = "./labels/30_fps.csv"
log_dir = "./measurements/30_fps/"
# video_files = os.listdir(video_path)
# video_names = [name.replace('.mov','') for name in video_files]
N_frame = 50
N_warmup = 5

test_case = "tensor_jpeg"


measurement_path = log_dir+"/"
map_output_path = measurement_path+ "map.csv"
time_output_path = measurement_path+ "time.csv"



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

def load_ground_truth(gt_path):
    frame_labels = []
    df_gt = pd.read_csv(gt_path)
    gt_group = df_gt.groupby("frame_id")
    for key in gt_group.groups.keys():
        df = gt_group.get_group(key)
        labels = []
        boxes = []
        for i in range(len(df)):
            row = df.iloc[i].to_list()
            boxes.append(row[4:])
            labels.append(row[2])

        frame_labels.append(dict(boxes=tensor(boxes,dtype=torch.float32),labels=tensor(labels,dtype=torch.int32),) )
    return frame_labels

def sort_func(name):
    num = name.split(".")[0].split("_")[1]
    return int(num)

def load_video_frames(frame_dir, samples_number=-1): #samples_number = -1 to load all frames
    files = os.listdir(frame_dir)
    files.sort(key=sort_func)

    test_frames = []
    if samples_number==-1:
        for test_file in files:
            frame = cv2.imread(frame_dir+test_file)
            frame = cv2.resize(frame,(416,416),interpolation = cv2.INTER_AREA)
            test_frames.append(frame)
    else:
        for test_file in files[0:samples_number]:
            frame = cv2.imread(frame_dir+test_file)
            frame = cv2.resize(frame,(416,416),interpolation = cv2.INTER_AREA)
            test_frames.append(frame)

    print("Load total number of video frames: ", len(test_frames))
    return test_frames

################################### Main function ###################################

if __name__ == "__main__":
    # Load Model
    model = models_split_tiny.load_model("../pytorchyolo/config/yolov3-tiny.cfg","../pytorchyolo/weights/yolov3-tiny.weights")
    model.set_split_layer(7) # layer <7
    model = model.eval()
    
    # Load videos
    test_frames = load_video_frames(frame_path, N_frame)
    frame_labels = load_ground_truth(label_path)

            
    frame_predicts = []
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
    #####################################################################
    for index in range(len(test_frames)):
        frame = test_frames[index]

        with torch.no_grad():
            ##### Head Model #####
            time_start.record()
            frame_tensor = convert_rgb_frame_to_tensor(frame)
            head_tensor = model(frame_tensor, 1)
            inference_result = model(head_tensor,2)
            detection = non_max_suppression(inference_result, 0.5, 0.5)
            time_end.record()
            torch.cuda.synchronize()
            head_time.append(time_start.elapsed_time(time_end))
            ##### Head Model #####
    
            print(detection)
            if len(detection[0])!= 0:
                pred = dict(boxes=tensor(detection[0].numpy()[:,0:4]),
                        scores=tensor(detection[0].numpy()[:,4]),
                        labels=tensor(detection[0].numpy()[:,5],dtype=torch.int32), )
            else:
                pred = dict(boxes=tensor([]),
                        scores=tensor([]),
                        labels=tensor([],dtype=torch.int32),)
            frame_predicts.append(pred)
    metric = MeanAveragePrecision(iou_type="bbox") 
    metric.update(frame_predicts, frame_labels[0:N_frame])
    maps = metric.compute()
    print(maps)
        
