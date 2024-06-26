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
from split_framework.yolov3_tensor_jpeg import SplitFramework
import requests
import pickle
from torchmetrics.detection import MeanAveragePrecision
from torch.profiler import profile, record_function, ProfilerActivity
################################### Varialbe init ###################################
video_path = "../dataset/test/"
label_path = "../dataset/test_label/"
video_files = os.listdir(video_path)
video_names = [name.replace('.mov','') for name in video_files]
N_frame = 105
N_warmup = 5

test_case = "frame_local"

log_dir = "../measurements/"
measurement_path = log_dir+test_case+"/"
map_output_path = measurement_path+ "map.csv"
time_output_path = measurement_path+ "time.csv"

model_split_layer = 7
dummy_head_tensor = torch.rand([1,128,26,26])
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
    title = ("video_name,"
            "map,"
            "map_50,"
            "map_75,"
            "map_small,"
            "map_medium,"
            "map_large,"
            "mar_1,"
            "mar_100,"
            "mar_small,"
            "mar_medium,"
            "mar_large\n")
    f.write(title)

with open(time_output_path,'a') as f:
    title = ("video_name,"
            "model_time_mean,"
            "model_time_std\n"
            )
    f.write(title)

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

def load_ground_truth(video_name):
    frame_labels = []
    df_gt = pd.read_csv(label_path+video_name+".csv")
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

def load_video_frames(video_dir, video_name, samples_number=-1): #samples_number = -1 to load all frames
    video_path = video_dir +video_name +".mov"

    test_frames = []

    cap = cv2.VideoCapture(video_path) # origin frame rate is 30 fps
    exist_flag = True
    counter = 0
    while exist_flag and (samples_number==-1 or counter<samples_number):
        ret, frame = cap.read()
        if ret:
            test_frames.append(cv2.resize(frame,(416,416),interpolation = cv2.INTER_AREA))
            counter+=1
        
        if ret:
            exist_flag = True
        else:
            exist_flag = False
        
    cap.release()

    print("Load total number of video frames: ", len(test_frames))
    return test_frames

################################### Main function ###################################

if __name__ == "__main__":
    # Load Model
    model = models_split_tiny.load_model("../pytorchyolo/config/yolov3-tiny.cfg","../pytorchyolo/weights/yolov3-tiny.weights")
    model.set_split_layer(model_split_layer) # layer <7
    model = model.eval()
    
    # Load videos
    for video_name in video_names:
        print("Testing with video: "+video_name)
        test_frames = load_video_frames(video_path,video_name, N_frame)
        frame_labels = load_ground_truth(video_name)

        for i in range(1):
            print("In iter",i)
            reset_required = True

            frame_predicts = []


            time_start = torch.cuda.Event(enable_timing=True)
            time_end = torch.cuda.Event(enable_timing=True)
            ################## Init measurement lists ##########################
            cpu_time_client = []
            # cuda_time =[]
            cpu_mem_client = []
            cuda_mem_client = []

            model_time =[]

            #####################################################################
            for index in range(len(test_frames)):

                inWarmup = True
                if index+1 > N_warmup:
                    inWarmup = False
                
                frame = test_frames[index]
                ################## Perform Object detection #############################
                with torch.no_grad():

                    #####  Warmup phase #####
                    if inWarmup:
                        frame_tensor = convert_rgb_frame_to_tensor(frame)
                        head_tensor = model(frame_tensor, 1)
                        inference_result = model(head_tensor,2)
                        detection = non_max_suppression(inference_result, 0.5, 0.5)
                        continue
                    #####  Warmup phase #####
                    ##### Model #####
                    time_start.record()
                    frame_tensor = convert_rgb_frame_to_tensor(frame)
                    head_tensor = model(frame_tensor, 1)
                    inference_result = model(head_tensor,2)
                    detection = non_max_suppression(inference_result, 0.5, 0.5)
                    time_end.record()
                    torch.cuda.synchronize()
                    model_time.append(time_start.elapsed_time(time_end))
                    ##### Model #####
                        
                ##################### Collect resource usage ##########################
                

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
            metric.update(frame_predicts, frame_labels[N_warmup:N_frame])
            maps = metric.compute()

            with open(map_output_path,'a') as f:
                f.write(video_name+","
                        +str(maps["map"].item())+","
                        +str(maps["map_50"].item())+","
                        +str(maps["map_75"].item())+","
                        +str(maps["map_small"].item())+","
                        +str(maps["map_medium"].item())+","
                        +str(maps["map_large"].item())+","
                        +str(maps["mar_1"].item())+","
                        +str(maps["mar_100"].item())+","
                        +str(maps["mar_small"].item())+","
                        +str(maps["mar_medium"].item())+","
                        +str(maps["mar_large"].item())+"\n"
                        )
                
            with open(time_output_path,'a') as f:
                f.write(video_name+","
                        +str(np.array(model_time).mean())+","
                        +str(np.array(model_time).std())+"\n"
                )