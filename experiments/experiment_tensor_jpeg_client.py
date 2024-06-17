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
################################### Varialbe init ###################################
video_path = "../dataset/test/"
label_path = "../dataset/test_label/"
video_files = os.listdir(video_path)
video_names = [name.replace('.mov','') for name in video_files]
N_frame = 10

test_case = "tensor_jpeg"
service_path = "http://10.0.1.23:8090/jpeg_tensor"

log_dir = "../measurements/"
measurement_path = log_dir+test_case+"/"
map_output_path = measurement_path+ "map.csv"
perf_output_path = measurement_path+ "time.csv"
resource_output_path = measurement_path+"resource.csv"

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
    video_path = video_dir +video_name

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

def object_detection(id, model, frame,framework, tail_model_address):
    with torch.no_grad():
        # Execute head model
    
        tensor = convert_rgb_frame_to_tensor(frame)
        head_tensor = model(tensor, 1)
        data_to_trans = framework.split_framework_encode(id, head_tensor)

        r = requests.post(url=tail_model_address, data=data_to_trans)
        result = pickle.loads(r.content)
        
    return result["detection"]
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
            frame_predicts = []
            thresh = 0.05
            quality = 60+10*i

            sf = SplitFramework(device="cuda")
            sf.set_reference_tensor(dummy_head_tensor)
            sf.set_pruning_threshold(thresh)
            sf.set_jpeg_quality(quality)

            for index in range(len(test_frames)):
                frame = test_frames[index]
                frame = convert_rgb_frame_to_tensor(frame)
                detection = object_detection(frame)
                print(detection)