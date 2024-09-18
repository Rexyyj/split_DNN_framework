################################### setting path ###################################
import sys
sys.path.append('../')
################################### import libs ###################################
import cv2
from  pytorchyolo import detect, models_split_tiny
from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS
import torchvision.transforms as transforms
import numpy as np
from pytorchyolo.utils.utils import load_classes, rescale_boxes, non_max_suppression, print_environment_info,xywh2xyxy_np
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
image_path = "../pytorchyolo/data/custom/images/"
label_path = "../pytorchyolo/data/custom/labels/"
N_warmup = 5
split_layer= int(sys.argv[1])

test_case = "football_tensor_jpeg"
service_uri = "http://10.0.1.23:8090/tensor_jpeg"
reset_uri = "http://10.0.1.23:8090/reset"

log_dir = "../measurements/yolo_tiny_splitpoint/layer_"+str(split_layer)+"/"
measurement_path = log_dir+test_case+"/"
map_output_path = measurement_path+ "map.csv"
time_output_path = measurement_path+ "time.csv"

files = open("./test_files.txt","r")
test_files_raw = files.readlines()
test_files = []
for line in test_files_raw:
    test_files.append(line.replace("\n",""))
# Note: model split layer should -1 for the actual split point
if split_layer==8:
    model_split_layer = 7
    dummy_head_tensor = torch.rand([1, 128, 26, 26])
elif split_layer==7:
    model_split_layer = 6
    dummy_head_tensor = torch.rand([1, 128, 52, 52])
elif split_layer==6:
    model_split_layer = 5
    dummy_head_tensor = torch.rand([1, 64, 52, 52])
elif split_layer==5:
    model_split_layer = 4
    dummy_head_tensor = torch.rand([1, 64, 104, 104])
elif split_layer==4:
    model_split_layer = 3
    dummy_head_tensor = torch.rand([1, 32, 104, 104])
elif split_layer==3:
    model_split_layer = 2
    dummy_head_tensor = torch.rand([1, 32, 208, 208])
elif split_layer==2:
    model_split_layer = 1
    dummy_head_tensor = torch.rand([1, 16, 208, 208])
elif split_layer==1:
    model_split_layer = 0
    dummy_head_tensor = torch.rand([1, 16, 416, 416])
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
            "pruning_thresh,"
            "jepg_quality,"
            "data_size_mean,"
            "data_size_std,"
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
            "pruning_thresh,"
            "jepg_quality,"
            "head_time_mean,"
            "head_time_std,"
            "tail_time_mean,"
            "tail_time_std,"
            "framework_time_mean,"
            "framework_time_std,"
            "jpeg_time_mean,"
            "jpeg_time_std,"
            "encode_time_mean,"
            "encode_time_std,"
            "decode_time_mean,"
            "decode_time_std,"
            "request_time_mean,"
            "request_time_std\n"
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

def load_ground_truth():
    frame_labels = []
    for test_file in test_files:
        gt_path = label_path+test_file
        gt_path = gt_path.replace(".jpg",".txt")
        f = open(gt_path,"r")
        gts = f.readlines()
        labels = []
        boxes = []
        for gt in gts:
            temp = [float(elem) for elem in gt.replace("\n","").split(" ")]
            labels.append(int(temp[0]))
            box = xywh2xyxy_np(np.array(temp[1:]))
            boxes.append(box*416)
        frame_labels.append(dict(boxes=tensor(boxes,dtype=torch.float32),labels=tensor(labels,dtype=torch.int32),) )
    return frame_labels

def load_video_frames(): #samples_number = -1 to load all frames

    test_frames = []
    for test_file in test_files:
        frame = cv2.imread(image_path+test_file)
        frame = cv2.resize(frame,(416,416),interpolation = cv2.INTER_AREA)
        test_frames.append(frame)

    print("Load total number of video frames: ", len(test_frames))
    return test_frames

################################### Main function ###################################

if __name__ == "__main__":
    # Load Model
    model = models_split_tiny.load_model("../pytorchyolo/config/yolov3-tiny.cfg","../pytorchyolo/checkpoints/yolov3_ckpt_300.pth")
    model.set_split_layer(model_split_layer) # layer <7
    model = model.eval()
    video_name = "football"
    # Load videos
    test_frames = load_video_frames()
    frame_labels = load_ground_truth()

    for j in range(10):
        for i in range(5):
            reset_required = True
            while reset_required:
                r = requests.post(url=reset_uri)
                result = pickle.loads(r.content)
                if result["reset_status"] == True:
                    reset_required = False
                else:
                    print("Reset edge reference tensor failed...")
                time.sleep(1)

            
            frame_predicts = []
            thresh = 0.01*(j+1)
            quality =60+10*i
            # thresh = 0.02
            # quality =100
            print("Testing threshold: ",thresh,", Jpeg quality: ",quality)
            sf = SplitFramework(device="cuda")
            sf.set_reference_tensor(dummy_head_tensor)
            sf.set_pruning_threshold(thresh)
            sf.set_jpeg_quality(quality)

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
                        framework_t,jpeg_t,data_to_trans = sf.split_framework_encode(index, head_tensor)
                        r = requests.post(url=service_uri, data=data_to_trans)
                        response = pickle.loads(r.content)
                        continue
                    #####  Warmup phase #####

                    ##### Head Model #####
                    time_start.record()
                    frame_tensor = convert_rgb_frame_to_tensor(frame)
                    head_tensor = model(frame_tensor, 1)
                    time_end.record()
                    torch.cuda.synchronize()
                    head_time.append(time_start.elapsed_time(time_end))
                    ##### Head Model #####
                
                    ##### Framework Encoding #####
                    time_start.record()
                    framework_t, jpeg_t,data_to_trans = sf.split_framework_encode(index, head_tensor)
                    time_end.record()
                    torch.cuda.synchronize()
                    encode_time.append(time_start.elapsed_time(time_end))
                    transfer_data_size.append(len(data_to_trans))
                    framework_time.append(framework_t)
                    jpeg_time.append(jpeg_t)
                    ##### Framework Encoding #####

                    ##### Send request #####
                    time_start.record()
                    r = requests.post(url=service_uri, data=data_to_trans)
                    response = pickle.loads(r.content)
                    time_end.record()
                    torch.cuda.synchronize()
                    request_time.append(time_start.elapsed_time(time_end))
                    ##### Send request #####
                ##################### Collect resource usage ##########################
                tail_time.append(response["tail_time"])
                decode_time.append(response["decode_time"])
                ##################### 
                detection = response["detection"]

                if len(detection[0])!= 0:
                    pred = dict(boxes=tensor(detection[0].numpy()[:,0:4]),
                                scores=tensor(detection[0].numpy()[:,4]),
                                labels=tensor(detection[0].numpy()[:,5],dtype=torch.int32), )
                    # print("------------------------------------")
                    # print(pred)
                    # print("***********")
                    # print(frame_labels[index])
                    # print("------------------------------------")
                else:
                    pred = dict(boxes=tensor([]),
                                scores=tensor([]),
                                labels=tensor([],dtype=torch.int32),)
                frame_predicts.append(pred)
            metric = MeanAveragePrecision(iou_type="bbox") 
            metric.update(frame_predicts, frame_labels[N_warmup:])
            maps = metric.compute()

            with open(map_output_path,'a') as f:
                f.write(video_name+","
                        +str(thresh)+","
                        +str(quality)+","
                        +str(np.array(transfer_data_size).mean())+","
                        +str(np.array(transfer_data_size).std())+","
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
                        +str(thresh)+","
                        +str(quality)+","
                        +str(np.array(head_time).mean())+","
                        +str(np.array(head_time).std())+","
                        +str(np.array(tail_time).mean())+","
                        +str(np.array(tail_time).std())+","
                        +str(np.array(framework_time).mean())+","
                        +str(np.array(framework_time).std())+","
                        +str(np.array(jpeg_time).mean())+","
                        +str(np.array(jpeg_time).std())+","
                        +str(np.array(encode_time).mean())+","
                        +str(np.array(encode_time).std())+","
                        +str(np.array(decode_time).mean())+","
                        +str(np.array(decode_time).std())+","
                        +str(np.array(request_time).mean())+","
                        +str(np.array(request_time).std())+"\n"
                )
                