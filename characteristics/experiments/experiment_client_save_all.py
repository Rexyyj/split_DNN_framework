################################### setting path ###################################
import sys
sys.path.append('../../')
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
from split_framework.yolov3_tensor_jpeg_chara import SplitFramework
# from split_framework.yolov3_tensor_regression_chara import SplitFramework
import requests
import pickle
from torchmetrics.detection import MeanAveragePrecision
from torch.profiler import profile, record_function, ProfilerActivity
################################### Varialbe init ###################################
video_path = "../../dataset/test/"
label_path = "../../dataset/test_label/"
video_files = os.listdir(video_path)
# video_files = os.listdir(video_path)
# video_names = [name.replace('.mov','') for name in video_files]
video_names =["b610204c-e3c8c65f"]
N_frame = 25
N_warmup = 5
split_layer= int(sys.argv[1])

test_case = "tensor_jpeg_all"
service_uri = "http://10.0.1.34:8090/tensor_jpeg"
reset_uri = "http://10.0.1.34:8090/reset"

log_dir = "../measurements/layer_"+str(split_layer)+"/"
measurement_path = log_dir+test_case+"/"
map_output_path = measurement_path+ "map.csv"
time_output_path = measurement_path+ "time.csv"

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
            "sparsity,"
            "decomposability,"
            "regularity,"
            "pictoriality,"
            "map_50\n")
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
    model = models_split_tiny.load_model("../../pytorchyolo/config/yolov3-tiny.cfg","../../pytorchyolo/weights/yolov3-tiny.weights")
    model.set_split_layer(model_split_layer) # layer <7
    model = model.eval()
    
    for video_name in video_names:
        print("Testing with video: "+video_name)
        test_frames = load_video_frames(video_path,video_name, N_frame)
        frame_labels = load_ground_truth(video_name)

        for j in range(1):
            for i in range(1):
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
                # thresh = 0.0 * (j+1)
                # quality =i+1
                thresh = 0.0 * (j+1)
                quality =60 + i*10
                print("Testing threshold: ",thresh,", Jpeg quality: ",quality)
                sf = SplitFramework(device="cuda")
                sf.set_reference_tensor(dummy_head_tensor)
                sf.set_pruning_threshold(thresh)
                sf.set_quality(quality)

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
                sparsity = []
                decomposability =[]
                regularity =[]
                pictoriality =[]
                map_50 =[]
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
                        framework_t, jpeg_t,data_to_trans = sf.split_framework_encode(index, head_tensor,characteristic=True)
                        time_end.record()
                        torch.cuda.synchronize()
                        sp,de,re,pi = sf.get_tensor_characteristics()
                        sparsity.append(sp)
                        decomposability.append(de)
                        regularity.append(re)
                        pictoriality.append(pi)
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
                    print(detection)
                    if len(detection[0])!= 0:
                        pred = dict(boxes=tensor(detection[0].numpy()[:,0:4]),
                                    scores=tensor(detection[0].numpy()[:,4]),
                                    labels=tensor(detection[0].numpy()[:,5],dtype=torch.int32), )
                        metric = MeanAveragePrecision(iou_type="bbox") 
                        metric.update([pred], [frame_labels[index]])
                        maps = metric.compute()
                        print(maps)
                        map_50.append(maps["map_50"].item())
                        
                    else:
                        pred = dict(boxes=tensor([]),
                                    scores=tensor([]),
                                    labels=tensor([],dtype=torch.int32),)
                        map_50.append(0)
                        

                    with open(map_output_path,'a') as f:
                        f.write(video_name+","
                                +str(thresh)+","
                                +str(quality)+","
                                +str(transfer_data_size[-1])+","
                                +str(sparsity[-1])+","
                                +str(decomposability[-1])+","
                                +str(regularity[-1])+","
                                +str(pictoriality[-1])+","
                                +str(map_50[-1])+"\n"
                                )