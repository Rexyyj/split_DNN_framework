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
import requests
import pickle
from torchmetrics.detection import MeanAveragePrecision
from torch.profiler import profile, record_function, ProfilerActivity
import simplejpeg
################################### Varialbe init ###################################
video_path = "../dataset/test/"
label_path = "../dataset/test_label/"
video_files = os.listdir(video_path)
video_names = [name.replace('.mov','') for name in video_files]
N_frame = 100

test_case = "frame_jpeg"
service_uri = "http://10.0.1.23:8090/frame_jpeg"

log_dir = "../measurements/"
measurement_path = log_dir+test_case+"/"
map_output_path = measurement_path+ "map.csv"
time_output_path = measurement_path+ "time.csv"
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
    title = ("video_name,"
            "jepg_quality,"
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
            "jepg_quality,"
            "model_time_mean,"
            "model_time_std,"
            "encode_time_mean,"
            "encode_time_std,"
            "decode_time_mean,"
            "decode_time_std,"
            "request_time_mean,"
            "request_time_std\n"
            )
    f.write(title)

with open(resource_output_path,'a') as f:
    title = ("video_name,"
            "jepg_quality,"
            "data_size_mean,"
            "data_size_std,"
            "cpu_cli_mean,"
            "cpu_cli_std,"
            "cpu_cli_mem_mean,"
            "cpu_cli_mem_std,"
            "cpu_edge_mean,"
            "cpu_edge_std,"
            "cuda_edge_mean,"
            "cuda_edge_std,"
            "cpu_edge_mem_mean,"
            "cpu_edge_mem_std,"
            "cuda_edge_mem_mean,"
            "cuda_edge_mem_std\n")
    f.write(title)
################################### Utility functions ###################################
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

        for i in range(5):
            print("In iter",i)

            frame_predicts = []
            # thresh = 0.05
            quality =60+10*i

            time_start = torch.cuda.Event(enable_timing=True)
            time_end = torch.cuda.Event(enable_timing=True)
            ################## Init measurement lists ##########################
            cpu_time_client = []
            # cuda_time =[]
            cpu_mem_client = []
            # cuda_mem_client = []
            cpu_time_edge = []
            cuda_time_edge =[]
            cpu_mem_edge = []
            cuda_mem_edge = []
            transfer_data_size =[]

            model_time =[]
            encode_time=[]
            decode_time=[]
            request_time=[]
            #####################################################################
            for index in range(len(test_frames)):
                
                frame = test_frames[index]
                ################## Perform Object detection #############################
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
                    with record_function("model_inference"):
                        with torch.no_grad():
                        
                            #####  Encoding #####
                            time_start.record()
                            encoded_data = simplejpeg.encode_jpeg(frame,quality)
                            payload = {"id":index, "frame":encoded_data}
                            data_to_trans =  pickle.dumps(payload)
                            time_end.record()
                            torch.cuda.synchronize()
                            encode_time.append(time_start.elapsed_time(time_end))
                            transfer_data_size.append(len(data_to_trans))
                            #####  Encoding #####

                            ##### Send request #####
                            time_start.record()
                            r = requests.post(url=service_uri, data=data_to_trans)
                            response = pickle.loads(r.content)
                            time_end.record()
                            torch.cuda.synchronize()
                            request_time.append(time_start.elapsed_time(time_end))
                            ##### Send request #####
                ##################### Collect resource usage ##########################
                resource_mea = prof.key_averages().table(sort_by="cuda_time_total", row_limit=1)
                mea=list(filter(None,resource_mea.split('\n')[3].split(" ")) ) 
                cpu_time_client.append(float(str(mea[2]).replace("m","").replace("s","")))
                # cuda_time.append(float(str(mea[8]).replace("m","").replace("s","")))
                cpu_mem_client.append(abs(float(mea[8]))/1000 if mea[9]=='Kb' else abs(float(mea[8])))
                cpu_time_edge.append(response["cpu_time"])
                cuda_time_edge.append(response["cuda_time"])
                cpu_mem_edge.append(response["cpu_mem"])
                cuda_mem_edge.append(response["cuda_mem"])
                model_time.append(response["model_time"])
                decode_time.append(response["decode_time"])
                ##################### 
                detection = response["detection"]

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

            with open(map_output_path,'a') as f:
                f.write(video_name+","
                        +str(quality)+","
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
                        +str(quality)+","
                        +str(np.array(model_time).mean())+","
                        +str(np.array(model_time).std())+","
                        +str(np.array(encode_time).mean())+","
                        +str(np.array(encode_time).std())+","
                        +str(np.array(decode_time).mean())+","
                        +str(np.array(decode_time).std())+","
                        +str(np.array(request_time).mean())+","
                        +str(np.array(request_time).std())+"\n"
                )
                
            with open(resource_output_path,'a') as f:
                f.write(video_name+","
                        +str(quality)+","
                        +str(np.array(transfer_data_size).mean())+","
                        +str(np.array(transfer_data_size).std())+","
                        +str(np.array(cpu_time_client).mean())+","
                        +str(np.array(cpu_time_client).std())+","
                        +str(np.array(cpu_mem_client).mean())+","
                        +str(np.array(cpu_mem_client).std())+","
                        +str(np.array(cpu_time_edge).mean())+","
                        +str(np.array(cpu_time_edge).std())+","
                        +str(np.array(cuda_time_edge).mean())+","
                        +str(np.array(cuda_time_edge).std())+","
                        +str(np.array(cpu_mem_edge).mean())+","
                        +str(np.array(cpu_mem_edge).std())+","
                        +str(np.array(cuda_mem_edge).mean())+","
                        +str(np.array(cuda_mem_edge).std())+"\n"
                )