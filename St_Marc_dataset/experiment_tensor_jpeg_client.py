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

N_warmup = 0
split_layer= int(sys.argv[1])
test_fps = int(sys.argv[2])

testdata_path = "./data/test_"+str(test_fps)+"_fps_cleaned.txt"
class_name_path = "./data/coco.names"
log_dir = "./measurements/jpeg/"+str(test_fps)+"_fps/"

# testdata_path = "./data/test_5_fps_cleaned.txt"
# class_name_path = "./data/coco.names"
# log_dir = "./measurements/jpeg/5_fps/"

test_case = "tensor"
service_uri = "http://10.0.1.34:8090/tensor"
reset_uri = "http://10.0.1.34:8090/reset"


measurement_path = log_dir+"/"
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
    os.mkdir(log_dir)
except:
    os.system("rm -rf "+log_dir)
    os.mkdir(log_dir)
        

with open(map_output_path,'a') as f:
    title = ("pruning_thresh,"
            "jepg_quality,"
            "data_size_mean,"
            "data_size_std,"
            "map\n")
    f.write(title)

with open(time_output_path,'a') as f:
    title = ("pruning_thresh,"
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
    return precision, recall, AP, f1, ap_class
################################### Main function ###################################

if __name__ == "__main__":
    # Load Model
    model = models_split_tiny.load_model("../pytorchyolo/config/yolov3-tiny.cfg","./ckpt/yolov3_ckpt_300.pth")
    model.set_split_layer(model_split_layer) # layer <7
    model = model.eval()
    
    dataloader = create_data_loader(testdata_path)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    class_names = load_classes(class_name_path)  # List of class names
    for j in range(5):
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
            thresh = 0.02*(j+1)
            quality =60+10*i
            # thresh = 0.0
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


            labels = []
            sample_metrics = []  # List of tuples (TP, confs, pred)
            frame_index = 0
            for _, imgs, targets in tqdm.tqdm(dataloader, desc="testing"):
                frame_index+=1
                # Warmup phase
                imgs = Variable(imgs.type(Tensor), requires_grad=False)
                if frame_index <= N_warmup:
                    with torch.no_grad():
                        head_tensor = model(imgs, 1)
                        framework_t,jpeg_t,data_to_trans = sf.split_framework_encode(frame_index, head_tensor)
                        r = requests.post(url=service_uri, data=data_to_trans)
                        response = pickle.loads(r.content)
                        continue
                
                # Real measurements
                # Extract labels
                labels += targets[:, 1].tolist()
                # Rescale target
                targets[:, 2:] = xywh2xyxy(targets[:, 2:])
                targets[:, 2:] *= 416

                with torch.no_grad():
                    ##### Head Model #####
                    time_start.record()
                    head_tensor = model(imgs, 1)
                    time_end.record()
                    torch.cuda.synchronize()
                    head_time.append(time_start.elapsed_time(time_end))
                    ##### Head Model #####

                    ##### Framework Encoding #####
                    time_start.record()
                    framework_t, jpeg_t,data_to_trans = sf.split_framework_encode(frame_index, head_tensor)
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
                tail_time.append(response["tail_time"])
                decode_time.append(response["decode_time"])
                detection = response["detection"]
                sample_metrics += get_batch_statistics(detection, targets, iou_threshold=0.1)
        
            # Concatenate sample statistics
            true_positives, pred_scores, pred_labels = [
                np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
            metrics_output = ap_per_class(
                true_positives, pred_scores, pred_labels, labels)

            precision, recall, AP, f1, ap_class = print_eval_stats(metrics_output, class_names, True)
            ## Save data
            with open(map_output_path,'a') as f:
                f.write(str(thresh)+","
                        +str(quality)+","
                        +str(np.array(transfer_data_size).mean())+","
                        +str(np.array(transfer_data_size).std())+","
                        +str((AP[0]+AP[1])/2)+"\n"
                        )
                
            with open(time_output_path,'a') as f:
                f.write(str(thresh)+","
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
                