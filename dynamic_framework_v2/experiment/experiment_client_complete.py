################################### setting path ###################################
import sys
sys.path.append('../')
sys.path.append('../../')
################################### import libs ###################################
from  pytorchyolo import  models_split_tiny
import numpy as np
import time
import torch
import math
import os
from split_framework.split_framework_dynamic import SplitFramework
import tqdm
import numpy as np
import requests, pickle
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from terminaltables import AsciiTable
from pytorchyolo.utils.utils import load_classes, ap_per_class, get_batch_statistics, xywh2xyxy
from pytorchyolo.utils.datasets import ListDataset
from pytorchyolo.utils.transforms import DEFAULT_TRANSFORMS
from algorithm.manager_full import Manager
from split_framework.gnb import GNB
################################### Varialbe init ###################################

split_layer= int(sys.argv[1])

testdata_path = "../../St_Marc_dataset/data/test_30_fps_long_cleaned.txt"
# testdata_path = "../../St_Marc_dataset/data/test_0.txt"
class_name_path = "../../St_Marc_dataset/data/coco.names"
log_dir = "../measurements/"

test_case = "full_manager_test4"
service_uri = "http://10.0.1.34:8092/tensor"
reset_uri = "http://10.0.1.34:8092/reset"

measurement_path = log_dir+test_case+"/"
map_output_path = measurement_path+ "map.csv"
time_output_path = measurement_path+ "time.csv"
characteristic_output_path = measurement_path+ "characteristic.csv"
manager_output_path = measurement_path + "manager.csv"

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
    os.system("rm -rf "+measurement_path)
    os.system("mkdir -p "+measurement_path)
        
with open(manager_output_path,"a") as f:
    title= (
        "frame_id,"
        "bandwidth,"
        "mAP_drop,"
        "target_fps,"
        "technique,"
        "feasibility,"
        "target_cmp,"
        "target_snr,"
        "est_cmp,"
        "est_snr,"
        "pruning_thresh,"
        "quality,"
        "jpeg_F,"
        "decom_F,"
        "reg_F\n"
    )
    f.write(title)

with open(map_output_path,'a') as f:
    title = ("pruning_thresh,"
            "quality,"
            "technique,"
            "bandwidth,"
            "mAP_drop,"
            "frame_id,"
            "feasible,"
            "sensitivity,"
            "map\n")
    f.write(title)

with open(time_output_path,'a') as f:
    title = ("pruning_thresh,"
            "quality,"
            "technique,"
            "bandwidth,"
            "mAP_drop,"
            "frame_id,"
            "model_head_time,"
            "model_tail_time,"
            "framework_head_time,"
            "framework_tail_time,"
            "framework_response_time,"
            "compression_time,"
            "decompression_time,"
            "overall_time\n"
            )
    f.write(title)

with open(characteristic_output_path,'a') as f:
    title = ("pruning_thresh,"
            "quality,"
            "technique,"
            "bandwidth,"
            "mAP_drop,"
            "frame_id,"
            "sparsity,"
            "decomposability,"
            "regularity,"
            "pictoriality,"
            "compression_ratio,"
            "datasize_est,"
            "datasize_real,"
            "reconstruct_snr,"
            "target_cmp,"
            "target_snr\n")
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

def write_manager_data(frame_id, bandwidth, mAP_drop, target_fps,manager):

    with open(manager_output_path,"a") as f:
        f.write(
            str(frame_id)+","
            +str(bandwidth)+","
            +str(mAP_drop)+","
            +str(target_fps)+","
            +str(manager.get_compression_technique())+","
            +str(manager.get_feasibility())+","
            +str(manager.get_target_cmp())+","
            +str(manager.get_target_snr())+","
            +str(manager.get_est_cmp())+","
            +str(manager.get_est_snr())+","
            +str(manager.get_pruning_threshold())+","
            +str(manager.get_compression_quality())+","
            +str(manager.get_result_jpeg_f())+","
            +str(manager.get_result_decom_f())+","
            +str(manager.get_result_reg_f())+"\n"
        )

def write_time_data(sf, thresh,quality,tech,bandwidth, mAP_drop,frame_id):
    model_head_time, model_tail_time = sf.get_model_time_measurement()
    fw_head_time,fw_tail_time,fw_response_time = sf.get_framework_time_measurement()
    compression_time, decompression_time = sf.get_compression_time_measurement()
    overall_time = sf.get_overall_time_measurement()

    with open(time_output_path,'a') as f:
        f.write(str(thresh)+","
                +str(quality)+","
                +str(tech)+","
                +str(bandwidth)+","
                +str(mAP_drop)+","
                +str(frame_id)+","
                +str(model_head_time)+","
                +str(model_tail_time)+","
                +str(fw_head_time)+","
                +str(fw_tail_time)+","
                +str(fw_response_time)+","
                +str(compression_time)+","
                +str(decompression_time)+","
                +str(overall_time)+"\n"
                )
        
def write_characteristic(sf, manager,bandwidth,mAP_drop,frame_id):
    sparsity, decomposability,regularity,pictoriality = sf.get_tensor_characteristics()
    datasize_est, datasize_real = sf.get_data_size()
    reconstruct_snr = sf.get_reconstruct_snr()
    target_cmp, target_snr = manager.get_intermedia_measurements()
    cmp_ratio = (128*26*26*4)/datasize_est

    with open(characteristic_output_path,'a') as f:
        f.write(str(manager.get_pruning_threshold())+","
                +str(manager.get_compression_quality())+","
                +str(manager.get_compression_technique())+","
                +str(bandwidth)+","
                +str(mAP_drop)+","
                +str(frame_id)+","
                +str(sparsity)+","
                +str(decomposability)+","
                +str(regularity)+","
                +str(pictoriality)+","
                +str(cmp_ratio)+","
                +str(datasize_est)+","
                +str(datasize_real)+","
                +str(reconstruct_snr)+","
                +str(target_cmp)+","
                +str(target_snr)+"\n"
                )
        
def write_map( thresh,quality,tech,bandwidth,mAP_drop,frame_id,feasibility,sensitivity,map_value):
    with open(map_output_path,'a') as f:
                f.write(str(thresh)+","
                        +str(quality)+","
                        +str(tech)+","
                        +str(bandwidth)+","
                        +str(mAP_drop)+","
                        +str(frame_id)+","
                        +str(feasibility)+","
                        +str(sensitivity)+","
                        +str(map_value)+"\n"
                        )
################################### Main function ###################################
# User movement
user_loc = [0,0]
user_speed = [0.3,0,3]
user_dir = 1
bs_loc = [5.2,5.3]
gnb = GNB()
def update_user_loc_and_get_bw():
    global user_dir, user_loc, user_speed,bs_loc
    if user_dir == 1:
        user_loc[0] += user_speed[0]
        user_loc[1] += user_speed[1]
        if user_loc[0] >10 or user_loc[1]>10:
            user_dir = -1
    else:
        user_loc[0] -= user_speed[0]
        user_loc[1] -= user_speed[1]
        if user_loc[0]<0 or user_loc[1]<0:
            user_dir = 1
    dist = math.dist(user_loc, bs_loc)
    cqi = round(15 * (30-dist)/30)
    bw = gnb.estimate_throughput_cqi(cqi)/3
    return bw

##################################################################################

if __name__ == "__main__":
    # Load Model
    model = models_split_tiny.load_model("../../pytorchyolo/config/yolov3-tiny.cfg","../ckpt/yolov3_ckpt_300.pth")
    model.set_split_layer(model_split_layer) # layer <7
    model = model.eval()
    
    dataloader = create_data_loader(testdata_path)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    class_names = load_classes(class_name_path)  # List of class names

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
    sf = SplitFramework(device="cuda", model=model)
    sf.set_reference_tensor(dummy_head_tensor)
    manager = Manager()
    
    previouse_bandwidth = 0
    previouse_snr =0

    ################## Init measurement lists ##########################
    frame_index = 0
    for _, imgs, targets in tqdm.tqdm(dataloader, desc="testing"):
        frame_index+=1

        # availble bandwith calculation
        available_bandwidth = update_user_loc_and_get_bw()
        mAP_drop =0.30
        target_fps = 10
        # technique = 1

        # interframe similarity calculation

        # check framework update events and run manager 
        if frame_index< manager.get_testing_frame_length()+1:
            manager.update_requirements(mAP_drop,target_fps,available_bandwidth,frame_index)
            fesiable = manager.get_feasibility()
            write_manager_data(frame_index,available_bandwidth,mAP_drop,target_fps, manager)
        elif abs(available_bandwidth-previouse_bandwidth)>1e6:
            manager.update_requirements(mAP_drop,target_fps,available_bandwidth,frame_index)
            fesiable = manager.get_feasibility()
            write_manager_data(frame_index,available_bandwidth,mAP_drop,target_fps, manager)
            previouse_bandwidth = available_bandwidth
        elif (manager.get_target_snr() - previouse_snr) > 0.1:
            manager.update_requirements(mAP_drop,target_fps,available_bandwidth,frame_index)
            fesiable = manager.get_feasibility()
            write_manager_data(frame_index,available_bandwidth,mAP_drop,target_fps, manager)
            previouse_bandwidth = available_bandwidth

        
        # thresh, quality = manager.get_configuration() 
        # set framework configuration
        sf.set_compression_technique(manager.get_compression_technique()) # set to jpeg
        sf.set_quality(manager.get_compression_quality())
        sf.set_pruning_threshold(manager.get_pruning_threshold())


        # Real measurements
        imgs = Variable(imgs.type(Tensor), requires_grad=False)
        # Extract labels
        labels = targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= 416

        try:
            data_size,_ = sf.get_data_size()
            # print(data_size)
            cmp= 128*26*26*4/data_size
        except:
            cmp = 0
            print("Get cmp error")
        manager.update_sample_points(manager.get_compression_technique(),(manager.get_pruning_threshold(),manager.get_compression_quality()),cmp,sf.get_reconstruct_snr())
        previouse_snr = sf.get_reconstruct_snr()
        
        detection = sf.split_framework_client(imgs,service_uri=service_uri)
        write_time_data(sf,manager.get_pruning_threshold(),manager.get_compression_quality(),manager.get_compression_technique(),available_bandwidth,mAP_drop,frame_index)
        write_characteristic(sf,manager,available_bandwidth,mAP_drop,frame_index)
        sample_metrics = get_batch_statistics(detection, targets, iou_threshold=0.1)

        # Concatenate sample statistics
        try:
            true_positives, pred_scores, pred_labels = [
                np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
            metrics_output = ap_per_class(
                true_positives, pred_scores, pred_labels, labels)
    
            sensitivity = np.sum(true_positives) / len(labels)
            precision, recall, AP, f1, ap_class = print_eval_stats(metrics_output, class_names, True)
            ## Save data
            write_map(manager.get_pruning_threshold(),manager.get_compression_quality(),manager.get_compression_technique(),available_bandwidth,mAP_drop,frame_index,fesiable,sensitivity,AP.mean())
        except:
            write_map(manager.get_pruning_threshold(),manager.get_compression_quality(),manager.get_compression_technique(),manager.get_compression_technique(),available_bandwidth,mAP_drop,frame_index,fesiable,0, 0)

        

        