################################### setting path ###################################
import sys
sys.path.append('../')
sys.path.append('../../')
################################### import libs ###################################
from  pytorchyolo import  models_split_tiny
import numpy as np
import time
import torch
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
from pytorchyolo.utils.utils import non_max_suppression
################################### Varialbe init ###################################
__COMPRESSION_TECHNIQUE__ = "jpeg"

N_warmup = 0
split_layer= int(sys.argv[1])

cfg_path = "../../pytorchyolo/config/yolov3-tiny.cfg"


testdata_path = "../../St_Marc_dataset/data/test_30_fps_long_cleaned.txt"
class_name_path = "../../St_Marc_dataset/data/coco.names"
log_dir = "../measurements/"
model_path = "../ckpt/stmarc.pth"

# testdata_path = "../../dataset/football/test_long.txt"
# class_name_path = "../../dataset/football/classes.names"
# log_dir = "../measurements_bev/"
# model_path = "../ckpt/bev.pth"

# testdata_path = "../../dataset/vidvrd/test_long.txt"
# class_name_path = "../../dataset/vidvrd/classes.names"
# log_dir = "../measurements_vidvrd/"
# model_path = "../ckpt/vidVRD.pth"

test_case = "direct_split"
service_uri = "http://10.0.1.34:8092/tensor"
reset_uri = "http://10.0.1.34:8092/reset"

measurement_path = log_dir+test_case+"/"
map_output_path = measurement_path+ "map.csv"
time_output_path = measurement_path+ "time.csv"
characteristic_output_path = measurement_path+ "characteristic.csv"

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
        

with open(map_output_path,'a') as f:
    title = ("pruning_thresh,"
            "quality,"
            "technique,"
            "frame_id,"
            "datasize,"
            "sensitivity,"
            "map\n")
    f.write(title)

with open(time_output_path,'a') as f:
    title = (
            "frame_id,"
            "head_time,"
            "serialize_time,"
            "tail_time\n"
            )
    f.write(title)

# with open(characteristic_output_path,'a') as f:
#     title = ("pruning_thresh,"
#             "quality,"
#             "technique,"
#             "frame_id,"
#             "sparsity,"
#             "decomposability,"
#             "regularity,"
#             "pictoriality,"
#             "datasize_est,"
#             "datasize_real,"
#             "reconstruct_snr\n")
#     f.write(title)


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

# def write_time_data(sf, thresh,quality,tech,frame_id):
#     model_head_time, model_tail_time = sf.get_model_time_measurement()
#     fw_head_time,fw_tail_time,fw_response_time = sf.get_framework_time_measurement()
#     compression_time, decompression_time = sf.get_compression_time_measurement()
#     overall_time = sf.get_overall_time_measurement()
    
#     if __COMPRESSION_TECHNIQUE__ =="sketchml":
#         quality= str(quality[0])+"-"+str(quality[1])+"-"+str(quality[2])

#     with open(time_output_path,'a') as f:
#         f.write(str(thresh)+","
#                 +str(quality)+","
#                 +str(tech)+","
#                 +str(frame_id)+","
#                 +str(model_head_time)+","
#                 +str(model_tail_time)+","
#                 +str(fw_head_time)+","
#                 +str(fw_tail_time)+","
#                 +str(fw_response_time)+","
#                 +str(compression_time)+","
#                 +str(decompression_time)+","
#                 +str(overall_time)+"\n"
#                 )
        
# def write_characteristic(sf, thresh,quality,tech,frame_id):
#     sparsity, decomposability,regularity,pictoriality = sf.get_tensor_characteristics()
#     datasize_est, datasize_real = sf.get_data_size()
#     reconstruct_snr = sf.get_reconstruct_snr()

#     if __COMPRESSION_TECHNIQUE__ =="sketchml":
#         quality= str(quality[0])+"-"+str(quality[1])+"-"+str(quality[2])


#     with open(characteristic_output_path,'a') as f:
#         f.write(str(thresh)+","
#                 +str(quality)+","
#                 +str(tech)+","
#                 +str(frame_id)+","
#                 +str(sparsity)+","
#                 +str(decomposability)+","
#                 +str(regularity)+","
#                 +str(pictoriality)+","
#                 +str(datasize_est)+","
#                 +str(datasize_real)+","
#                 +str(reconstruct_snr)+"\n"
#                 )
        
def write_map( thresh,quality,tech,frame_id,datasize,sensitivity,map_value):
    if __COMPRESSION_TECHNIQUE__ =="sketchml":
        quality= str(quality[0])+"-"+str(quality[1])+"-"+str(quality[2])
    with open(map_output_path,'a') as f:
                f.write(str(thresh)+","
                        +str(quality)+","
                        +str(tech)+","
                        +str(frame_id)+","
                        +str(datasize)+","
                        +str(sensitivity)+","
                        +str(map_value)+"\n"
                        )
################################### Main function ###################################

if __name__ == "__main__":
    # Load Model
    model = models_split_tiny.load_model(cfg_path,model_path)
    model.set_split_layer(model_split_layer) # layer <7
    model = model.eval()
    time_start = torch.cuda.Event(enable_timing=True)
    time_end = torch.cuda.Event(enable_timing=True)
    dataloader = create_data_loader(testdata_path)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    class_names = load_classes(class_name_path)  # List of class names
    for j in range(1):
        for i in range(1):
            reset_required = True
            # while reset_required:
            #     r = requests.post(url=reset_uri)
            #     result = pickle.loads(r.content)
            #     if result["reset_status"] == True:
            #         reset_required = False
            #     else:
            #         print("Reset edge reference tensor failed...")
            #     time.sleep(1)

            
            ################## Init measurement lists ##########################
            # labels = []
            # sample_metrics = []  # List of tuples (TP, confs, pred)
            frame_index = 0
            for _, imgs, targets in tqdm.tqdm(dataloader, desc="testing"):
                frame_index+=1

                # Real measurements
                # Extract labels
                labels = targets[:, 1].tolist()
                # Rescale target
                targets[:, 2:] = xywh2xyxy(targets[:, 2:])
                targets[:, 2:] *= 416

                time_start.record()
                imgs = Variable(imgs.type(Tensor), requires_grad=False)
                with torch.no_grad():
                    head_tensor = model(imgs,1)
                    # print(detection)
                time_end.record()
                torch.cuda.synchronize()
                head_time = time_start.elapsed_time(time_end)

                time_start.record()
                tensor_cpu = head_tensor.cpu()
                tensor_bytes = pickle.dumps(tensor_cpu)
                datasize_to_trans = len(tensor_bytes)
                time_end.record()
                torch.cuda.synchronize()
                ser_time = time_start.elapsed_time(time_end)


                time_start.record()
                with torch.no_grad():
                    inference_result = model(head_tensor,2)
                    # print(detection)
                time_end.record()
                torch.cuda.synchronize()
                tail_time = time_start.elapsed_time(time_end)
                
                detection = non_max_suppression(inference_result, 0.01, 0.5)
                sample_metrics = get_batch_statistics(detection, targets, iou_threshold=0.1)




                # Concatenate sample statistics
                true_positives, pred_scores, pred_labels = [
                    np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
                metrics_output = ap_per_class(
                    true_positives, pred_scores, pred_labels, labels)
        
                sensitivity = np.sum(true_positives) / len(labels)
                precision, recall, AP, f1, ap_class = print_eval_stats(metrics_output, class_names, True)
                ## Save data
                write_map(0,0,0,frame_index,datasize_to_trans,sensitivity,AP.mean())
                with open(time_output_path,'a') as f:
                    f.write(str(frame_index)+","
                            +str(head_time)+","
                            +str(ser_time)+","
                            +str(tail_time)+"\n"
                            )
                

                