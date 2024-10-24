################################### setting path ###################################
import sys
sys.path.append('../../')
################################### import libs ###################################
from  pytorchyolo import  models_split_tiny
import numpy as np
import time
import torch
import os
from split_framework.stable.split_framework_jpeg import SplitFramework
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

################################### Varialbe init ###################################
__COMPRESSION_TECHNIQUE__ = "jpeg"

N_warmup = 0
split_layer= int(sys.argv[1])

testdata_path = "../../St_Marc_dataset/data/test_30_fps_cleaned.txt"
# testdata_path = "../../St_Marc_dataset/data/test_0.txt"
class_name_path = "../../St_Marc_dataset/data/coco.names"
log_dir = "../measurements/"

test_case = "jpeg_sens"
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
            "sensitivity,"
            "map\n")
    f.write(title)

with open(time_output_path,'a') as f:
    title = ("pruning_thresh,"
            "quality,"
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
            "frame_id,"
            "sparsity,"
            "decomposability,"
            "regularity,"
            "pictoriality,"
            "datasize_est,"
            "datasize_real,"
            "reconstruct_snr\n")
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

def write_time_data(sf, thresh,quality,frame_id):
    model_head_time, model_tail_time = sf.get_model_time_measurement()
    fw_head_time,fw_tail_time,fw_response_time = sf.get_framework_time_measurement()
    compression_time, decompression_time = sf.get_compression_time_measurement()
    overall_time = sf.get_overall_time_measurement()
    
    if __COMPRESSION_TECHNIQUE__ =="sketchml":
        quality= str(quality[0])+"-"+str(quality[1])+"-"+str(quality[2])

    with open(time_output_path,'a') as f:
        f.write(str(thresh)+","
                +str(quality)+","
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
        
def write_characteristic(sf, thresh,quality,frame_id):
    sparsity, decomposability,regularity,pictoriality = sf.get_tensor_characteristics()
    datasize_est, datasize_real = sf.get_data_size()
    reconstruct_snr = sf.get_reconstruct_snr()

    if __COMPRESSION_TECHNIQUE__ =="sketchml":
        quality= str(quality[0])+"-"+str(quality[1])+"-"+str(quality[2])


    with open(characteristic_output_path,'a') as f:
        f.write(str(thresh)+","
                +str(quality)+","
                +str(frame_id)+","
                +str(sparsity)+","
                +str(decomposability)+","
                +str(regularity)+","
                +str(pictoriality)+","
                +str(datasize_est)+","
                +str(datasize_real)+","
                +str(reconstruct_snr)+"\n"
                )
        
def write_map( thresh,quality,sensitivity,map_value):
    if __COMPRESSION_TECHNIQUE__ =="sketchml":
        quality= str(quality[0])+"-"+str(quality[1])+"-"+str(quality[2])
    with open(map_output_path,'a') as f:
                f.write(str(thresh)+","
                        +str(quality)+","
                        +str(sensitivity)+","
                        +str(map_value)+"\n"
                        )
################################### Main function ###################################

if __name__ == "__main__":
    # Load Model
    model = models_split_tiny.load_model("../../pytorchyolo/config/yolov3-tiny.cfg","../ckpt/yolov3_ckpt_300.pth")
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
            thresh = 0.05*(j+1)
            
            if __COMPRESSION_TECHNIQUE__ == "jpeg":
                quality =60+10*i
            elif __COMPRESSION_TECHNIQUE__ == "regression":
                quality = i+1
            elif __COMPRESSION_TECHNIQUE__ =="decomposition":
                quality = (i+1)*2
            elif __COMPRESSION_TECHNIQUE__ =="sketchml":
                quality = [256, (i+1)*0.1, (i+1)]
            else:
                raise Exception("Unknown compression techique!")
            # thresh = 0.0
            # quality =100
            print("Testing threshold: ",thresh,", Quality: ",quality)
            sf = SplitFramework(device="cuda", model=model)
            sf.set_reference_tensor(dummy_head_tensor)
            sf.set_pruning_threshold(thresh)
            sf.set_quality(quality)
            ################## Init measurement lists ##########################
            labels = []
            sample_metrics = []  # List of tuples (TP, confs, pred)
            frame_index = 0
            for _, imgs, targets in tqdm.tqdm(dataloader, desc="testing"):
                frame_index+=1
                # Warmup phase
                imgs = Variable(imgs.type(Tensor), requires_grad=False)
                if frame_index <= N_warmup:
                    with torch.no_grad():
                        detection=sf.split_framework_client(imgs,service_uri=service_uri)
                        continue
                
                # Real measurements
                # Extract labels
                labels += targets[:, 1].tolist()
                # Rescale target
                targets[:, 2:] = xywh2xyxy(targets[:, 2:])
                targets[:, 2:] *= 416
                
                detection = sf.split_framework_client(imgs,service_uri=service_uri)
                write_time_data(sf,thresh,quality,frame_index)
                write_characteristic(sf,thresh,quality,frame_index)
                sample_metrics += get_batch_statistics(detection, targets, iou_threshold=0.1)
        
            # Concatenate sample statistics
            true_positives, pred_scores, pred_labels = [
                np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
            metrics_output = ap_per_class(
                true_positives, pred_scores, pred_labels, labels)
    
            sensitivity = np.sum(true_positives) / len(labels)
            precision, recall, AP, f1, ap_class = print_eval_stats(metrics_output, class_names, True)
            ## Save data
            write_map(thresh,quality,sensitivity,(AP[0]+AP[1])/2)
                

                