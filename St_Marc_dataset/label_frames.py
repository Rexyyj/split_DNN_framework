from ultralytics import YOLO
import cv2
import pandas as pd
import torch
from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS
import torchvision.transforms as transforms
import numpy as np
import os

def convert_rgb_frame_to_tensor(image):
    img_size = 416
    # Configure input
    input_img = transforms.Compose([
    DEFAULT_TRANSFORMS,
    Resize(img_size)])(
        (image, np.zeros((1, 5))))[0].unsqueeze(0)
    input_img = input_img.cuda()

    return input_img

def read_saved_frames(frame_dir):
    frames = []
    
    # Get a sorted list of all image file names in the directory
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    for frame_file in frame_files:
        # Construct the full file path
        frame_path = os.path.join(frame_dir, frame_file)
        
        # Read the frame/image using OpenCV
        frame = cv2.imread(frame_path)
        
        if frame is not None:
            frames.append(frame)
        else:
            print(f"Warning: Failed to load frame {frame_path}")
    
    print(f"Total frames read: {len(frames)}")
    return frames


if __name__ == "__main__":

    model = YOLO('yolov8x.pt')
    video_path= "./stmarc_video.avi"

    fps_list = [30, 15, 10, 5]

    for fps in fps_list:
        sampling_rate = 30/fps
        output_dir = f'./frames/{fps}_fps'
        frames = read_saved_frames(output_dir)

        df_result = pd.DataFrame(columns=["video_id","frame_id", 'class_id',"predict_acc","xmin","ymin","xmax","ymax"])
        indexes = []
        time_measurement = []

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        for index in range(len(frames)):
        # for index in range(1):
            indexes.append(index)
            frame = frames[index]

            input_tensor = convert_rgb_frame_to_tensor(frame)
            
            start.record()
            results = model.predict(input_tensor)
            end.record()
            # results = results.cpu()
            torch.cuda.synchronize()
            time_measurement.append(start.elapsed_time(end))
            # [[x1, y1, x2, y2, confidence, class]]
            if len(results)!= 0 :
                for result in results:
                    for i in range(len(result.boxes.cls)):
                        df2 = pd.DataFrame([[video_path,
                                            index, 
                                            result.boxes.cls[i].item(),
                                            result.boxes.conf[i].item(),
                                            result.boxes.xyxy[i][0].item(),
                                            result.boxes.xyxy[i][1].item(),
                                            result.boxes.xyxy[i][2].item(),
                                            result.boxes.xyxy[i][3].item()]], columns=["video_id","frame_id", 'class_id',"predict_acc","xmin","ymin","xmax","ymax"])
                        df_result=pd.concat([df_result,df2])
            else:
                df2 = pd.DataFrame([[video_path,index, -1,-1,-1,-1,-1,-1]], columns=["video_id","frame_id", 'class_id',"predict_acc","xmin","ymin","xmax","ymax"])
                df_result=pd.concat([df_result,df2])

        df_result.to_csv(f'./labels/{fps}_fps.csv',index=False)

