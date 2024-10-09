from ultralytics import YOLO
import cv2
import pandas as pd
import torch
# from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS
import torchvision.transforms as transforms
import numpy as np
import os

def save_video_frames(video_path, output_dir, sampling_rate=1):
    test_frames = []

    # Create the directory to save the frames if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)  # Open the video file
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()  # Read a frame
        
        if not ret:
            break  # Break the loop if no more frames
        
        # Only process and save frames at intervals defined by the sampling_rate
        if frame_count % sampling_rate == 0:
            # Resize the frame
            resized_frame = cv2.resize(frame, (416, 416), interpolation=cv2.INTER_AREA)

            # Save the frame to a file
            frame_filename = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_filename, resized_frame)

            # Optional: store the frame in the list
            test_frames.append(resized_frame)
            
            saved_count += 1  # Increment the count of saved frames

        frame_count += 1  # Increment the frame counter

    cap.release()

    print(f"Total frames captured: {frame_count}")
    print(f"Frames saved: {saved_count}")
    print(f"Frames saved in: {output_dir}")

if __name__ == "__main__":

    video_path= "./stmarc_video.avi"

    model = YOLO('yolov8x.pt')

    fps_list = [30, 15, 10, 5]

    for fps in fps_list:
        sampling_rate = 30/fps
        output_dir = f'./frames/{fps}_fps'
        save_video_frames(video_path, output_dir, sampling_rate)