import csv
import os

# Path to the CSV file and output directory
csv_file = 'raw_labels/30_fps.csv'  # Replace with the actual CSV file path
output_dir = './30_fps/labels'  # Replace with the desired output directory

# Image size (fixed at 416x416)
image_width = 416
image_height = 416

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to format the frame_id to the desired file name
def format_frame_id(frame_id):
    return f"frame_{int(frame_id):05d}.txt"

# Read the CSV file and process each row
with open(csv_file, 'r') as file:
    reader = csv.DictReader(file)
    
    # Dictionary to store data by frame_id
    frame_data = {}

    for row in reader:
        confidence = float(row['predict_acc'])

        if confidence < 0.55:
            continue

        frame_id = row['frame_id']
        class_id = row['class_id'].split('.')[0]
        xmin = float(row['xmin'])
        ymin = float(row['ymin'])
        xmax = float(row['xmax'])
        ymax = float(row['ymax'])
        
        # Calculate center coordinates, width, and height
        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0
        width = xmax - xmin
        height = ymax - ymin
        
        # Scale coordinates to [0, 1] based on image size (416x416)
        x_center /= image_width
        y_center /= image_height
        width /= image_width
        height /= image_height
        
        # Store the annotation for this frame
        if frame_id not in frame_data:
            frame_data[frame_id] = []
        
        # Append the scaled annotation in the desired format
        frame_data[frame_id].append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

# Write the annotations to text files
for frame_id, annotations in frame_data.items():
    # Format the filename
    filename = format_frame_id(frame_id)
    file_path = os.path.join(output_dir, filename)
    
    # Write all annotations for the same frame into one file
    with open(file_path, 'w') as f:
        for annotation in annotations:
            f.write(annotation + '\n')

print("Text files with scaled annotations created successfully.")
