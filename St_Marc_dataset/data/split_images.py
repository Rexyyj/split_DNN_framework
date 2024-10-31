import os
import random

# Path to the folder containing the images
image_dir = 'images'

# Get the list of image files
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
image_files = sorted(image_files)
print(image_files)

# Shuffle the image files to ensure randomness
# random.shuffle(image_files)

# Define split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Calculate the number of images for each subset
total_images = len(image_files)
train_count = int(total_images * train_ratio)
val_count = int(total_images * val_ratio)
test_count = total_images - train_count - val_count  # Remaining for test

# Split the images into train, val, and test sets
train_images = image_files[:train_count]
val_images = image_files[train_count:train_count + val_count]
test_images = image_files[train_count + val_count:]

# Write the image paths into .txt files
def write_to_file(file_list, filename):
    with open(filename, 'w') as f:
        for item in file_list:
            f.write(f"{item}\n")

# create empty files if they don't exist
open('train.txt', 'a').close()
open('valid.txt', 'a').close()
open('test.txt', 'a').close()

write_to_file(train_images, 'train.txt')
write_to_file(val_images, 'valid.txt')
write_to_file(test_images, 'test.txt')

print(f"Train set: {len(train_images)} images")
print(f"Validation set: {len(val_images)} images")
print(f"Test set: {len(test_images)} images")
