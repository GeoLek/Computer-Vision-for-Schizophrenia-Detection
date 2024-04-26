import os
import random
import shutil

# Define your dataset directory and the output directories for splits
dataset_dir = '/home/orion/Geo/Projects/Computer-Vision-for-Schizophrenia-Detection/Build the 3D CNN/all/SCHZ'
output_dir = '/home/orion/Geo/Projects/Computer-Vision-for-Schizophrenia-Detection/Build the 3D CNN/all/SCHZ/split'

# Directories for train, test, and validation splits
train_dir = os.path.join(output_dir, 'train')
test_dir = os.path.join(output_dir, 'test')
val_dir = os.path.join(output_dir, 'val')

# Make sure directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Read all file names directly from the dataset directory
file_list = [f for f in os.listdir(dataset_dir) if f.endswith('.nii.gz')]

# Apply a more random shuffle
random.shuffle(file_list)

# Define split ratios
train_ratio = 0.7  # 70% for training
val_ratio = 0.1  # 10% for validation
test_ratio = 0.2  # 20% for testing (the remainder)

# Calculate split indices
train_split = int(train_ratio * len(file_list))
val_split = train_split + int(val_ratio * len(file_list))

# Move files to the respective directories
for i, file_name in enumerate(file_list):
    src_path = os.path.join(dataset_dir, file_name)

    if i < train_split:
        dst_path = os.path.join(train_dir, file_name)
    elif i < val_split:
        dst_path = os.path.join(val_dir, file_name)
    else:
        dst_path = os.path.join(test_dir, file_name)

    shutil.move(src_path, dst_path)

print(f"Dataset split: {train_ratio * 100}% train, {val_ratio * 100}% validation, {test_ratio * 100}% test.")
