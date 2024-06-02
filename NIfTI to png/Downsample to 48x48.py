import os
from PIL import Image

def downsample_image(input_filepath, output_filepath, size=(48, 48)):
    with Image.open(input_filepath) as img:
        img_resized = img.resize(size, Image.Resampling.LANCZOS)
        img_resized.save(output_filepath)
        print(f"Downsampled image saved to {output_filepath}")

def process_and_downsample(data_dir, new_data_dir):
    for group in ['SCHZ', 'HC']:
        group_dir = os.path.join(data_dir, group)
        new_group_dir = os.path.join(new_data_dir, group)
        os.makedirs(new_group_dir, exist_ok=True)

        for subject in os.listdir(group_dir):
            subject_dir = os.path.join(group_dir, subject, 'func')
            new_subject_dir = os.path.join(new_group_dir, subject, 'func')
            os.makedirs(new_subject_dir, exist_ok=True)

            for task_file in os.listdir(subject_dir):
                if task_file.endswith('.png'):
                    input_filepath = os.path.join(subject_dir, task_file)
                    output_filepath = os.path.join(new_subject_dir, task_file)
                    downsample_image(input_filepath, output_filepath)

# Define the original and new data directories
data_dir = '/home/orion/Geo/UCLA data/FMRIPrep/Brain Mask_2D Converted/'
new_data_dir = '/home/orion/Geo/UCLA data/FMRIPrep/Brain Mask_2D Converted/Downsampled/'

# Process and downsample the PNG files
process_and_downsample(data_dir, new_data_dir)
