import os
import numpy as np
import nibabel as nib
import random

# Directories and parameters
data_dir = '/home/orion/Geo/UCLA data/FMRIPrep/padded5/'
new_data_dir = '/home/orion/Geo/UCLA data/FMRIPrep/padr5/'
target_shape = (65, 77, 49)
max_time_length = 300

def pad_data(data, max_time_length):
    """ Pad data to have max_time_length time points. """
    if data.shape[-1] < max_time_length:
        padding = max_time_length - data.shape[-1]
        pad_width = [(0, 0), (0, 0), (0, 0), (0, padding)]
        data = np.pad(data, pad_width, mode='constant')
    elif data.shape[-1] > max_time_length:
        data = data[..., :max_time_length]
    return data

def save_nifti(data, file_path):
    """ Save NIfTI data. """
    new_img = nib.Nifti1Image(data, np.eye(4))
    nib.save(new_img, file_path)

# Prepare directories
os.makedirs(new_data_dir, exist_ok=True)

for group in ['SCHZ', 'HC']:
    group_dir = os.path.join(data_dir, group)
    new_group_dir = os.path.join(new_data_dir, group)
    os.makedirs(new_group_dir, exist_ok=True)

    all_data = []

    for subject in os.listdir(group_dir):
        subject_dir = os.path.join(group_dir, subject, 'func')
        for task_file in os.listdir(subject_dir):
            if 'preproc.nii.gz' in task_file:
                bold_path = os.path.join(subject_dir, task_file)
                img = nib.load(bold_path)
                data = img.get_fdata()

                # Pad data
                padded_data = pad_data(data, max_time_length)
                all_data.append(padded_data)

    current_count = len(all_data)
    print(f"Total sample count for group {group}: {current_count}")

    # Save data
    for i, sample in enumerate(all_data):
        new_subject_dir = os.path.join(new_group_dir, f'subject_{i:04d}', 'func')
        os.makedirs(new_subject_dir, exist_ok=True)
        save_path = os.path.join(new_subject_dir, 'preproc.nii.gz')
        save_nifti(sample, save_path)
        if i % 100 == 0:
            print(f"Saved {i} samples for group {group}")

print('Data padding completed.')
