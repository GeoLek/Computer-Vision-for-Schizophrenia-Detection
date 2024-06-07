import os
import numpy as np
import nibabel as nib
import random
from scipy.ndimage import rotate, zoom

# Directories and parameters
data_dir = '/home/orion/Geo/UCLA data/FMRIPrep/SCHZ2'
new_data_dir = '/home/orion/Geo/UCLA data/FMRIPrep/SCHZ-Augmented2/'
#target_shape = (32, 38, 24)
target_shape = (65, 77, 49)
max_time_length = 300

# Augmentation settings
augmentation_factor = 162  # Number of samples per class


def augment_data(data):
    """ Apply data augmentation techniques. """
    augmented_data = []

    # Rotation
    angles = [5, -5, 10, -10]  # Rotate by these angles in degrees
    for angle in angles:
        augmented_data.append(rotate(data, angle, axes=(0, 1), reshape=False, mode='nearest'))

    # Zoom in and out
    zoom_factors = [0.9, 1.1]  # Zoom factors for in and out
    for factor in zoom_factors:
        zoomed = zoom(data, (1, 1, 1, factor))
        # Ensure the time dimension is back to the original size if necessary
        if zoomed.shape[-1] != data.shape[-1]:
            if zoomed.shape[-1] < data.shape[-1]:
                zoomed = pad_data(zoomed, data.shape[-1])
            else:
                zoomed = zoomed[..., :data.shape[-1]]
        augmented_data.append(zoomed)

    return augmented_data


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
subjects = []
labels = []

for group in ['SCHZ', 'HC']:
    group_dir = os.path.join(data_dir, group)
    label = 0 if group == 'SCHZ' else 1
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

    # Augment data
    current_count = len(all_data)
    print(f"Initial sample count for group {group}: {current_count}")

    while len(all_data) < augmentation_factor:
        sample = random.choice(all_data)
        augmented_samples = augment_data(sample)
        for augmented_sample in augmented_samples:
            if len(all_data) >= augmentation_factor:
                break
            all_data.append(augmented_sample)
            current_count += 1
            if current_count % 100 == 0:
                print(f"Current count for group {group}: {current_count}")

    print(f"Total sample count after augmentation for group {group}: {len(all_data)}")

    # Save data
    for i, sample in enumerate(all_data):
        new_subject_dir = os.path.join(new_group_dir, f'subject_{i:04d}', 'func')
        os.makedirs(new_subject_dir, exist_ok=True)
        save_path = os.path.join(new_subject_dir, 'preproc.nii.gz')
        save_nifti(sample, save_path)
        if i % 100 == 0:
            print(f"Saved {i} samples for group {group}")

print('Data augmentation and padding completed.')
