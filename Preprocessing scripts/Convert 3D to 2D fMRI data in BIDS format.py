import os
import nibabel as nib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

# Function to transform 3D NIfTI to 2D slices
def nifti_to_specific_slice(nifti_file_path, output_folder, slice_index=None):
    # Load the NIfTI file
    nifti = nib.load(nifti_file_path)
    data = nifti.get_fdata()

    # Normalize the data to 0-255
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data) * 255
    data = data.astype(np.uint8)

    # Determine the middle slice index if not provided
    if slice_index is None:
        slice_index = data.shape[2] // 2

    # Check if data is 3D or 4D, then extract slices accordingly
    if data.ndim == 3:  # For 3D data
        specific_slice = data[:, :, slice_index]
        img = Image.fromarray(specific_slice)
        img.save(f"{output_folder}/slice_{slice_index:03}.png")
    elif data.ndim == 4:  # For 4D data
        for t in range(data.shape[3]):
            specific_slice = data[:, :, slice_index, t]
            img = Image.fromarray(specific_slice)
            img.save(f"{output_folder}/time_{t:03}_slice_{slice_index:03}.png")

# Define paths
data_dir = '/home/orion/Geo/UCLA data/FMRIPrep/data_3d/'
output_dir = '/home/orion/Geo/UCLA data/FMRIPrep/data_2d/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process SCHZ and HC directories
for group in ['SCHZ', 'HC']:
    group_path = os.path.join(data_dir, group)
    output_group_path = os.path.join(output_dir, group)
    if not os.path.exists(output_group_path):
        os.makedirs(output_group_path)
    for subject in os.listdir(group_path):
        subject_path = os.path.join(group_path, subject)
        output_subject_path = os.path.join(output_group_path, subject)
        if not os.path.exists(output_subject_path):
            os.makedirs(output_subject_path)
        func_path = os.path.join(subject_path, 'func')
        output_func_path = os.path.join(output_subject_path, 'func')
        if not os.path.exists(output_func_path):
            os.makedirs(output_func_path)
        for file_name in os.listdir(func_path):
            if file_name.endswith('.nii.gz'):
                file_path = os.path.join(func_path, file_name)
                nifti_to_specific_slice(file_path, output_func_path)
