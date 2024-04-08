import nibabel as nib
from PIL import Image
import numpy as np
import os


def nifti_to_specific_slice(nifti_file_path, parent_output_folder, slice_index=None):
    # Load the NIfTI file
    nifti = nib.load(nifti_file_path)
    data = nifti.get_fdata()

    # Normalize the data to 0-255
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data) * 255
    data = data.astype(np.uint8)

    # Extract the base name for creating a unique subfolder
    base_name = os.path.splitext(os.path.basename(nifti_file_path))[0]
    output_folder = os.path.join(parent_output_folder, base_name)

    # Ensure the specific output folder for this NIfTI file exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

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


def convert_all_nifti_for_specific_slice(nifti_directory, parent_output_folder, slice_index=None):
    # List all NIfTI files in the given directory
    for file_name in os.listdir(nifti_directory):
        if file_name.endswith('.nii') or file_name.endswith('.nii.gz'):
            nifti_file_path = os.path.join(nifti_directory, file_name)
            nifti_to_specific_slice(nifti_file_path, parent_output_folder, slice_index)

# Example usage: Convert all NIfTI files for a specific slice across all time points
nifti_directory = "/home/orion/Geo/Projects/Computer-Vision-for-Schizophrenia-Detection/test"
parent_output_folder = "home/orion/Geo/Projects/Computer-Vision-for-Schizophrenia-Detection/test/"
convert_all_nifti_for_specific_slice(nifti_directory, parent_output_folder, None)
