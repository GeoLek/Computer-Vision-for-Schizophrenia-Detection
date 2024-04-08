import nibabel as nib
from PIL import Image
import numpy as np
import os

def nifti_to_image_sequence_all_slices(nifti_file_path, parent_output_folder):
    # Load the NIfTI file
    nifti = nib.load(nifti_file_path)
    data = nifti.get_fdata()

    # Normalize the data to 0-255
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data) * 255
    data = data.astype(np.uint8)

    # Extract the base name of the NIfTI file to use in creating a unique subfolder
    base_name = os.path.splitext(os.path.basename(nifti_file_path))[0]
    output_folder = os.path.join(parent_output_folder, base_name)

    # Ensure the specific output folder for this NIfTI file exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Check if the data is 3D or 4D and iterate accordingly
    if data.ndim == 3:  # For 3D data, iterate over each slice in the last dimension
        for z in range(data.shape[2]):
            slice = data[:, :, z]
            img = Image.fromarray(slice)
            img.save(f"{output_folder}/slice_{z:03}.png")
    elif data.ndim == 4:  # For 4D data, iterate over time points and slices
        for t in range(data.shape[3]):
            for z in range(data.shape[2]):
                slice = data[:, :, z, t]
                img = Image.fromarray(slice)
                img.save(f"{output_folder}/time_{t:03}_slice_{z:03}.png")

def convert_all_nifti_in_directory(nifti_directory, parent_output_folder):
    # List all NIfTI files in the given directory
    for file_name in os.listdir(nifti_directory):
        if file_name.endswith('.nii') or file_name.endswith('.nii.gz'):
            nifti_file_path = os.path.join(nifti_directory, file_name)
            nifti_to_image_sequence_all_slices(nifti_file_path, parent_output_folder)


# Example usage: Convert all NIfTI files and save the output in separate folders
nifti_directory = "/home/orion/Geo/Projects/Computer-Vision-for-Schizophrenia-Detection/test"
parent_output_folder = ("/home/orion/Geo/Projects/Computer-Vision-for-Schizophrenia-Detection/test/png")
convert_all_nifti_in_directory(nifti_directory, parent_output_folder)
